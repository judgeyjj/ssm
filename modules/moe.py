"""
Heterogeneous Mixture of Experts (MoE) implementation.

Combines CausalConv experts (for local patterns) and Mamba experts
(for long-range dependencies) with spectral entropy-biased routing.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mamba import CausalConv1d, MambaBlock


def compute_spectral_entropy(x: torch.Tensor, n_fft: int = 256) -> torch.Tensor:
    """
    Compute spectral entropy of the input signal.
    
    Spectral entropy measures the "flatness" of the spectrum:
    - High entropy: noise-like, broadband signal
    - Low entropy: tonal, narrowband signal
    
    This is used to bias routing decisions in the MoE layer.
    
    Args:
        x: Input tensor of shape (B, T, D).
        n_fft: FFT size for spectral analysis.
        
    Returns:
        Spectral entropy of shape (B, 1), normalized to [0, 1].
    """
    B, T, D = x.shape
    
    # Use a sliding window approach for local spectral entropy
    window_size = min(n_fft, T)
    hop_size = max(1, window_size // 2)
    
    # Unfold to get windows: (B, num_windows, D, window_size)
    if T >= window_size:
        x_windowed = x.unfold(1, window_size, hop_size)
    else:
        # Fallback for very short sequences
        x_windowed = x.unsqueeze(1).transpose(2, 3)  # (B, 1, D, T)
    
    # Compute FFT magnitude spectrum
    spectrum = torch.fft.rfft(x_windowed, dim=-1).abs()
    
    # Normalize to probability distribution
    spectrum = spectrum + 1e-10  # Avoid log(0)
    prob = spectrum / spectrum.sum(dim=-1, keepdim=True)
    
    # Compute entropy: H = -sum(p * log(p))
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
    
    # Normalize by max entropy (log of spectrum size)
    max_entropy = math.log(spectrum.shape[-1])
    entropy = entropy / max_entropy
    
    # Average over all dimensions except batch -> (B, 1)
    if entropy.dim() == 4:
        entropy = entropy.mean(dim=(1, 2, 3))
    elif entropy.dim() == 3:
        entropy = entropy.mean(dim=(1, 2))
    else:
        entropy = entropy.mean(dim=1)
    
    return entropy.unsqueeze(-1)


class CausalConvExpert(nn.Module):
    """
    Causal Convolution Expert for MoE.
    
    Captures local patterns with depth-wise causal convolutions.
    Good for low-entropy (tonal) signals.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 7, expand: int = 2):
        super().__init__()
        d_hidden = d_model * expand
        
        self.fc_in = nn.Linear(d_model, d_hidden)
        self.act1 = nn.GELU()
        self.conv = CausalConv1d(d_hidden, d_hidden, kernel_size, groups=d_hidden)
        self.act2 = nn.GELU()
        self.fc_out = nn.Linear(d_hidden, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D).
        Returns:
            Output tensor of shape (B, T, D).
        """
        h = self.fc_in(x)  # (B, T, d_hidden)
        h = self.act1(h)
        h = h.transpose(1, 2)  # (B, d_hidden, T)
        h = self.conv(h)
        h = h.transpose(1, 2)  # (B, T, d_hidden)
        h = self.act2(h)
        return self.fc_out(h)


class MambaExpert(nn.Module):
    """
    Mamba-based Expert for MoE.
    
    Captures long-range dependencies with selective SSM.
    Good for high-entropy (noise-like) signals.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D).
        Returns:
            Output tensor of shape (B, T, D).
        """
        return self.mamba(x)


class HeterogeneousMoERouter(nn.Module):
    """
    Router for Heterogeneous MoE with spectral entropy bias.
    
    Routes tokens to experts based on learned weights biased by
    the spectral characteristics of the input.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Main routing projection
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Entropy-based bias (learnable scaling)
        self.entropy_proj = nn.Linear(1, num_experts, bias=True)
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self, x: torch.Tensor, entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for each token.
        
        Args:
            x: Input tensor of shape (B, T, D).
            entropy: Spectral entropy of shape (B, 1).
            
        Returns:
            Tuple of (routing_weights, selected_experts, router_logits).
            - routing_weights: (B, T, top_k)
            - selected_experts: (B, T, top_k)
            - router_logits: (B, T, num_experts) for load balancing loss
        """
        B, T, D = x.shape
        
        # Compute base routing logits
        router_logits = self.gate(x)  # (B, T, num_experts)
        
        # Add entropy-based bias
        # High entropy -> bias toward Mamba experts (later indices)
        # Low entropy -> bias toward Conv experts (earlier indices)
        entropy_bias = self.entropy_proj(entropy)  # (B, num_experts)
        entropy_bias = entropy_bias.unsqueeze(1).expand(-1, T, -1)
        
        router_logits = router_logits + entropy_bias
        
        # Apply temperature scaling
        router_logits = router_logits / (self.temperature.abs() + 1e-6)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        
        # Normalize weights (softmax over selected)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return routing_weights, selected_experts, router_logits


class HeterogeneousMoE(nn.Module):
    """
    Heterogeneous Mixture of Experts layer.
    
    Combines CausalConv1d experts (for local patterns) and 
    MambaBlock experts (for long-range dependencies).
    Routing is biased by spectral entropy of the input.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_conv_experts: int = 4,
        num_mamba_experts: int = 4,
        top_k: int = 2,
        kernel_size: int = 7,
        d_state: int = 16,
    ):
        """
        Args:
            d_model: Hidden dimension.
            num_experts: Total number of experts (should equal conv + mamba).
            num_conv_experts: Number of CausalConv experts.
            num_mamba_experts: Number of Mamba experts.
            top_k: Number of experts to route each token to.
            kernel_size: Kernel size for conv experts.
            d_state: State dimension for Mamba experts.
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create heterogeneous experts
        self.experts = nn.ModuleList()
        
        # CausalConv experts (indices 0 to num_conv_experts-1)
        # Good for local patterns, low entropy signals
        for _ in range(num_conv_experts):
            self.experts.append(CausalConvExpert(d_model, kernel_size))
        
        # Mamba experts (indices num_conv_experts to end)
        # Good for long-range dependencies, high entropy signals
        for _ in range(num_mamba_experts):
            self.experts.append(MambaExpert(d_model, d_state))
        
        # Router with entropy bias
        self.router = HeterogeneousMoERouter(d_model, num_experts, top_k)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with expert routing.
        
        Args:
            x: Input tensor of shape (B, T, D).
            
        Returns:
            Tuple of (output, aux_loss).
            - output: Shape (B, T, D)
            - aux_loss: Load balancing auxiliary loss (scalar)
        """
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        
        # Compute spectral entropy for routing bias
        entropy = compute_spectral_entropy(x)  # (B, 1)
        
        # Get routing decisions
        routing_weights, selected_experts, router_logits = self.router(x, entropy)
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_load_balance_loss(router_logits, selected_experts)
        
        # Apply experts with sparse routing
        output = self._apply_experts_sparse(x, routing_weights, selected_experts)
        
        return output + residual, aux_loss
    
    def _apply_experts_sparse(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply selected experts using sparse computation.
        
        This processes each expert only on tokens that are routed to it,
        then combines outputs according to routing weights.
        
        Args:
            x: Input tensor (B, T, D).
            routing_weights: Weights for selected experts (B, T, top_k).
            selected_experts: Indices of selected experts (B, T, top_k).
            
        Returns:
            Combined output tensor (B, T, D).
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Create mask for tokens routed to this expert
            # selected_experts: (B, T, top_k)
            expert_mask = (selected_experts == expert_idx)  # (B, T, top_k)
            
            # Check if any tokens are routed to this expert
            if not expert_mask.any():
                continue
            
            # Get routing weight for this expert where selected
            # routing_weights: (B, T, top_k)
            expert_weight = torch.where(
                expert_mask,
                routing_weights,
                torch.zeros_like(routing_weights)
            ).sum(dim=-1)  # (B, T)
            
            # Find which batch-time positions have this expert
            has_expert = expert_weight > 0  # (B, T)
            
            if not has_expert.any():
                continue
            
            # Process all tokens through expert (could be optimized further)
            expert_output = expert(x)  # (B, T, D)
            
            # Weighted addition
            output = output + expert_output * expert_weight.unsqueeze(-1)
        
        return output
    
    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.
        
        Uses the auxiliary loss from Switch Transformer paper:
        L = num_experts * sum_i(f_i * P_i)
        
        Where:
        - f_i: fraction of tokens dispatched to expert i
        - P_i: average router probability for expert i
        
        Args:
            router_logits: Raw router logits (B, T, E).
            selected_experts: Selected expert indices (B, T, top_k).
            
        Returns:
            Load balance loss (scalar).
        """
        B, T, E = router_logits.shape
        
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(selected_experts, E).float()  # (B, T, top_k, E)
        expert_mask = expert_mask.sum(dim=2)  # (B, T, E)
        tokens_per_expert = expert_mask.mean(dim=(0, 1))  # (E,)
        
        # Router probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)  # (B, T, E)
        avg_router_probs = router_probs.mean(dim=(0, 1))  # (E,)
        
        # Load balance loss
        aux_loss = E * (tokens_per_expert * avg_router_probs).sum()
        
        return aux_loss

