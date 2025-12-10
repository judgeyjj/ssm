"""
Heterogeneous Mixture of Experts (MoE) implementation.

使用 RMSNorm 和 Causal Feature Extraction，确保 streaming 一致性。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mamba import CausalConv1d, MambaBlock, RMSNorm


def compute_spectral_entropy(x: torch.Tensor, n_fft: int = 256) -> torch.Tensor:
    """
    Compute a causal proxy for spectral entropy/complexity.
    
    为了保证 Forward 和 Streaming 的严格一致性，我们计算
    "Log Variance over a causal sliding window"。
    这在物理上表征了信号的动态复杂度，与谱熵高度相关，
    且计算是严格因果 O(T) 的。
    """
    B, T, D = x.shape
    
    # 1. Reduce to single channel (energy representative)
    x_mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
    
    # 2. Window size for local statistics
    win_size = min(64, T)
    
    # 3. Compute local variance using simple causal pooling
    # Var(X) = E[X^2] - (E[X])^2
    # Use Average Pooling as a causal moving average
    
    # Pad left to ensure causality
    padding = win_size - 1
    x_mean_pad = F.pad(x_mean.transpose(1, 2), (padding, 0))  # (B, 1, T+pad)
    
    # Moving Average of X
    mu = F.avg_pool1d(x_mean_pad, kernel_size=win_size, stride=1)
    
    # Moving Average of X^2
    mu_sq = F.avg_pool1d(x_mean_pad ** 2, kernel_size=win_size, stride=1)
    
    # Variance = E[X^2] - (E[X])^2
    # Clamp to avoid negative values due to precision
    var = F.relu(mu_sq - mu ** 2)
    
    # Log variance as entropy proxy
    entropy = torch.log(var + 1e-6)
    
    # Restore shape (B, T, 1)
    entropy = entropy.transpose(1, 2)
    
    # Normalize roughly to [0, 1] range for stability
    # Assuming standard normalized input, var is ~1
    entropy = (entropy + 10.0) / 20.0
    
    return entropy


class CausalConvExpert(nn.Module):
    """Causal Convolution Expert for MoE."""
    
    def __init__(self, d_model: int, kernel_size: int = 7, expand: int = 2):
        super().__init__()
        d_hidden = d_model * expand
        self.d_hidden = d_hidden
        self.kernel_size = kernel_size
        
        self.fc_in = nn.Linear(d_model, d_hidden)
        self.act1 = nn.GELU()
        self.conv = CausalConv1d(d_hidden, d_hidden, kernel_size, groups=d_hidden)
        self.act2 = nn.GELU()
        self.fc_out = nn.Linear(d_hidden, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = self.act1(h)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = self.act2(h)
        return self.fc_out(h)
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        
        if state is None:
            conv_buffer = None
        else:
            conv_buffer = state.get('conv_buffer', None)
        
        h = self.fc_in(x)
        h = self.act1(h)
        h = h.transpose(1, 2)
        h, new_conv_buffer = self.conv.forward_with_buffer(h, conv_buffer)
        h = h.transpose(1, 2)
        h = self.act2(h)
        output = self.fc_out(h)
        
        new_state = {'conv_buffer': new_conv_buffer}
        return output, new_state


class MambaExpert(nn.Module):
    """Mamba-based Expert for MoE."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        output, new_state = self.mamba.forward_with_state(x, state)
        return output, new_state


class HeterogeneousMoERouter(nn.Module):
    """Router for Heterogeneous MoE with spectral entropy bias."""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.entropy_proj = nn.Linear(1, num_experts, bias=True)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self, x: torch.Tensor, entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        router_logits = self.gate(x)
        
        # Entropy is now (B, T, 1), fully causal and aligned
        entropy_bias = self.entropy_proj(entropy) # (B, T, Experts)
        
        router_logits = router_logits + entropy_bias
        router_logits = router_logits / (self.temperature.abs() + 1e-6)
        
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return routing_weights, selected_experts, router_logits


class HeterogeneousMoE(nn.Module):
    """Heterogeneous Mixture of Experts layer with RMSNorm."""
    
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
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_conv_experts = num_conv_experts
        self.num_mamba_experts = num_mamba_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList()
        
        for _ in range(num_conv_experts):
            self.experts.append(CausalConvExpert(d_model, kernel_size))
        
        for _ in range(num_mamba_experts):
            self.experts.append(MambaExpert(d_model, d_state))
        
        self.router = HeterogeneousMoERouter(d_model, num_experts, top_k)
        
        # 使用 RMSNorm 替代 LayerNorm
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, _, aux_loss = self.forward_with_state(x, None)
        return output, aux_loss
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[Dict[int, dict]] = None
    ) -> Tuple[torch.Tensor, Dict[int, dict], torch.Tensor]:
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        
        if state is None:
            state = {}
            # Need to maintain buffer for entropy computation in streaming
            entropy_buffer = None
        else:
            entropy_buffer = state.get('entropy_buffer', None)
        
        # Handle Streaming for Entropy Calculation
        if T < 64 and entropy_buffer is not None:
             # Concatenate buffer for causal window computation
             x_for_entropy = torch.cat([entropy_buffer, x], dim=1)
             entropy_full = compute_spectral_entropy(x_for_entropy)
             entropy = entropy_full[:, -T:, :] # Take only the new part
             
             # Update buffer (keep last 63 samples)
             new_entropy_buffer = x_for_entropy[:, -(64-1):, :]
        elif T < 64 and entropy_buffer is None:
             # First chunk, padded automatically inside compute_spectral_entropy
             entropy = compute_spectral_entropy(x)
             new_entropy_buffer = x[:, -(64-1):, :]
        else:
             # Forward mode or Long sequence
             entropy = compute_spectral_entropy(x)
             new_entropy_buffer = None # Not needed/handled for pure forward
        
        routing_weights, selected_experts, router_logits = self.router(x, entropy)
        aux_loss = self._compute_load_balance_loss(router_logits, selected_experts)
        output, new_expert_states = self._apply_experts_with_state(
            x, routing_weights, selected_experts, state
        )
        
        # Merge expert states and entropy buffer
        new_state = new_expert_states
        if new_entropy_buffer is not None:
            new_state['entropy_buffer'] = new_entropy_buffer
            
        return output + residual, new_state, aux_loss
    
    def _apply_experts_with_state(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        state: Dict[int, dict],
    ) -> Tuple[torch.Tensor, Dict[int, dict]]:
        B, T, D = x.shape
        output = torch.zeros_like(x)
        new_state = {}
        
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (selected_experts == expert_idx)
            
            if not expert_mask.any():
                if expert_idx in state:
                    new_state[expert_idx] = state[expert_idx]
                continue
            
            weight_mask = torch.where(
                expert_mask,
                routing_weights,
                torch.zeros_like(routing_weights)
            ).sum(dim=-1)
            
            expert_state = state.get(expert_idx, None)
            expert_output, expert_new_state = expert.forward_with_state(x, expert_state)
            new_state[expert_idx] = expert_new_state
            output = output + expert_output * weight_mask.unsqueeze(-1)
        
        return output, new_state
    
    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        B, T, E = router_logits.shape
        
        expert_mask = F.one_hot(selected_experts, E).float()
        expert_mask = expert_mask.sum(dim=2)
        tokens_per_expert = expert_mask.mean(dim=(0, 1))
        
        router_probs = F.softmax(router_logits, dim=-1)
        avg_router_probs = router_probs.mean(dim=(0, 1))
        
        aux_loss = E * (tokens_per_expert * avg_router_probs).sum()
        
        return aux_loss
