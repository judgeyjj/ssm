"""
Heterogeneous Mixture of Experts (MoE) implementation.

使用 RMSNorm 和修改后的 Spectral Entropy 计算，确保 streaming 一致性。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mamba import CausalConv1d, MambaBlock, RMSNorm


def compute_spectral_entropy(x: torch.Tensor, n_fft: int = 256) -> torch.Tensor:
    """
    Compute spectral entropy of the input signal.
    
    为了 streaming 一致性，使用固定窗口大小计算局部频谱熵，
    不依赖输入长度。
    """
    B, T, D = x.shape
    
    # 使用固定窗口大小，不依赖 T
    window_size = min(n_fft, T)
    
    if T < window_size:
        # 短序列：直接用整个序列
        x_windowed = x.unsqueeze(1)  # (B, 1, T, D)
    else:
        # 只使用最后一个窗口，保证 streaming 一致性
        x_windowed = x[:, -window_size:, :].unsqueeze(1)  # (B, 1, window_size, D)
    
    # FFT along last dimension
    spectrum = torch.fft.rfft(x_windowed, dim=2).abs()
    
    spectrum = spectrum + 1e-10
    prob = spectrum / spectrum.sum(dim=2, keepdim=True)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=2)
    
    max_entropy = math.log(spectrum.shape[2])
    entropy = entropy / max_entropy
    
    # Average over all dimensions except batch
    entropy = entropy.mean(dim=(1, 2))
    
    return entropy.unsqueeze(-1)


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
        entropy_bias = self.entropy_proj(entropy)
        entropy_bias = entropy_bias.unsqueeze(1).expand(-1, T, -1)
        
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
        
        entropy = compute_spectral_entropy(x)
        
        routing_weights, selected_experts, router_logits = self.router(x, entropy)
        aux_loss = self._compute_load_balance_loss(router_logits, selected_experts)
        output, new_state = self._apply_experts_with_state(
            x, routing_weights, selected_experts, state
        )
        
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
