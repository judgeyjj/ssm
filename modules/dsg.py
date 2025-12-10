"""
Dynamic Spectral Gating (DSG) implementation.

Replaces standard Squeeze-and-Excitation with causal context aggregation
for streaming-compatible channel attention.

修改: 支持 stateful forward，用于 streaming 推理时保持状态连续性。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from modules.mamba import CausalConv1d
from modules.moe import HeterogeneousMoE


class CausalDSG(nn.Module):
    """
    Causal Dynamic Spectral Gating (DSG) block.
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        reduced_channels = max(channels // reduction, 8)
        
        self.context_conv = CausalConv1d(
            channels, channels, kernel_size, groups=channels
        )
        
        self.attention = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )
        
        self.spectral_gate = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Tanh(),
        )
        
        self.mix_coef = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_state(x, None)
        return output
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward with state for streaming."""
        input_format = "BCT" if x.dim() == 3 and x.shape[1] == self.channels else "BTC"
        
        if input_format == "BTC":
            x = x.transpose(1, 2)
        
        B, C, T = x.shape
        
        # Get buffer
        if state is None:
            conv_buffer = None
        else:
            conv_buffer = state.get('conv_buffer', None)
        
        # Causal context aggregation with buffer
        context, new_conv_buffer = self.context_conv.forward_with_buffer(x, conv_buffer)
        
        context = context.transpose(1, 2)
        attn = self.attention(context)
        spectral = self.spectral_gate(context)
        gate = attn + torch.sigmoid(self.mix_coef) * spectral
        gate = gate.transpose(1, 2)
        output = x * gate
        
        if input_format == "BTC":
            output = output.transpose(1, 2)
        
        new_state = {'conv_buffer': new_conv_buffer}
        return output, new_state


class DSGModule(nn.Module):
    """Full DSG module with residual connection and normalization."""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.dsg = CausalDSG(channels, reduction, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_state(x, None)
        return output
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward with state for streaming."""
        residual = x
        x = self.norm(x)
        x, new_state = self.dsg.forward_with_state(x, state)
        return x + residual, new_state


class FASSMoEBlock(nn.Module):
    """
    Combined FASS-MoE block: MoE + DSG.
    
    支持 forward_with_state 用于精确的流式推理。
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.moe = HeterogeneousMoE(
            d_model=config.hidden_channels,
            num_experts=config.num_experts,
            num_conv_experts=config.num_experts // 2,
            num_mamba_experts=config.num_experts // 2,
            top_k=config.num_experts_per_tok,
            kernel_size=config.kernel_size,
            d_state=config.mamba_d_state,
        )
        
        self.dsg = DSGModule(
            channels=config.hidden_channels,
            reduction=4,
            kernel_size=31,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, aux_loss = self.moe(x)
        x = self.dsg(x)
        return x, aux_loss
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Forward with state for streaming.
        
        Args:
            x: Input tensor (B, T, D).
            state: Dictionary containing:
                - 'moe': MoE expert states
                - 'dsg': DSG conv buffer
                
        Returns:
            Tuple of (output, new_state, aux_loss).
        """
        if state is None:
            moe_state = None
            dsg_state = None
        else:
            moe_state = state.get('moe', None)
            dsg_state = state.get('dsg', None)
        
        # MoE layer
        x, new_moe_state, aux_loss = self.moe.forward_with_state(x, moe_state)
        
        # DSG layer
        x, new_dsg_state = self.dsg.forward_with_state(x, dsg_state)
        
        new_state = {
            'moe': new_moe_state,
            'dsg': new_dsg_state,
        }
        
        return x, new_state, aux_loss
