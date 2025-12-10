"""
Dynamic Spectral Gating (DSG) implementation.

Replaces standard Squeeze-and-Excitation with causal context aggregation
for streaming-compatible channel attention.
"""

from typing import Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from modules.mamba import CausalConv1d
from modules.moe import HeterogeneousMoE


class CausalDSG(nn.Module):
    """
    Causal Dynamic Spectral Gating (DSG) block.
    
    Replaces standard Squeeze-and-Excitation with causal context aggregation.
    Uses CausalConv1d instead of global pooling to maintain causality for streaming.
    
    Features:
    - Causal context aggregation via depth-wise convolution
    - Channel-wise attention (excitation)
    - Learnable spectral modulation
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        kernel_size: int = 31,
    ):
        """
        Args:
            channels: Number of input/output channels.
            reduction: Reduction ratio for bottleneck.
            kernel_size: Kernel size for causal context aggregation.
        """
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction, 8)
        
        # Causal context aggregation (replaces global pooling)
        self.context_conv = CausalConv1d(
            channels, channels, kernel_size, groups=channels
        )
        
        # Channel attention with bottleneck
        self.attention = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )
        
        # Spectral modulation branch
        self.spectral_gate = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Tanh(),
        )
        
        # Learnable mixing coefficient
        self.mix_coef = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal dynamic spectral gating.
        
        Args:
            x: Input tensor of shape (B, C, T) or (B, T, C).
            
        Returns:
            Gated output with same shape as input.
        """
        # Detect input format
        input_format = "BCT" if x.dim() == 3 and x.shape[1] == self.channels else "BTC"
        
        if input_format == "BTC":
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        B, C, T = x.shape
        
        # Causal context aggregation
        context = self.context_conv(x)  # (B, C, T)
        
        # Transpose for linear layers: (B, C, T) -> (B, T, C)
        context = context.transpose(1, 2)
        
        # Channel attention (excitation)
        attn = self.attention(context)  # (B, T, C)
        
        # Spectral gating
        spectral = self.spectral_gate(context)  # (B, T, C)
        
        # Combine attention and spectral gating with learnable mix
        gate = attn + torch.sigmoid(self.mix_coef) * spectral
        
        # Apply gating
        gate = gate.transpose(1, 2)  # (B, C, T)
        output = x * gate
        
        if input_format == "BTC":
            output = output.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        
        return output


class DSGModule(nn.Module):
    """
    Full DSG module with residual connection and normalization.
    
    Wraps CausalDSG with pre-normalization and residual connection
    for stable training in deep networks.
    """
    
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
        """
        Args:
            x: Input tensor of shape (B, T, C).
        Returns:
            Output tensor of shape (B, T, C).
        """
        residual = x
        x = self.norm(x)
        x = self.dsg(x)
        return x + residual


class FASSMoEBlock(nn.Module):
    """
    Combined FASS-MoE block: MoE + DSG.
    
    A single transformer-like block combining heterogeneous MoE
    with causal dynamic spectral gating.
    
    Architecture:
        x -> MoE (with residual) -> DSG (with residual) -> output
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Heterogeneous MoE layer
        self.moe = HeterogeneousMoE(
            d_model=config.hidden_channels,
            num_experts=config.num_experts,
            num_conv_experts=config.num_experts // 2,
            num_mamba_experts=config.num_experts // 2,
            top_k=config.num_experts_per_tok,
            kernel_size=config.kernel_size,
            d_state=config.mamba_d_state,
        )
        
        # Causal DSG
        self.dsg = DSGModule(
            channels=config.hidden_channels,
            reduction=4,
            kernel_size=31,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE and DSG.
        
        Args:
            x: Input tensor of shape (B, T, D).
            
        Returns:
            Tuple of (output, aux_loss).
            - output: Processed tensor (B, T, D)
            - aux_loss: MoE load balancing loss
        """
        # MoE layer (includes residual)
        x, aux_loss = self.moe(x)
        
        # DSG layer (includes residual)
        x = self.dsg(x)
        
        return x, aux_loss

