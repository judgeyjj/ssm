"""
Normalization layers for streaming-compatible processing.

使用 RMSNorm 替代 LayerNorm，避免 streaming 和 forward 的不一致。
RMSNorm 不依赖均值，只依赖 RMS，对序列长度更稳定。
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    与 LayerNorm 相比：
    - LayerNorm: (x - mean) / std * gamma + beta
    - RMSNorm: x / rms * gamma
    
    RMSNorm 不减去均值，因此对序列长度更稳定，
    更适合 streaming 场景。
    
    参考: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim).
        Returns:
            Normalized tensor of same shape.
        """
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class StreamingGroupNorm(nn.Module):
    """
    Streaming-compatible Group Normalization.
    
    在 streaming 模式下，使用 Instance Norm 的行为（每个样本独立归一化），
    避免依赖 batch 或 sequence 长度。
    
    对于 (B, C, T) 输入，在 C 维度分组，在 T 维度上归一化。
    """
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T).
        Returns:
            Normalized tensor of same shape.
        """
        B, C, T = x.shape
        
        # Reshape to (B, G, C//G, T)
        x = x.view(B, self.num_groups, C // self.num_groups, T)
        
        # Compute mean and var over (C//G, T) for each group
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x = x.view(B, C, T)
        
        # Apply learnable parameters
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        return x


class CausalRMSNorm(nn.Module):
    """
    Causal RMS Normalization with running statistics.
    
    为了实现严格的 streaming 一致性，使用因果的 running RMS。
    每个时间步的归一化只依赖于当前及之前的数据。
    
    但注意：这会改变模型的行为，需要重新训练。
    """
    
    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
        # Running statistics (not trained)
        self.register_buffer('running_sq_mean', torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (non-causal, for training)."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
    
    def forward_causal(
        self, x: torch.Tensor, running_sq: torch.Tensor = None
    ) -> tuple:
        """
        Causal forward with running statistics.
        
        Args:
            x: Input of shape (B, T, D) or (B, D).
            running_sq: Running squared mean from previous step.
            
        Returns:
            Tuple of (output, new_running_sq).
        """
        if running_sq is None:
            running_sq = torch.ones(x.shape[0], self.dim, device=x.device, dtype=x.dtype)
        
        if x.dim() == 2:
            # Single timestep: (B, D)
            sq = x ** 2
            running_sq = (1 - self.momentum) * running_sq + self.momentum * sq
            rms = torch.sqrt(running_sq + self.eps)
            output = (x / rms) * self.weight
        else:
            # Sequence: (B, T, D)
            B, T, D = x.shape
            outputs = []
            for t in range(T):
                sq = x[:, t, :] ** 2
                running_sq = (1 - self.momentum) * running_sq + self.momentum * sq
                rms = torch.sqrt(running_sq + self.eps)
                out_t = (x[:, t, :] / rms) * self.weight
                outputs.append(out_t)
            output = torch.stack(outputs, dim=1)
        
        return output, running_sq

