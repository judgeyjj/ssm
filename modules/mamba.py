"""
Mamba (Selective State Space Model) implementation.

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
Supports both parallel (training) and recurrent (streaming) modes.

修改: 
1. 集成 mamba_ssm 的 CUDA Kernel (selective_scan_fn) 用于加速训练
2. 保持 Python 实现作为 Fallback 和推理使用
3. 保持 RMSNorm 和 State Management 的一致性
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing optimization kernels from mamba_ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_KERNEL = True
    print("🚀 [Mamba] Detected mamba_ssm kernel. Training will be fast.")
except ImportError:
    HAS_MAMBA_KERNEL = False
    print("⚠️ [Mamba] mamba_ssm kernel not found. Using slow Python fallback.")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that only looks at past context.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)
    
    def forward_with_buffer(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward with explicit buffer management."""
        B, C, T = x.shape
        
        if buffer is None:
            buffer = torch.zeros(B, C, self.padding, device=x.device, dtype=x.dtype)
        
        x_padded = torch.cat([buffer, x], dim=-1)
        new_buffer = x_padded[:, :, -self.padding:].clone() if self.padding > 0 else buffer
        output = self.conv(x_padded)
        
        return output, new_buffer


class MambaBlock(nn.Module):
    """
    Mamba block implementing Selective State Space Model (S6).
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = CausalConv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        dt_init_std = 0.001
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        # Initialize A as in standard Mamba
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_state(x, None)
        return output
    
    def forward_with_state(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass with explicit state management."""
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        
        if state is None:
            ssm_h = None
            conv_buffer = None
        else:
            ssm_h = state.get('ssm_h', None)
            conv_buffer = state.get('conv_buffer', None)
        
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_conv = x_proj.transpose(1, 2)
        x_conv, new_conv_buffer = self.conv1d.forward_with_buffer(x_conv, conv_buffer)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        ssm_params = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            ssm_params, [1, self.d_state, self.d_state], dim=-1
        )
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        # -----------------------------------------------------------
        # Use CUDA Kernel if available and no initial state (Training)
        # -----------------------------------------------------------
        if HAS_MAMBA_KERNEL and ssm_h is None:
            # selective_scan_fn expects:
            # u: (B, D, L)
            # delta: (B, D, L)
            # A: (D, N)
            # B: (B, N, L)
            # C: (B, N, L)
            # D: (D)
            # z: (B, D, L)  <-- Optional, we handle gating separately usually, 
            #                   but `selective_scan_fn` can do it.
            #                   Here we keep it simple and match Python logic.
            
            # Prepare inputs for kernel
            u = x_conv.transpose(1, 2)                  # (B, D, L)
            delta = dt.transpose(1, 2)                  # (B, D, L)
            A_in = A                                    # (D, N)
            B_in = B_param.transpose(1, 2)              # (B, N, L)
            C_in = C_param.transpose(1, 2)              # (B, N, L)
            D_in = self.D                               # (D)
            
            # Run kernel
            # returns (B, D, L)
            y = selective_scan_fn(
                u, delta, A_in, B_in, C_in, D_in,
                z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False
            )
            
            y = y.transpose(1, 2) # (B, L, D)
            new_ssm_h = None      # Kernel doesn't easily return state in this mode
            
        else:
            # Use Python Fallback (Slower, but works for Streaming/CPU)
            y, new_ssm_h = self._selective_ssm_stateful(x_conv, dt, A, B_param, C_param, ssm_h)
            y = y + x_conv * self.D
            
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        new_state = {
            'ssm_h': new_ssm_h,
            'conv_buffer': new_conv_buffer,
        }
        
        return y + residual, new_state
    
    def _selective_ssm_stateful(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        initial_h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Python implementation of Selective SSM."""
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        if initial_h is None:
            h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        else:
            h = initial_h
        
        dt_expanded = dt.unsqueeze(-1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(dt_expanded * A_expanded)
        
        B_expanded = B.unsqueeze(2)
        B_bar = dt_expanded * B_expanded
        
        x_expanded = x.unsqueeze(-1)
        
        outputs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t] * x_expanded[:, t]
            y_t = torch.einsum("bdn,btn->bd", h, C[:, t:t+1].expand(-1, -1, d_state))
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        final_h = h
        
        return output, final_h
    
    def step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Recurrent step for single-timestep streaming inference."""
        B, D = x.shape
        
        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            conv_state = torch.zeros(B, self.d_inner, self.d_conv - 1, device=x.device, dtype=x.dtype)
        else:
            h, conv_state = state
        
        x_normed = self.norm(x)
        xz = self.in_proj(x_normed)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Conv Step
        conv_state = torch.cat([conv_state, x_proj.unsqueeze(-1)], dim=-1)
        x_conv = (conv_state * self.conv1d.conv.weight.squeeze(1)).sum(dim=-1)
        if self.conv1d.conv.bias is not None:
            x_conv = x_conv + self.conv1d.conv.bias
        conv_state = conv_state[:, :, 1:]
        x_conv = F.silu(x_conv)
        
        # SSM Step
        ssm_params = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            ssm_params, [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = dt.unsqueeze(-1) * B_param.unsqueeze(1)
        
        h = A_bar * h + B_bar * x_conv.unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", h, C_param)
        
        y = y + x_conv * self.D
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        return y, (h, conv_state)
