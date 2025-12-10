"""
Mamba (Selective State Space Model) implementation.

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
Supports both parallel (training) and recurrent (streaming) modes.

修改: 支持 stateful forward，用于 streaming 推理时保持状态连续性。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that only looks at past context.
    
    Applies left-padding to ensure output[t] only depends on input[:t+1].
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
        """
        Args:
            x: Input tensor of shape (B, C, T).
        Returns:
            Output tensor of shape (B, C_out, T).
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)
    
    def forward_with_buffer(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming forward with explicit buffer management.
        
        Args:
            x: Input tensor of shape (B, C, T).
            buffer: Previous samples buffer of shape (B, C, kernel_size - 1).
            
        Returns:
            Tuple of (output, new_buffer).
        """
        B, C, T = x.shape
        
        if buffer is None:
            buffer = torch.zeros(B, C, self.padding, device=x.device, dtype=x.dtype)
        
        # Concatenate buffer with input
        x_padded = torch.cat([buffer, x], dim=-1)
        
        # Update buffer for next chunk
        new_buffer = x_padded[:, :, -self.padding:].clone() if self.padding > 0 else buffer
        
        # Apply convolution
        output = self.conv(x_padded)
        
        return output, new_buffer


class MambaBlock(nn.Module):
    """
    Mamba block implementing Selective State Space Model (S6).
    
    Supports:
    - forward(): 标准并行模式，用于训练
    - forward_with_state(): 带状态的前向传播，用于精确流式推理
    - step(): 单步递归模式，用于最精确的流式推理
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
        
        # Input projection: x -> (z, x_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal conv for local context
        self.conv1d = CausalConv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
        )
        
        # SSM parameters projection
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Delta (dt) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias for stable training
        dt_init_std = 0.001
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        # A parameter (state transition)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard parallel forward pass for training.
        
        Args:
            x: Input tensor of shape (B, T, D).
            
        Returns:
            Output tensor of shape (B, T, D).
        """
        output, _ = self.forward_with_state(x, None)
        return output
    
    def forward_with_state(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with explicit state management for streaming.
        
        Args:
            x: Input tensor of shape (B, T, D).
            state: Dictionary containing:
                - 'ssm_h': SSM hidden state (B, d_inner, d_state)
                - 'conv_buffer': Conv1d buffer (B, d_inner, d_conv - 1)
            
        Returns:
            Tuple of (output, new_state).
        """
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        
        # Parse state
        if state is None:
            ssm_h = None
            conv_buffer = None
        else:
            ssm_h = state.get('ssm_h', None)
            conv_buffer = state.get('conv_buffer', None)
        
        # Input projection
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, T, d_inner)
        
        # Causal convolution with buffer
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv, new_conv_buffer = self.conv1d.forward_with_buffer(x_conv, conv_buffer)
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)
        
        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, T, d_state * 2 + 1)
        dt, B_param, C_param = torch.split(
            ssm_params, [1, self.d_state, self.d_state], dim=-1
        )
        
        # Delta transformation
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Run selective SSM with initial state
        y, new_ssm_h = self._selective_ssm_stateful(x_conv, dt, A, B_param, C_param, ssm_h)
        
        # Skip connection with D
        y = y + x_conv * self.D
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        # New state
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
        """
        Selective SSM computation with initial state support.
        
        Args:
            x: Input (B, T, d_inner).
            dt: Time delta (B, T, d_inner).
            A: State transition (d_inner, d_state).
            B: Input matrix (B, T, d_state).
            C: Output matrix (B, T, d_state).
            initial_h: Initial hidden state (B, d_inner, d_state), or None for zeros.
            
        Returns:
            Tuple of:
            - output: (B, T, d_inner)
            - final_h: (B, d_inner, d_state) - 用于下一个 chunk
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize hidden state
        if initial_h is None:
            h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        else:
            h = initial_h
        
        # Discretization
        dt_expanded = dt.unsqueeze(-1)  # (B, T, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        
        A_bar = torch.exp(dt_expanded * A_expanded)  # (B, T, d_inner, d_state)
        
        B_expanded = B.unsqueeze(2)  # (B, T, 1, d_state)
        B_bar = dt_expanded * B_expanded  # (B, T, d_inner, d_state)
        
        x_expanded = x.unsqueeze(-1)  # (B, T, d_inner, 1)
        
        # Sequential scan (保留状态)
        outputs = []
        for t in range(seq_len):
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x_expanded[:, t]
            # y_t = sum_n(C_t[n] * h_t[n])
            y_t = torch.einsum("bdn,btn->bd", h, C[:, t:t+1].expand(-1, -1, d_state))
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        final_h = h  # (B, d_inner, d_state)
        
        return output, final_h
    
    def step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Recurrent step for single-timestep streaming inference.
        
        Args:
            x: Input tensor of shape (B, D) - single timestep.
            state: Tuple of (h, conv_state) from previous step.
            
        Returns:
            Tuple of (output, new_state).
        """
        B, D = x.shape
        
        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            conv_state = torch.zeros(B, self.d_inner, self.d_conv - 1, device=x.device, dtype=x.dtype)
        else:
            h, conv_state = state
        
        x_normed = self.norm(x)
        
        # Input projection
        xz = self.in_proj(x_normed)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Update conv state
        conv_state = torch.cat([conv_state, x_proj.unsqueeze(-1)], dim=-1)
        x_conv = (conv_state * self.conv1d.conv.weight.squeeze(1)).sum(dim=-1)
        if self.conv1d.conv.bias is not None:
            x_conv = x_conv + self.conv1d.conv.bias
        conv_state = conv_state[:, :, 1:]
        
        x_conv = F.silu(x_conv)
        
        # SSM parameters
        ssm_params = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            ssm_params, [1, self.d_state, self.d_state], dim=-1
        )
        
        # Delta transformation
        dt = F.softplus(self.dt_proj(dt))
        
        # Discretize A
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = dt.unsqueeze(-1) * B_param.unsqueeze(1)
        
        # SSM step
        h = A_bar * h + B_bar * x_conv.unsqueeze(-1)
        
        # Output
        y = torch.einsum("bdn,bn->bd", h, C_param)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        return y, (h, conv_state)
