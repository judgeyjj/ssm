"""
Mamba (Selective State Space Model) implementation.

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
Supports both parallel (training) and recurrent (streaming) modes.
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


class MambaBlock(nn.Module):
    """
    Mamba block implementing Selective State Space Model (S6).
    
    Supports both parallel (training) and recurrent (streaming) modes.
    
    The selective SSM allows input-dependent state transitions, enabling
    the model to selectively remember or forget information based on content.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        """
        Args:
            d_model: Input/output dimension.
            d_state: SSM state dimension (N).
            d_conv: Local convolution width.
            expand: Expansion factor for inner dimension.
        """
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
        # Projects to (dt, B, C) - delta, B matrix, C matrix
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Delta (dt) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias for stable training
        dt_init_std = 0.001
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        # A parameter (state transition) - learned log values for stability
        # Initialize A as negative values (for stability in continuous form)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel forward pass for training.
        
        Args:
            x: Input tensor of shape (B, T, D).
            
        Returns:
            Output tensor of shape (B, T, D).
        """
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, T, d_inner)
        
        # Causal convolution
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)
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
        
        # Run selective SSM
        y = self._selective_ssm_parallel(x_conv, dt, A, B_param, C_param)
        
        # Skip connection with D
        y = y + x_conv * self.D
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        return y + residual
    
    def _selective_ssm_parallel(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selective SSM computation using parallel associative scan.
        
        This implements the discretized SSM:
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C_t @ h_t
        
        Where A_bar = exp(dt * A) and B_bar = dt * B.
        
        Args:
            x: Input (B, T, d_inner).
            dt: Time delta (B, T, d_inner).
            A: State transition (d_inner, d_state).
            B: Input matrix (B, T, d_state).
            C: Output matrix (B, T, d_state).
            
        Returns:
            Output (B, T, d_inner).
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretization: A_bar = exp(dt * A)
        dt_expanded = dt.unsqueeze(-1)  # (B, T, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        
        A_bar = torch.exp(dt_expanded * A_expanded)  # (B, T, d_inner, d_state)
        
        # B_bar = dt * B
        B_expanded = B.unsqueeze(2)  # (B, T, 1, d_state)
        B_bar = dt_expanded * B_expanded  # (B, T, d_inner, d_state)
        
        # Input contribution
        x_expanded = x.unsqueeze(-1)  # (B, T, d_inner, 1)
        
        # Parallel associative scan for computing h_t
        # Using the associative operator: (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)
        # where a = A_bar, b = B_bar * x
        
        # For efficiency, we use a sequential implementation here
        # A true parallel scan would use custom CUDA kernels
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x_expanded[:, t]
            # y_t = sum_n(C_t[n] * h_t[n]) for each d_inner
            y_t = torch.einsum("bdn,btn->bd", h, C[:, t:t+1].expand(-1, -1, d_state))
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (B, T, d_inner)
    
    def step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Recurrent step for streaming inference.
        
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
        xz = self.in_proj(x_normed)  # (B, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, d_inner)
        
        # Update conv state and apply convolution
        conv_state = torch.cat([conv_state, x_proj.unsqueeze(-1)], dim=-1)
        x_conv = (conv_state * self.conv1d.conv.weight.squeeze(1)).sum(dim=-1)
        if self.conv1d.conv.bias is not None:
            x_conv = x_conv + self.conv1d.conv.bias
        conv_state = conv_state[:, :, 1:]  # Shift state
        
        x_conv = F.silu(x_conv)
        
        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, d_state * 2 + 1)
        dt, B_param, C_param = torch.split(
            ssm_params, [1, self.d_state, self.d_state], dim=-1
        )
        
        # Delta transformation
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # (B, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B_param.unsqueeze(1)  # (B, d_inner, d_state)
        
        # SSM step: h = A_bar * h + B_bar * x
        h = A_bar * h + B_bar * x_conv.unsqueeze(-1)
        
        # Output: y = C @ h
        y = torch.einsum("bdn,bn->bd", h, C_param)
        
        # Skip connection
        y = y + x_conv * self.D
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        return y, (h, conv_state)

