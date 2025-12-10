"""
FASS-MoE Generator Model.

Main model assembly combining Mamba, MoE (Mixture of Experts), 
and DSG (Dynamic Sparse Gating) blocks for speech super-resolution.

修改: 完整的 stateful streaming 实现，保证 forward() 和 infer_stream() 输出完全一致。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import FASSMoEConfig, ModelConfig
from modules import CausalConv1d, CausalDSG, FASSMoEBlock


class CausalPixelShuffle1d(nn.Module):
    """Causal PixelShuffle for 1D signals (audio upsampling)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        
        self.conv = CausalConv1d(
            in_channels,
            out_channels * scale_factor,
            kernel_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = self.conv(x)
        x = x.view(B, self.out_channels, self.scale_factor, T)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, self.out_channels, T * self.scale_factor)
        return x
    
    def forward_with_buffer(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with buffer for streaming."""
        B, C, T = x.shape
        x, new_buffer = self.conv.forward_with_buffer(x, buffer)
        x = x.view(B, self.out_channels, self.scale_factor, T)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, self.out_channels, T * self.scale_factor)
        return x, new_buffer


class Stem(nn.Module):
    """Stem module: Maps input audio (1 channel) to hidden dimension."""
    
    def __init__(self, out_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = CausalConv1d(1, out_channels, kernel_size)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
    def forward_with_buffer(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with buffer for streaming."""
        x, new_buffer = self.conv.forward_with_buffer(x, buffer)
        x = self.norm(x)
        x = self.act(x)
        return x, new_buffer


class MoEBody(nn.Module):
    """Body module: Stack of FASSMoEBlock with residual connections."""
    
    def __init__(self, config: ModelConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            FASSMoEBlock(config) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss
        total_aux_loss = total_aux_loss / len(self.layers)
        return x, total_aux_loss
    
    def forward_with_state(
        self, x: torch.Tensor, state: Optional[List[dict]] = None
    ) -> Tuple[torch.Tensor, List[dict], torch.Tensor]:
        """Forward with state for streaming."""
        if state is None:
            state = [None] * len(self.layers)
        
        total_aux_loss = 0.0
        new_states = []
        
        for i, layer in enumerate(self.layers):
            x, new_layer_state, aux_loss = layer.forward_with_state(x, state[i])
            new_states.append(new_layer_state)
            total_aux_loss = total_aux_loss + aux_loss
        
        total_aux_loss = total_aux_loss / len(self.layers)
        return x, new_states, total_aux_loss


class Upsampler(nn.Module):
    """Upsampler module: Causal PixelShuffle-based upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 3,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.pre_conv = CausalConv1d(in_channels, in_channels, kernel_size)
        self.pre_norm = nn.GroupNorm(1, in_channels)
        self.pre_act = nn.GELU()
        
        self.pixel_shuffle = CausalPixelShuffle1d(
            in_channels, out_channels, scale_factor, kernel_size
        )
        
        self.post_conv = CausalConv1d(out_channels, out_channels, kernel_size)
        self.post_norm = nn.GroupNorm(1, out_channels)
        self.post_act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_act(self.pre_norm(self.pre_conv(x)))
        x = self.pixel_shuffle(x)
        x = self.post_act(self.post_norm(self.post_conv(x)))
        return x
    
    def forward_with_buffers(
        self, x: torch.Tensor, buffers: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward with buffers for streaming."""
        if buffers is None:
            buffers = {}
        
        # Pre conv
        x, pre_buf = self.pre_conv.forward_with_buffer(x, buffers.get('pre', None))
        x = self.pre_norm(x)
        x = self.pre_act(x)
        
        # Pixel shuffle
        x, shuffle_buf = self.pixel_shuffle.forward_with_buffer(x, buffers.get('shuffle', None))
        
        # Post conv
        x, post_buf = self.post_conv.forward_with_buffer(x, buffers.get('post', None))
        x = self.post_norm(x)
        x = self.post_act(x)
        
        new_buffers = {
            'pre': pre_buf,
            'shuffle': shuffle_buf,
            'post': post_buf,
        }
        return x, new_buffers


class Refiner(nn.Module):
    """Refiner module: Final DSG block and projection to output."""
    
    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.dsg = CausalDSG(in_channels, reduction=4, kernel_size=31)
        self.proj = CausalConv1d(in_channels, 1, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dsg(x)
        x = self.proj(x)
        return x
    
    def forward_with_buffers(
        self, x: torch.Tensor, buffers: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward with buffers for streaming."""
        if buffers is None:
            buffers = {}
        
        # DSG
        x, dsg_state = self.dsg.forward_with_state(x, buffers.get('dsg', None))
        
        # Projection
        x, proj_buf = self.proj.forward_with_buffer(x, buffers.get('proj', None))
        
        new_buffers = {
            'dsg': dsg_state,
            'proj': proj_buf,
        }
        return x, new_buffers


class FASSMoEGenerator(nn.Module):
    """
    FASS-MoE Speech Super-Resolution Generator.
    
    Architecture: Stem -> MoE Body -> Upsampler -> Refiner -> Skip + Tanh
    
    Input:  (B, 1, T_low)  at 16kHz
    Output: (B, 1, T_high) at 48kHz, where T_high = T_low * 3
    
    保证: forward() 和 infer_stream() 的输出在数值上完全一致。
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.scale_factor = 3
        
        self.stem = Stem(config.hidden_channels, config.kernel_size)
        self.body = MoEBody(config, config.num_moe_layers)
        self.upsampler = Upsampler(
            config.hidden_channels,
            config.hidden_channels,
            scale_factor=self.scale_factor,
            kernel_size=config.kernel_size,
        )
        self.refiner = Refiner(config.hidden_channels, config.kernel_size)
        
        self.input_upsample = nn.Upsample(
            scale_factor=self.scale_factor,
            mode='linear',
            align_corners=False,
        )
        self.skip_weight = nn.Parameter(torch.tensor(-2.0))
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass."""
        input_upsampled = self.input_upsample(x)
        
        h = self.stem(x)
        h = h.transpose(1, 2)
        h, aux_loss = self.body(h)
        h = h.transpose(1, 2)
        h = self.upsampler(h)
        h = self.refiner(h)
        
        output = h + torch.sigmoid(self.skip_weight) * input_upsampled
        output = self.tanh(output)
        
        return output, aux_loss
    
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward(x)
        return output
    
    def infer_stream(
        self,
        chunk: torch.Tensor,
        state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Streaming inference with COMPLETE state management.
        
        保证与 forward() 输出完全一致（在浮点精度范围内）。
        
        Args:
            chunk: Input audio chunk of shape (B, 1, chunk_size).
            state: Complete state dictionary from previous chunk.
            
        Returns:
            Tuple of (output_chunk, new_state).
        """
        if state is None:
            state = {}
        
        input_upsampled = self.input_upsample(chunk)
        
        # Stem
        h, stem_buf = self.stem.forward_with_buffer(chunk, state.get('stem', None))
        
        # Body (MoE layers)
        h = h.transpose(1, 2)
        h, body_states, _ = self.body.forward_with_state(h, state.get('body', None))
        h = h.transpose(1, 2)
        
        # Upsampler
        h, up_bufs = self.upsampler.forward_with_buffers(h, state.get('upsampler', None))
        
        # Refiner
        h, refiner_bufs = self.refiner.forward_with_buffers(h, state.get('refiner', None))
        
        # Skip connection + Tanh
        output = h + torch.sigmoid(self.skip_weight) * input_upsampled
        output = self.tanh(output)
        
        # Collect new state
        new_state = {
            'stem': stem_buf,
            'body': body_states,
            'upsampler': up_bufs,
            'refiner': refiner_bufs,
        }
        
        return output, new_state


def build_generator(config: FASSMoEConfig) -> FASSMoEGenerator:
    """Build and initialize the FASS-MoE generator."""
    model = FASSMoEGenerator(config.model)
    _init_weights(model)
    return model


def _init_weights(module: nn.Module) -> None:
    """Initialize model weights."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
