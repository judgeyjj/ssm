"""
FASS-MoE Generator Model.

Main model assembly combining Mamba, MoE (Mixture of Experts), 
and DSG (Dynamic Sparse Gating) blocks for speech super-resolution.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import FASSMoEConfig, ModelConfig
from modules import CausalConv1d, CausalDSG, FASSMoEBlock


class CausalPixelShuffle1d(nn.Module):
    """
    Causal PixelShuffle for 1D signals (audio upsampling).
    
    Upsamples by expanding channels with Conv1d, then reshuffling.
    This avoids checkerboard artifacts from TransposedConv.
    
    For scale_factor=3 (16kHz -> 48kHz):
    Input:  (B, C, T)
    Conv:   (B, C * 3, T)
    Shuffle: (B, C, T * 3)
    """
    
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
        
        # Expand channels
        x = self.conv(x)
        
        # Reshape for pixel shuffle
        x = x.view(B, self.out_channels, self.scale_factor, T)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, self.out_channels, T * self.scale_factor)
        
        return x


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


class Refiner(nn.Module):
    """Refiner module: Final DSG block and projection to output."""
    
    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.dsg = CausalDSG(in_channels, reduction=4, kernel_size=31)
        self.proj = CausalConv1d(in_channels, 1, kernel_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dsg(x)
        x = self.proj(x)
        x = self.tanh(x)
        return x


class FASSMoEGenerator(nn.Module):
    """
    FASS-MoE Speech Super-Resolution Generator.
    
    Architecture: Stem -> MoE Body -> Upsampler -> Refiner
    
    Input:  (B, 1, T_low)  at 16kHz
    Output: (B, 1, T_high) at 48kHz, where T_high = T_low * 3
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
        self.skip_weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_upsampled = self.input_upsample(x)
        
        h = self.stem(x)
        h = h.transpose(1, 2)
        h, aux_loss = self.body(h)
        h = h.transpose(1, 2)
        h = self.upsampler(h)
        output = self.refiner(h)
        
        output = output + torch.sigmoid(self.skip_weight) * input_upsampled
        return output, aux_loss
    
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward(x)
        return output
    
    def infer_stream(
        self,
        chunk: torch.Tensor,
        buffer_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Streaming inference with state management.
        
        Processes audio chunk-by-chunk, maintaining state for
        causal convolution buffers across chunks.
        """
        if buffer_dict is None:
            buffer_dict = {}
        
        B = chunk.shape[0]
        kernel_size = self.config.kernel_size
        
        input_upsampled = self.input_upsample(chunk)
        
        # Stem with buffer
        stem_buf = buffer_dict.get('stem', None)
        if stem_buf is None:
            stem_buf = torch.zeros(B, 1, kernel_size - 1, device=chunk.device, dtype=chunk.dtype)
        
        x_padded = torch.cat([stem_buf, chunk], dim=-1)
        buffer_dict['stem'] = x_padded[:, :, -(kernel_size - 1):]
        h = self.stem.conv.conv(x_padded)
        h = self.stem.norm(h)
        h = self.stem.act(h)
        
        # Body (process as chunk)
        h = h.transpose(1, 2)
        h, _ = self.body(h)
        h = h.transpose(1, 2)
        
        # Upsampler with buffers
        up_bufs = buffer_dict.get('upsampler', {})
        
        pre_buf = up_bufs.get('pre', torch.zeros(B, h.shape[1], kernel_size - 1, device=h.device, dtype=h.dtype))
        h_padded = torch.cat([pre_buf, h], dim=-1)
        up_bufs['pre'] = h_padded[:, :, -(kernel_size - 1):]
        h = self.upsampler.pre_conv.conv(h_padded)
        h = self.upsampler.pre_norm(h)
        h = self.upsampler.pre_act(h)
        
        shuffle_buf = up_bufs.get('shuffle', torch.zeros(B, h.shape[1], kernel_size - 1, device=h.device, dtype=h.dtype))
        h_padded = torch.cat([shuffle_buf, h], dim=-1)
        up_bufs['shuffle'] = h_padded[:, :, -(kernel_size - 1):]
        h = self.upsampler.pixel_shuffle.conv.conv(h_padded)
        
        B_h, _, T_conv = h.shape
        h = h.view(B_h, self.config.hidden_channels, self.scale_factor, T_conv)
        h = h.permute(0, 1, 3, 2)
        h = h.reshape(B_h, self.config.hidden_channels, T_conv * self.scale_factor)
        
        post_buf = up_bufs.get('post', torch.zeros(B, self.config.hidden_channels, kernel_size - 1, device=h.device, dtype=h.dtype))
        h_padded = torch.cat([post_buf, h], dim=-1)
        up_bufs['post'] = h_padded[:, :, -(kernel_size - 1):]
        h = self.upsampler.post_conv.conv(h_padded)
        h = self.upsampler.post_norm(h)
        h = self.upsampler.post_act(h)
        
        buffer_dict['upsampler'] = up_bufs
        
        # Refiner with buffers
        dsg_kernel = 31
        dsg_buf = buffer_dict.get('dsg', torch.zeros(B, h.shape[1], dsg_kernel - 1, device=h.device, dtype=h.dtype))
        h_padded = torch.cat([dsg_buf, h], dim=-1)
        buffer_dict['dsg'] = h_padded[:, :, -(dsg_kernel - 1):]
        
        context = self.refiner.dsg.context_conv.conv(h_padded)
        context_t = context.transpose(1, 2)
        attn = self.refiner.dsg.attention(context_t)
        spectral = self.refiner.dsg.spectral_gate(context_t)
        gate = attn + torch.sigmoid(self.refiner.dsg.mix_coef) * spectral
        gate = gate.transpose(1, 2)
        h = h * gate
        
        proj_buf = buffer_dict.get('proj', torch.zeros(B, h.shape[1], kernel_size - 1, device=h.device, dtype=h.dtype))
        h_padded = torch.cat([proj_buf, h], dim=-1)
        buffer_dict['proj'] = h_padded[:, :, -(kernel_size - 1):]
        output = self.refiner.proj.conv(h_padded)
        output = self.refiner.tanh(output)
        
        output = output + torch.sigmoid(self.skip_weight) * input_upsampled
        
        return output, buffer_dict


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
