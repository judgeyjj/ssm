"""
Configuration for FASS-MoE Speech Super-Resolution.

Includes configurations for:
- Audio processing
- Model architecture (FASS-MoE)
- Discriminator
- Training hyperparameters
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    input_sr: int = 16000
    target_sr: int = 48000
    # Important: segment_length must be divisible by (target_sr / input_sr)
    # 16200 / (48000/16000) = 16200 / 3 = 5400 (Integer)
    segment_length: int = 16200 
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_channels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0  # Nyquist of input


@dataclass
class ModelConfig:
    """
    FASS-MoE Generator configuration.
    
    Target: ~5M - 8M parameters for SOTA performance.
    """
    hidden_channels: int = 96      # 增加通道数 (64 -> 96)
    num_moe_layers: int = 8        # 增加层数 (4 -> 8)
    num_experts: int = 8           # 保持专家数量
    num_experts_per_tok: int = 2   # Top-2 routing
    kernel_size: int = 7
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    dropout: float = 0.1


@dataclass
class DiscriminatorConfig:
    """Projected GAN Discriminator configuration."""
    vit_name: str = "vit_base_patch16_224"  # Uses timm if available
    pretrained: bool = True
    img_size: Tuple[int, int] = (224, 224)
    # Projections from specific ViT layers
    interp_types: List[str] = field(default_factory=lambda: ["pool", "pool", "pool", "pool"])
    # Features from layers 3, 6, 9, 12
    feature_levels: List[int] = field(default_factory=lambda: [3, 6, 9, 12])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 16           # 适当减小 Batch Size 以适应更大的模型
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    decay: float = 0.999
    num_epochs: int = 200          # 增加训练轮数
    warmup_epochs: int = 5
    num_workers: int = 4           # Data loader workers
    checkpoint_interval: int = 5
    log_interval: int = 100
    val_interval: int = 1
    grad_clip: float = 10.0       # 梯度裁剪
    lambda_mr_stft: float = 1.0   # Reconstruction loss weight
    lambda_fm: float = 2.0        # Feature matching weight
    lambda_adv: float = 0.1       # Adversarial loss weight
    lambda_aux: float = 1.0       # Load balancing loss weight


@dataclass
class FASSMoEConfig:
    """Global configuration wrapper."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> FASSMoEConfig:
    """Returns the default configuration."""
    return FASSMoEConfig()
