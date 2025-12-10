"""
Configuration for FASS-MoE Speech Super-Resolution Model.

Based on paper specifications for audio super-resolution from 16kHz to 48kHz.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    input_sample_rate: int = 16000  # Low-resolution input (16kHz)
    target_sample_rate: int = 48000  # High-resolution target (48kHz)
    segment_length: int = 8192  # Segment length for high-res audio
    
    @property
    def upsampling_ratio(self) -> int:
        """Calculate the upsampling ratio."""
        return self.target_sample_rate // self.input_sample_rate
    
    @property
    def input_segment_length(self) -> int:
        """Segment length for low-res audio."""
        return self.segment_length // self.upsampling_ratio


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_channels: int = 64
    num_moe_layers: int = 4
    kernel_size: int = 7
    
    # Mamba block settings
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # MoE (Mixture of Experts) settings
    num_experts: int = 8
    num_experts_per_tok: int = 2  # Top-k experts to route
    
    # DSG (Dynamic Sparse Gating) settings
    dsg_threshold: float = 0.1


@dataclass
class DiscriminatorConfig:
    """ViT-based Projected GAN discriminator configuration."""
    patch_size: int = 16
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 4
    mlp_ratio: float = 4.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 0.01
    
    # Training duration
    num_epochs: int = 100
    steps_per_epoch: int = 1000
    
    # Loss weights
    adversarial_loss_weight: float = 1.0
    feature_matching_loss_weight: float = 2.0
    reconstruction_loss_weight: float = 45.0
    
    # Scheduler
    warmup_steps: int = 1000
    
    # Checkpointing
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 100


@dataclass
class FASSMoEConfig:
    """Master configuration combining all sub-configs."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42


def get_default_config() -> FASSMoEConfig:
    """Return the default configuration."""
    return FASSMoEConfig()

