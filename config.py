"""
Configuration for FASS-MoE Speech Super-Resolution.

Includes configurations for:
- Audio processing
- Model architecture (FASS-MoE)
- Discriminator
- Training hyperparameters
- Data paths
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    input_sr: int = 48000
    target_sr: int = 48000
    # Important: segment_length must be divisible by (target_sr / input_sr)
    segment_length: int = 16200 
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_channels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    effective_srs: List[int] = field(default_factory=lambda: [8000, 16000, 24000, 32000])


@dataclass
class ModelConfig:
    """FASS-MoE Generator configuration."""
    hidden_channels: int = 96
    num_moe_layers: int = 8
    num_experts: int = 8
    num_experts_per_tok: int = 2
    kernel_size: int = 7
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    dropout: float = 0.1
    use_checkpointing: bool = True


@dataclass
class DiscriminatorConfig:
    """Projected GAN Discriminator configuration."""
    vit_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    img_size: Tuple[int, int] = (224, 224)
    interp_types: List[str] = field(default_factory=lambda: ["pool", "pool", "pool", "pool"])
    feature_levels: List[int] = field(default_factory=lambda: [3, 6, 9, 12])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    grad_accum_steps: int = 2
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    decay: float = 0.999
    num_epochs: int = 200
    warmup_epochs: int = 5
    num_workers: int = 4
    checkpoint_interval: int = 5
    log_interval: int = 100
    val_interval: int = 1
    grad_clip: float = 1.0
    gan_start_epoch: int = 20
    # Loss Weights (HiFi-GAN / AudioSR style)
    lambda_mr_stft: float = 45.0
    lambda_fm: float = 2.0
    lambda_adv: float = 1.0
    lambda_aux: float = 0.01


@dataclass
class DataConfig:
    """Dataset paths."""
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""
    checkpoint_dir: str = "checkpoints"


@dataclass
class FASSMoEConfig:
    """Global configuration wrapper."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: str) -> "FASSMoEConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Helper to convert dict to dataclass
        def dict_to_dataclass(d, cls_type):
            valid_fields = {f.name for f in cls_type.__dataclass_fields__.values()}
            filtered_d = {k: v for k, v in d.items() if k in valid_fields}
            return cls_type(**filtered_d)

        # Create nested configs
        return cls(
            audio=dict_to_dataclass(config_dict.get("audio", {}), AudioConfig),
            model=dict_to_dataclass(config_dict.get("model", {}), ModelConfig),
            discriminator=dict_to_dataclass(config_dict.get("discriminator", {}), DiscriminatorConfig),
            training=dict_to_dataclass(config_dict.get("training", {}), TrainingConfig),
            data=dict_to_dataclass(config_dict.get("data", {}), DataConfig),
        )

    def save(self, path: str):
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def get_default_config() -> FASSMoEConfig:
    """Returns the default configuration."""
    return FASSMoEConfig()
