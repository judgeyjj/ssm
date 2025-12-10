"""
Projected GAN Discriminator components.

Contains ViT-based discriminator using frozen features with
learnable projection heads.
"""

from discriminator.mel import MelSpectrogram
from discriminator.vit import SimpleViT, TransformerBlock
from discriminator.projected_gan import (
    ProjectedDiscriminator,
    RandomProjection,
    DiscriminatorHead,
    build_discriminator,
)

__all__ = [
    "MelSpectrogram",
    "SimpleViT",
    "TransformerBlock",
    "ProjectedDiscriminator",
    "RandomProjection",
    "DiscriminatorHead",
    "build_discriminator",
]

