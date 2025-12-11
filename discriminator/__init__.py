"""
SOTA Discriminator components.
"""

from discriminator.hifi import HiFiDiscriminator

def build_discriminator(config):
    """Build HiFi-GAN style discriminator."""
    # Config is not strictly needed for standard HiFi-GAN D, 
    # but kept for interface compatibility.
    return HiFiDiscriminator()

__all__ = [
    "HiFiDiscriminator",
    "build_discriminator",
]
