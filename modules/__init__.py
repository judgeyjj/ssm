"""
FASS-MoE Model Components.

This package contains the core building blocks for the FASS-MoE generator:
- Mamba (Selective State Space Model)
- MoE (Mixture of Experts)
- DSG (Dynamic Spectral Gating)
"""

from modules.mamba import CausalConv1d, MambaBlock
from modules.moe import (
    CausalConvExpert,
    HeterogeneousMoE,
    HeterogeneousMoERouter,
    MambaExpert,
    compute_spectral_entropy,
)
from modules.dsg import CausalDSG, DSGModule, FASSMoEBlock

__all__ = [
    # Mamba
    "CausalConv1d",
    "MambaBlock",
    # MoE
    "CausalConvExpert",
    "MambaExpert",
    "HeterogeneousMoERouter",
    "HeterogeneousMoE",
    "compute_spectral_entropy",
    # DSG
    "CausalDSG",
    "DSGModule",
    "FASSMoEBlock",
]

