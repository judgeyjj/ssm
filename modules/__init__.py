"""
Model components for FASS-MoE.
"""

from modules.mamba import CausalConv1d, MambaBlock, RMSNorm
from modules.moe import (
    CausalConvExpert,
    HeterogeneousMoE,
    HeterogeneousMoERouter,
    MambaExpert,
    compute_spectral_entropy,
)
from modules.dsg import CausalDSG, DSGModule, FASSMoEBlock
from modules.norm import CausalRMSNorm, StreamingGroupNorm

__all__ = [
    # Mamba
    'CausalConv1d',
    'MambaBlock',
    'RMSNorm',
    # MoE
    'CausalConvExpert',
    'MambaExpert',
    'HeterogeneousMoE',
    'HeterogeneousMoERouter',
    'compute_spectral_entropy',
    # DSG
    'CausalDSG',
    'DSGModule',
    'FASSMoEBlock',
    # Norm
    'CausalRMSNorm',
    'StreamingGroupNorm',
]
