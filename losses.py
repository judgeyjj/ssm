"""
Loss functions for FASS-MoE training (SOTA HiFi-GAN style).

Implements LSGAN, Feature Matching, and Mel-Spectrogram losses.
Adapted for Multi-Period (MPD) and Multi-Scale (MSD) discriminators.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss for audio reconstruction."""
    
    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_sizes: List[int] = None,
        win_sizes: List[int] = None,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes or [512, 1024, 2048]
        self.hop_sizes = hop_sizes or [50, 120, 240]
        self.win_sizes = win_sizes or [240, 600, 1200]
        
        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(w), requires_grad=False)
            for w in self.win_sizes
        ])
    
    def _stft(self, x, fft_size, hop_size, win_size, window):
        window = window.to(x.device)
        if win_size < fft_size:
            pad = (fft_size - win_size) // 2
            window = F.pad(window, (pad, fft_size - win_size - pad))
        if x.dim() == 3: x = x.squeeze(1)
        stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_size, 
                          window=window[:win_size], return_complex=True)
        return stft.abs()
    
    def forward(self, pred, target):
        sc_loss = 0.0
        mag_loss = 0.0
        for fft_size, hop_size, win_size, window in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes, self.windows
        ):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size, window)
            target_mag = self._stft(target, fft_size, hop_size, win_size, window)
            
            sc_loss += torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
            mag_loss += F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))
        
        return sc_loss / len(self.fft_sizes), mag_loss / len(self.fft_sizes)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for MPD/MSD.
    Sum of L1 distance between feature maps of real and fake.
    """
    def __init__(self):
        super().__init__()

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss


class GeneratorLoss(nn.Module):
    """
    LSGAN Generator Loss.
    L_G = sum((D(G(z)) - 1)^2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, disc_outputs: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for dg in disc_outputs:
            loss += torch.mean((dg - 1) ** 2)
        return loss


class DiscriminatorLoss(nn.Module):
    """
    LSGAN Discriminator Loss.
    L_D = sum((D(x) - 1)^2 + (D(G(z)))^2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((dr - 1) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += (r_loss + g_loss)
        return loss


class CombinedGeneratorLoss(nn.Module):
    """
    Combined generator loss for FASS-MoE (SOTA Config).
    """
    def __init__(
        self,
        lambda_recon: float = 45.0, # STFT
        lambda_fm: float = 2.0,     # Feature Matching
        lambda_adv: float = 1.0,    # GAN
        lambda_aux: float = 1.0,    # Load Balance
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_fm = lambda_fm
        self.lambda_adv = lambda_adv
        self.lambda_aux = lambda_aux
        
        self.mr_stft = MultiResolutionSTFTLoss()
        self.feature_matching = FeatureMatchingLoss()
        self.gan_loss = GeneratorLoss()
    
    def forward(
        self,
        fake_audio,
        real_audio,
        fake_logits, # List[Tensor]
        fmap_r,      # List[List[Tensor]]
        fmap_g,      # List[List[Tensor]]
        aux_loss,
    ):
        # 1. Reconstruction (STFT)
        sc_loss, mag_loss = self.mr_stft(fake_audio, real_audio)
        recon_loss = sc_loss + mag_loss
        
        # 2. Adversarial (LSGAN)
        adv_loss = self.gan_loss(fake_logits)
        
        # 3. Feature Matching
        fm_loss = self.feature_matching(fmap_r, fmap_g)
        
        # Total
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_adv * adv_loss +
            self.lambda_fm * fm_loss +
            self.lambda_aux * aux_loss
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'adv': adv_loss.item(),
            'fm': fm_loss.item(),
            'aux': aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
        }
