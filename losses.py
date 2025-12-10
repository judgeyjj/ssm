"""
Loss functions for FASS-MoE training.

Implements Multi-Resolution STFT, Feature Matching, and Hinge losses
for adversarial training of audio super-resolution models.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss for audio reconstruction.
    
    Computes spectral convergence and log magnitude losses
    at multiple STFT resolutions for robust frequency matching.
    
    This loss captures both fine-grained and coarse spectral details
    by analyzing the signal at different time-frequency resolutions.
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_sizes: List[int] = None,
        win_sizes: List[int] = None,
    ):
        """
        Args:
            fft_sizes: List of FFT sizes for each resolution.
            hop_sizes: List of hop sizes for each resolution.
            win_sizes: List of window sizes for each resolution.
        """
        super().__init__()
        self.fft_sizes = fft_sizes or [512, 1024, 2048]
        self.hop_sizes = hop_sizes or [50, 120, 240]
        self.win_sizes = win_sizes or [240, 600, 1200]
        
        # Pre-compute Hann windows
        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(w), requires_grad=False)
            for w in self.win_sizes
        ])
    
    def _stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
        window: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute STFT magnitude.
        
        Args:
            x: Input audio tensor.
            fft_size: FFT size.
            hop_size: Hop size.
            win_size: Window size.
            window: Window tensor.
            
        Returns:
            STFT magnitude tensor.
        """
        # Ensure window is on same device
        window = window.to(x.device)
        
        # Pad window if needed
        if win_size < fft_size:
            pad = (fft_size - win_size) // 2
            window = F.pad(window, (pad, fft_size - win_size - pad))
        
        # Squeeze channel dimension if present
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window[:win_size],
            return_complex=True,
        )
        
        return stft.abs()
    
    def _spectral_convergence_loss(
        self, pred_mag: torch.Tensor, target_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral convergence loss.
        
        Measures the normalized Frobenius norm of the difference.
        """
        return torch.norm(target_mag - pred_mag, p='fro') / (
            torch.norm(target_mag, p='fro') + 1e-8
        )
    
    def _log_magnitude_loss(
        self, pred_mag: torch.Tensor, target_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Log magnitude loss.
        
        L1 loss in log domain for perceptually-relevant comparison.
        """
        pred_log = torch.log(pred_mag + 1e-8)
        target_log = torch.log(target_mag + 1e-8)
        return F.l1_loss(pred_log, target_log)
    
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            pred: Predicted audio of shape (B, 1, T).
            target: Target audio of shape (B, 1, T).
            
        Returns:
            Tuple of (spectral_convergence_loss, log_magnitude_loss).
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        for fft_size, hop_size, win_size, window in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes, self.windows
        ):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size, window)
            target_mag = self._stft(target, fft_size, hop_size, win_size, window)
            
            sc_loss = sc_loss + self._spectral_convergence_loss(pred_mag, target_mag)
            mag_loss = mag_loss + self._log_magnitude_loss(pred_mag, target_mag)
        
        num_resolutions = len(self.fft_sizes)
        return sc_loss / num_resolutions, mag_loss / num_resolutions


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss between real and generated samples.
    
    Matches intermediate discriminator features to stabilize GAN training.
    The real features are detached to prevent gradients from flowing
    through the discriminator during generator updates.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            real_features: List of feature tensors from real samples.
            fake_features: List of feature tensors from generated samples.
            
        Returns:
            Feature matching loss (scalar).
        """
        loss = 0.0
        
        for real_feat, fake_feat in zip(real_features, fake_features):
            # Detach real features to not update discriminator
            loss = loss + F.l1_loss(fake_feat, real_feat.detach())
        
        return loss / len(real_features)


class HingeLoss(nn.Module):
    """
    Hinge loss for GAN training.
    
    Discriminator loss: max(0, 1 - real) + max(0, 1 + fake)
    Generator loss: -fake
    
    Hinge loss provides more stable training compared to standard
    GAN losses by using a margin-based formulation.
    """
    
    def __init__(self):
        super().__init__()
    
    def discriminator_loss(
        self,
        real_logits: List[torch.Tensor],
        fake_logits: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute discriminator hinge loss.
        
        The discriminator aims to:
        - Output > 1 for real samples
        - Output < -1 for fake samples
        
        Args:
            real_logits: List of logits for real samples.
            fake_logits: List of logits for fake samples.
            
        Returns:
            Discriminator loss (scalar).
        """
        loss = 0.0
        
        for real, fake in zip(real_logits, fake_logits):
            # Hinge loss for real: max(0, 1 - real)
            loss = loss + torch.mean(F.relu(1.0 - real))
            # Hinge loss for fake: max(0, 1 + fake)
            loss = loss + torch.mean(F.relu(1.0 + fake))
        
        return loss / len(real_logits)
    
    def generator_loss(self, fake_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute generator hinge loss.
        
        The generator aims to maximize discriminator output for fakes,
        which is equivalent to minimizing -fake.
        
        Args:
            fake_logits: List of logits for generated samples.
            
        Returns:
            Generator loss (scalar).
        """
        loss = 0.0
        
        for fake in fake_logits:
            # Generator wants to maximize fake logits
            loss = loss - torch.mean(fake)
        
        return loss / len(fake_logits)


class CombinedGeneratorLoss(nn.Module):
    """
    Combined generator loss for FASS-MoE training.
    
    Combines:
    - Multi-resolution STFT loss (reconstruction)
    - Feature matching loss (perceptual)
    - Adversarial hinge loss (GAN)
    - Load balancing loss (MoE auxiliary)
    """
    
    def __init__(
        self,
        lambda_recon: float = 45.0,
        lambda_fm: float = 2.0,
        lambda_adv: float = 1.0,
        lambda_aux: float = 0.01,
    ):
        """
        Args:
            lambda_recon: Weight for reconstruction (MR-STFT) loss.
            lambda_fm: Weight for feature matching loss.
            lambda_adv: Weight for adversarial loss.
            lambda_aux: Weight for MoE load balancing loss.
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_fm = lambda_fm
        self.lambda_adv = lambda_adv
        self.lambda_aux = lambda_aux
        
        self.mr_stft = MultiResolutionSTFTLoss()
        self.feature_matching = FeatureMatchingLoss()
        self.hinge = HingeLoss()
    
    def forward(
        self,
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor,
        fake_logits: List[torch.Tensor],
        fake_features: List[torch.Tensor],
        real_features: List[torch.Tensor],
        aux_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined generator loss.
        
        Args:
            fake_audio: Generated audio.
            real_audio: Target audio.
            fake_logits: Discriminator logits for fake.
            fake_features: Discriminator features for fake.
            real_features: Discriminator features for real.
            aux_loss: MoE load balancing loss.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Reconstruction loss
        sc_loss, mag_loss = self.mr_stft(fake_audio, real_audio)
        recon_loss = sc_loss + mag_loss
        
        # Feature matching loss
        fm_loss = self.feature_matching(real_features, fake_features)
        
        # Adversarial loss
        adv_loss = self.hinge.generator_loss(fake_logits)
        
        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_fm * fm_loss
            + self.lambda_adv * adv_loss
            + self.lambda_aux * aux_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'fm': fm_loss.item(),
            'adv': adv_loss.item(),
            'aux': aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
        }
        
        return total_loss, loss_dict

