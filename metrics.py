"""
Evaluation Metrics for Speech Super-Resolution.

Implements metrics:
1. LSD (Log-Spectral Distance): The most critical metric for Bandwidth Extension.
2. Si-SNR (Scale-Invariant SNR): Robust time-domain metric.
"""

import torch
import torch.nn.functional as F


def compute_lsd(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 2048) -> float:
    """
    Compute Log-Spectral Distance (LSD) strictly following the formula:
    
    LSD = 1/T * sum_t( sqrt( 1/F * sum_f( (log10(S(t,f)^2 / S_hat(t,f)^2))^2 ) ) )
    
    This is computed **per-frame**: for each time frame t, compute RMSE over frequency,
    then average over all frames.
    
    Args:
        pred: Predicted audio (B, 1, T) or (B, T)  -- S_hat (generated)
        target: Target audio (B, 1, T) or (B, T)   -- S (ground truth)
        n_fft: FFT size (default 2048 for 48kHz)
        
    Returns:
        Average LSD value in dB (scalar)
    """
    if pred.dim() == 3: pred = pred.squeeze(1)
    if target.dim() == 3: target = target.squeeze(1)
    
    # STFT -> shape: [Batch, Freq, Time]
    stft_pred = torch.stft(pred, n_fft=n_fft, hop_length=n_fft//4, win_length=n_fft, return_complex=True).abs()
    stft_target = torch.stft(target, n_fft=n_fft, hop_length=n_fft//4, win_length=n_fft, return_complex=True).abs()
    
    # Power Spectrogram: S(t,f)^2 (add epsilon for numerical stability)
    power_pred = stft_pred ** 2 + 1e-8
    power_target = stft_target ** 2 + 1e-8
    
    # Log ratio: log10( S(t,f)^2 / S_hat(t,f)^2 )
    # = log10(S^2) - log10(S_hat^2)
    log_ratio = torch.log10(power_target) - torch.log10(power_pred)
    
    # Squared difference
    diff_squared = log_ratio ** 2  # [B, F, T]
    
    # Step 1: 1/F * sum_f(...) -> Mean over Frequency (dim=1)
    mean_over_freq = torch.mean(diff_squared, dim=1)  # [B, T]
    
    # Step 2: sqrt(...)
    rmse_per_frame = torch.sqrt(mean_over_freq)  # [B, T]
    
    # Step 3: 1/T * sum_t(...) -> Mean over Time (dim=1)
    lsd_per_sample = torch.mean(rmse_per_frame, dim=1)  # [B]
    
    # Return batch average
    return lsd_per_sample.mean().item()


def compute_sisnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Scale-Invariant SNR (Si-SNR).
    """
    if pred.dim() == 3: pred = pred.squeeze(1)
    if target.dim() == 3: target = target.squeeze(1)
    
    # Zero mean
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # Projection
    # s_target = <s, t> * t / ||t||^2
    dot_product = torch.sum(pred * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
    projection = dot_product * target / target_energy
    
    # Noise
    noise = pred - projection
    
    # SI-SNR
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8)
    sisnr = 10 * torch.log10(ratio)
    
    return sisnr.mean().item()

