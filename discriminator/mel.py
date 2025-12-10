"""
Mel-spectrogram transform for discriminator input.

Converts raw audio waveform to mel-spectrogram representation
suitable for ViT-based discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogram(nn.Module):
    """
    Mel-spectrogram transform for converting audio to 2D representation.
    
    Converts raw audio waveform to mel-spectrogram suitable for ViT input.
    Uses a pre-computed mel filterbank for efficiency.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        """
        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT size.
            hop_length: Hop length between STFT frames.
            n_mels: Number of mel frequency bands.
            f_min: Minimum frequency for mel filterbank.
            f_max: Maximum frequency (defaults to Nyquist).
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_max = f_max or sample_rate / 2
        
        # Pre-compute mel filterbank
        mel_basis = self._create_mel_filterbank(
            n_fft, n_mels, sample_rate, f_min, self.f_max
        )
        self.register_buffer('mel_basis', mel_basis)
        
        # Hann window for STFT
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
    
    def _create_mel_filterbank(
        self,
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        f_min: float,
        f_max: float,
    ) -> torch.Tensor:
        """
        Create mel filterbank matrix.
        
        Returns:
            Filterbank matrix of shape (n_mels, n_fft // 2 + 1).
        """
        def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
            return 2595.0 * torch.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        # Create mel frequency points
        mel_min = hz_to_mel(torch.tensor(f_min))
        mel_max = hz_to_mel(torch.tensor(f_max))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
        
        # Create triangular filterbank
        n_freqs = n_fft // 2 + 1
        filterbank = torch.zeros(n_mels, n_freqs)
        
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rising slope
            for j in range(left, center):
                if j < n_freqs:
                    filterbank[i, j] = (j - left) / max(center - left, 1)
            
            # Falling slope
            for j in range(center, right):
                if j < n_freqs:
                    filterbank[i, j] = (right - j) / max(right - center, 1)
        
        return filterbank
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: Audio tensor of shape (B, 1, T) or (B, T).
            
        Returns:
            Mel-spectrogram of shape (B, 1, n_mels, time_frames).
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (B, T)
        
        B, T = audio.shape
        
        # Pad for STFT
        pad_amount = self.n_fft // 2
        audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        
        # Magnitude spectrogram
        magnitude = stft.abs()  # (B, n_freqs, time_frames)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_basis, magnitude)  # (B, n_mels, time_frames)
        
        # Log compression for perceptual scaling
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))
        
        # Add channel dimension for ViT compatibility
        mel_spec = mel_spec.unsqueeze(1)  # (B, 1, n_mels, time_frames)
        
        return mel_spec

