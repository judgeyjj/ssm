"""
Dataset for Speech Super-Resolution.

Handles data loading and on-the-fly resampling from high-res to low-res
using high-quality Sinc interpolation to avoid aliasing artifacts.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from config import AudioConfig, FASSMoEConfig


class LowPassFilter(torch.nn.Module):
    """
    Low-pass filter using sinc-based FIR filter.
    
    Applies anti-aliasing filter before downsampling to prevent
    frequency folding artifacts.
    """
    
    def __init__(
        self,
        cutoff_freq: float,
        sample_rate: int,
        kernel_size: int = 101,
    ):
        """
        Initialize low-pass filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz.
            sample_rate: Sample rate of the input signal.
            kernel_size: Size of the FIR filter kernel (must be odd).
        """
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        
        # Ensure odd kernel size for symmetric filter
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Normalized cutoff frequency (0 to 1, where 1 is Nyquist)
        normalized_cutoff = cutoff_freq / (sample_rate / 2)
        
        # Create sinc filter kernel
        kernel = self._create_sinc_kernel(kernel_size, normalized_cutoff)
        self.register_buffer("kernel", kernel)
        self.padding = kernel_size // 2
    
    def _create_sinc_kernel(
        self, kernel_size: int, normalized_cutoff: float
    ) -> torch.Tensor:
        """Create a windowed sinc filter kernel."""
        n = torch.arange(kernel_size, dtype=torch.float32)
        center = (kernel_size - 1) / 2
        
        # Sinc function
        x = n - center
        sinc = torch.where(
            x == 0,
            torch.tensor(2.0 * normalized_cutoff),
            torch.sin(2.0 * torch.pi * normalized_cutoff * x) / (torch.pi * x),
        )
        
        # Apply Blackman window for better stopband attenuation
        window = (
            0.42
            - 0.5 * torch.cos(2.0 * torch.pi * n / (kernel_size - 1))
            + 0.08 * torch.cos(4.0 * torch.pi * n / (kernel_size - 1))
        )
        
        kernel = sinc * window
        
        # Normalize to preserve signal amplitude
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply low-pass filter to input signal.
        
        Args:
            x: Input tensor of shape (C, T) or (B, C, T).
            
        Returns:
            Filtered signal with same shape as input.
        """
        input_dim = x.dim()
        
        if input_dim == 2:
            # (C, T) -> (1, C, T)
            x = x.unsqueeze(0)
        
        batch_size, channels, length = x.shape
        
        # Process each channel separately
        x = x.view(batch_size * channels, 1, length)
        x = torch.nn.functional.pad(x, (self.padding, self.padding), mode="reflect")
        x = torch.nn.functional.conv1d(x, self.kernel)
        x = x.view(batch_size, channels, -1)
        
        if input_dim == 2:
            x = x.squeeze(0)
        
        return x


class SSRDataset(Dataset):
    """
    Speech Super-Resolution Dataset.
    
    Loads high-resolution audio and dynamically creates low-resolution
    versions using low-pass filtering and Sinc interpolation resampling.
    
    Args:
        audio_paths: List of paths to high-resolution audio files.
        config: Audio configuration containing sample rates and segment length.
        train: Whether this is for training (enables random cropping).
    """
    
    def __init__(
        self,
        audio_paths: List[Union[str, Path]],
        config: AudioConfig,
        train: bool = True,
    ):
        self.audio_paths = [Path(p) for p in audio_paths]
        self.config = config
        self.train = train
        
        # Validate paths
        self.audio_paths = [p for p in self.audio_paths if p.exists()]
        if len(self.audio_paths) == 0:
            raise ValueError("No valid audio files found in the provided paths.")
        
        # Low-pass filter at 8kHz (Nyquist frequency for 16kHz target)
        # Applied before downsampling to prevent aliasing
        self.lowpass_filter = LowPassFilter(
            cutoff_freq=8000,  # 8kHz cutoff
            sample_rate=config.target_sr,
            kernel_size=101,
        )
        
        # High-quality Sinc resampler for downsampling
        # Using kaiser_best for highest quality interpolation
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=config.target_sr,
            new_freq=config.input_sr,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            dtype=torch.float32,
        )
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load high-res audio and create low-res version via degradation.
        
        Degradation pipeline:
        1. Load high-res audio (48kHz)
        2. Random crop to segment_length
        3. Apply low-pass filter (8kHz cutoff)
        4. Downsample to 16kHz using sinc interpolation
        5. Normalize both to [-1, 1]
        
        Args:
            idx: Index of the audio file to load.
            
        Returns:
            Tuple of (low_res_audio, high_res_audio) tensors.
            - low_res_audio: Shape (1, input_segment_length) at 16kHz
            - high_res_audio: Shape (1, segment_length) at 48kHz
        """
        audio_path = self.audio_paths[idx]
        
        # Load high-resolution audio (48kHz)
        high_res_audio, sample_rate = self._load_audio(audio_path)
        
        # Resample to target sample rate if needed
        if sample_rate != self.config.target_sr:
            high_res_audio = self._resample_to_target(high_res_audio, sample_rate)
        
        # Random crop (training) or center crop (validation) to segment length
        high_res_audio = self._crop_segment(high_res_audio)
        
        # Create degraded low-resolution version
        low_res_audio = self._create_low_res(high_res_audio)
        
        # Normalize both to [-1, 1]
        high_res_audio = self._normalize(high_res_audio)
        low_res_audio = self._normalize(low_res_audio)
        
        return low_res_audio, high_res_audio
    
    def _load_audio(self, path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and convert to mono if necessary.
        
        Args:
            path: Path to the audio file.
            
        Returns:
            Tuple of (audio_tensor, sample_rate).
            Audio tensor has shape (1, T).
        """
        waveform, sample_rate = torchaudio.load(path)
        
        # Convert to mono by averaging channels if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, sample_rate
    
    def _resample_to_target(
        self, audio: torch.Tensor, orig_sr: int
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate if needed.
        
        Args:
            audio: Input audio tensor of shape (1, T).
            orig_sr: Original sample rate.
            
        Returns:
            Resampled audio at target sample rate.
        """
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=self.config.target_sr,
            resampling_method="sinc_interp_kaiser",
        )
        return resampler(audio)
    
    def _crop_segment(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Crop audio to the required segment length.
        
        Uses random cropping for training and center cropping for validation.
        If audio is shorter than segment_length, it will be padded.
        
        Args:
            audio: Input audio tensor of shape (1, T).
            
        Returns:
            Cropped audio of shape (1, segment_length).
        """
        segment_length = self.config.segment_length
        audio_length = audio.shape[-1]
        
        if audio_length < segment_length:
            # Pad with zeros if audio is too short
            padding = segment_length - audio_length
            audio = torch.nn.functional.pad(audio, (0, padding), mode="constant", value=0)
        elif audio_length > segment_length:
            if self.train:
                # Random crop for training
                max_start = audio_length - segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                # Center crop for validation
                start = (audio_length - segment_length) // 2
            
            audio = audio[:, start : start + segment_length]
        
        return audio
    
    def _create_low_res(self, high_res: torch.Tensor) -> torch.Tensor:
        """
        Create degraded low-resolution audio from high-resolution input.
        
        Pipeline:
        1. Apply low-pass filter at 8kHz (anti-aliasing)
        2. Downsample from 48kHz to 16kHz using sinc interpolation
        
        Args:
            high_res: High-resolution audio tensor of shape (1, segment_length).
            
        Returns:
            Low-resolution audio of shape (1, input_segment_length).
        """
        # Step 1: Apply low-pass filter at 8kHz cutoff
        # This removes frequencies above Nyquist for 16kHz to prevent aliasing
        filtered = self.lowpass_filter(high_res)
        
        # Step 2: Downsample to 16kHz using high-quality sinc interpolation
        low_res = self.resampler(filtered)
        
        return low_res
    
    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to [-1, 1] range.
        
        Uses peak normalization to preserve relative dynamics.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            Normalized audio tensor in [-1, 1] range.
        """
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio


def find_audio_files(
    data_dir: Union[str, Path],
    extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> List[Path]:
    """
    Recursively find all audio files in a directory.
    
    Args:
        data_dir: Root directory to search.
        extensions: Tuple of valid audio file extensions.
        
    Returns:
        List of paths to audio files.
    """
    data_dir = Path(data_dir)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(data_dir.rglob(f"*{ext}"))
    
    return sorted(audio_files)


def create_dataloader(
    config: FASSMoEConfig,
    audio_paths: Optional[List[Union[str, Path]]] = None,
    train: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the SSR dataset.
    
    Args:
        config: Full FASS-MoE configuration.
        audio_paths: List of paths to high-resolution audio files.
                    If None, will search config.data_dir for audio files.
        train: Whether this is for training.
        
    Returns:
        Configured DataLoader instance.
    """
    if audio_paths is None:
        audio_paths = find_audio_files(config.data_dir)
    
    dataset = SSRDataset(
        audio_paths=audio_paths,
        config=config.audio,
        train=train,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=train,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=train,
    )
    
    return dataloader


def create_train_val_dataloaders(
    config: FASSMoEConfig,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with a random split.
    
    Args:
        config: Full FASS-MoE configuration.
        val_split: Fraction of data to use for validation.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    all_paths = find_audio_files(config.data_dir)
    
    # Random shuffle for split
    import random
    random.shuffle(all_paths)
    
    split_idx = int(len(all_paths) * (1 - val_split))
    train_paths = all_paths[:split_idx]
    val_paths = all_paths[split_idx:]
    
    train_loader = create_dataloader(config, train_paths, train=True)
    val_loader = create_dataloader(config, val_paths, train=False)
    
    return train_loader, val_loader
