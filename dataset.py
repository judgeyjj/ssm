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
from torch.utils.data.distributed import DistributedSampler

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
        
        # Effective sample rates (define different bandlimits), e.g. [8k,16k,24k,32k,48k]
        self.effective_srs = getattr(config, "effective_srs", [8000, 16000, 24000, 32000, 48000])
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
        low_res_audio, band_id = self._create_low_res(high_res_audio)
        
        # Normalize both to [-1, 1]
        high_res_audio = self._normalize(high_res_audio)
        low_res_audio = self._normalize(low_res_audio)
        
        return low_res_audio, high_res_audio, band_id
    
    def _load_audio(self, path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and convert to mono if necessary.
        
        Args:
            path: Path to the audio file.
            
        Returns:
            Tuple of (audio_tensor, sample_rate).
            Audio tensor has shape (1, T).
        """
        # Convert Path to str for compatibility with older torchaudio versions
        waveform, sample_rate = torchaudio.load(str(path))
        
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
    
    def _create_low_res(self, high_res: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Create degraded band-limited audio from high-resolution 48kHz input.
        
        Pipeline (48k -> band-limited 48k):
        1. Randomly pick an effective sample rate sr_eff from self.effective_srs
        2. Apply low-pass filter at cutoff = sr_eff / 2
        3. Optionally downsample to sr_eff then upsample back to 48k to simulate real resampling artifacts
        
        Args:
            high_res: High-resolution audio tensor of shape (1, segment_length).
            
        Returns:
            Band-limited audio of shape (1, segment_length) at 48kHz.
        """
        # Pick effective sample rate (defines Nyquist / effective bandwidth)
        if isinstance(self.effective_srs, list):
            effective_srs = self.effective_srs
        else:
            effective_srs = list(self.effective_srs)

        if len(effective_srs) == 0:
            effective_srs = [8000, 16000, 24000, 32000, 48000]

        idx = torch.randint(0, len(effective_srs), (1,)).item()
        sr_eff = int(effective_srs[idx])

        # Step 1: low-pass at sr_eff / 2 on 48k audio
        cutoff = sr_eff / 2.0
        lpf = LowPassFilter(
            cutoff_freq=cutoff,
            sample_rate=self.config.target_sr,
            kernel_size=101,
        )
        filtered = lpf(high_res)

        # Step 2: simulate real resampling pipeline: 48k -> sr_eff -> 48k
        if sr_eff < self.config.target_sr:
            down = torchaudio.transforms.Resample(
                orig_freq=self.config.target_sr,
                new_freq=sr_eff,
                resampling_method="sinc_interp_kaiser",
            )(filtered)
            band_limited = torchaudio.transforms.Resample(
                orig_freq=sr_eff,
                new_freq=self.config.target_sr,
                resampling_method="sinc_interp_kaiser",
            )(down)
        else:
            band_limited = filtered

        # Match length with high_res exactly (crop or pad)
        target_len = high_res.shape[-1]
        cur_len = band_limited.shape[-1]
        if cur_len < target_len:
            pad = target_len - cur_len
            band_limited = torch.nn.functional.pad(band_limited, (0, pad), mode="constant", value=0.0)
        elif cur_len > target_len:
            band_limited = band_limited[:, :target_len]

        return band_limited, idx
    
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
    
    if not data_dir.exists():
        print(f"Warning: Directory {data_dir} does not exist.")
        return []
    
    for ext in extensions:
        # Recursive glob search
        audio_files.extend(data_dir.rglob(f"*{ext}"))
    
    # Sort for deterministic order
    return sorted(audio_files)


def create_dataloader(
    config: FASSMoEConfig,
    data_dir: Optional[str] = None,
    audio_paths: Optional[List[Union[str, Path]]] = None,
    train: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for the SSR dataset.
    
    Args:
        config: Full FASS-MoE configuration.
        data_dir: Directory to search for audio files (if audio_paths is None).
        audio_paths: Specific list of audio paths (overrides data_dir).
        train: Whether this is for training.
        
    Returns:
        Configured DataLoader instance.
    """
    if audio_paths is None:
        if data_dir is None:
            raise ValueError("Must provide either data_dir or audio_paths")
        audio_paths = find_audio_files(data_dir)
        print(f"Found {len(audio_paths)} audio files in {data_dir}")
    
    dataset = SSRDataset(
        audio_paths=audio_paths,
        config=config.audio,
        train=train,
    )

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            drop_last=train,
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=train,
    )
    
    return dataloader


def create_dataloaders(
    config: FASSMoEConfig,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders using paths from config.
    
    Args:
        config: Full FASS-MoE configuration.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    print(f"Loading training data from: {config.data.train_dir}")
    train_loader = create_dataloader(
        config,
        data_dir=config.data.train_dir,
        train=True,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    
    print(f"Loading validation data from: {config.data.val_dir}")
    # For simplicity, run validation on full dataset on each rank
    val_loader = create_dataloader(
        config,
        data_dir=config.data.val_dir,
        train=False,
        distributed=False,
    )
    
    return train_loader, val_loader
