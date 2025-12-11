"""
Inference Script for FASS-MoE.

Supports batch processing of folders (recursive) or single files.

Usage:
    python inference.py --config config.yaml --ckpt checkpoints/best_model.pt --input /path/to/input --output /path/to/output
"""

import argparse
import os
from pathlib import Path
from typing import List, Union

import torch
import torchaudio
import tqdm

from config import FASSMoEConfig, get_default_config
from dataset import LowPassFilter
from generator import build_generator


def find_audio_files(path: Path) -> List[Path]:
    """Recursively find audio files."""
    if path.is_file():
        return [path]
    
    extensions = {'.wav', '.flac', '.mp3'}
    files = []
    for p in path.rglob('*'):
        if p.suffix.lower() in extensions:
            files.append(p)
    return sorted(files)


def load_model(config_path: str, checkpoint_path: str, device: str):
    """Load config and model checkpoint."""
    # Load config
    if os.path.exists(config_path):
        config = FASSMoEConfig.load(config_path)
    else:
        print(f"Warning: Config {config_path} not found, using defaults.")
        config = get_default_config()
    
    # Build generator
    # Need to handle scale factor correctly
    scale_factor = config.audio.target_sr // config.audio.input_sr
    model = build_generator(config).to(device)
    
    # Load weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'generator' in checkpoint:
        # Trained with trainer.py
        model.load_state_dict(checkpoint['generator'])
    elif 'generator_state_dict' in checkpoint:
        # Alternative key name (backwards compatibility)
        model.load_state_dict(checkpoint['generator_state_dict'])
    else:
        # Raw state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, config


def process_file(
    path: Path, 
    model: torch.nn.Module, 
    config: FASSMoEConfig, 
    output_dir: Path, 
    input_root: Path,
    device: str,
    simulate_lr: bool = True
):
    """Process a single audio file."""
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(path))
        
        # Prepare input
        if simulate_lr:
            # Simulation Mode: Input is HR, we degrade it first
            # 1. Resample to target SR first (if not already)
            if sr != config.audio.target_sr:
                resampler = torchaudio.transforms.Resample(sr, config.audio.target_sr).to(waveform.device)
                waveform = resampler(waveform)
            
            waveform = waveform.to(device)
            
            # 2. Apply LowPass
            lpf = LowPassFilter(8000, config.audio.target_sr).to(device)
            filtered = lpf(waveform)
            
            # 3. Downsample to input SR
            downsampler = torchaudio.transforms.Resample(
                config.audio.target_sr, 
                config.audio.input_sr,
                resampling_method="sinc_interp_kaiser"
            ).to(device)
            input_tensor = downsampler(filtered)
            
            # Save LR for reference? (Optional, skipping for now)
            
        else:
            # Real World Mode: Input is already LR
            # Just verify SR matches model input SR
            if sr != config.audio.input_sr:
                # Resample to match model input requirements
                resampler = torchaudio.transforms.Resample(sr, config.audio.input_sr).to(waveform.device)
                waveform = resampler(waveform)
            input_tensor = waveform.to(device)

        # Normalize
        input_tensor = input_tensor / (input_tensor.abs().max() + 1e-8)
        
        # Ensure (1, 1, T) shape
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0) # (1, C, T)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.view(1, 1, -1)
            
        # Inference
        with torch.no_grad():
            output_tensor = model.infer(input_tensor)
        
        # Determine output path maintaining directory structure
        if input_root.is_file():
            rel_path = path.name
        else:
            rel_path = path.relative_to(input_root)
            
        save_path = output_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        torchaudio.save(str(save_path), output_tensor.squeeze(0).cpu(), config.audio.target_sr)
        
    except Exception as e:
        print(f"Error processing {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="FASS-MoE Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input file or folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-sim", action="store_true", help="Disable LR simulation (assume input is already LR)")
    
    args = parser.parse_args()
    
    # Setup
    device = args.device
    model, config = load_model(args.config, args.ckpt, device)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = find_audio_files(input_path)
    print(f"Found {len(files)} audio files.")
    
    # Process
    for path in tqdm.tqdm(files):
        process_file(
            path, 
            model, 
            config, 
            output_dir, 
            input_path, 
            device, 
            simulate_lr=not args.no_sim
        )
    
    print("Inference complete.")


if __name__ == "__main__":
    main()
