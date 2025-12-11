"""
Overfitting Test Script.

验证 Generator 是否有能力"背下"单个样本。
这是验证模型容量和梯度流的最简单有效的方法。

Training Mode:
- Generator ONLY (No Discriminator, No GAN Loss)
- Loss: MR-STFT Loss only
- Data: Single dummy sample repeated
"""

import os
import shutil
import tempfile
from pathlib import Path

import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from config import get_default_config
from generator import build_generator
from losses import MultiResolutionSTFTLoss

def create_single_sample_dataset(root_dir: Path, length: int = 48000):
    """Create a single deterministic sine wave sample."""
    root_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a complex signal: Sum of sines to make it non-trivial
    t = torch.linspace(0, 1, length)
    wav = (
        0.5 * torch.sin(2 * torch.pi * 440 * t) + 
        0.3 * torch.sin(2 * torch.pi * 880 * t) +
        0.1 * torch.sin(2 * torch.pi * 1760 * t)
    )
    wav = wav.unsqueeze(0)  # (1, T)
    
    path = root_dir / "overfit_sample.wav"
    torchaudio.save(str(path), wav, 48000)
    return path

class SingleSampleDataset(Dataset):
    """Dataset that returns the SAME sample every time."""
    def __init__(self, path, config):
        self.path = path
        self.config = config
        # Load once
        self.hr, _ = torchaudio.load(str(path))

        # For sanity overfit: let LR == HR (identity mapping at 48k)
        self.lr = self.hr.clone()

        # Crop / pad to exact training size so that LR and HR have same length
        seg_len = config.audio.segment_length

        def _crop_or_pad(x):
            cur_len = x.shape[-1]
            if cur_len < seg_len:
                pad = seg_len - cur_len
                return torch.nn.functional.pad(x, (0, pad), mode="constant", value=0.0)
            return x[:, :seg_len]

        self.hr = _crop_or_pad(self.hr)
        self.lr = _crop_or_pad(self.lr)
        
        # Only one band for identity overfit
        self.band_id = 0
        
    def __len__(self):
        return 100  # Pretend we have 100 samples per epoch
    
    def __getitem__(self, idx):
        return self.lr, self.hr, self.band_id

def train_overfit():
    print("\n" + "=" * 60)
    print("🔬 FASS-MoE Overfitting Test (Generator Only)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Setup Config & Data
    config = get_default_config()
    # For overfit sanity check: force 48k->48k identity setting
    config.audio.input_sr = 48000
    config.audio.target_sr = 48000
    # Only one effective band so band_id is always 0
    config.audio.effective_srs = [48000]
    # Use small model for fast iteration check
    config.model.hidden_channels = 32 
    config.model.num_moe_layers = 4
    
    temp_dir = Path("temp_overfit_data")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    sample_path = create_single_sample_dataset(temp_dir)
    
    dataset = SingleSampleDataset(sample_path, config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Setup Model & Optimizer
    generator = build_generator(config).to(device)
    optimizer = AdamW(generator.parameters(), lr=1e-3) # Higher LR for quick convergence
    criterion = MultiResolutionSTFTLoss().to(device)
    
    print("✅ Model created. Starting training loop...")
    print("Target: Loss should drop significantly (e.g., < 0.5 within 100 steps)")
    
    # 3. Training Loop
    generator.train()
    
    try:
        for epoch in range(1, 21): # 20 Epochs
            total_loss = 0
            steps = 0
            
            for lr, hr, band_id in dataloader:
                lr, hr, band_id = lr.to(device), hr.to(device), band_id.to(device)
                
                optimizer.zero_grad()
                
                fake, _ = generator(lr, band_id=band_id)
                
                # Compute only Reconstruction Loss
                sc_loss, mag_loss = criterion(fake, hr)
                loss = sc_loss + mag_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss / steps
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
            
            # Save intermediate result
            if epoch % 10 == 0:
                out_path = Path(f"overfit_epoch_{epoch}.wav")
                # Save generated audio
                torchaudio.save(str(out_path), fake[0].cpu().detach(), 48000)
                print(f"   Saved output to {out_path}")
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if temp_dir.exists(): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    train_overfit()

