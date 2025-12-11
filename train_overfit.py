"""
Overfitting Test Script (Updated).

验证 Generator 是否有能力"背下"单个样本。
适配了新的 Trainer 接口 (band_id) 和 Config。

Training Mode:
- Generator ONLY (No Discriminator, No GAN Loss)
- Loss: MR-STFT Loss only
- Data: Single dummy sample repeated
"""

import os
import shutil
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
    
    # Create a complex signal: Sum of sines
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
        
        # Create LR manually (Blind SR simulation)
        from dataset import LowPassFilter
        self.lpf = LowPassFilter(8000, 48000)
        self.resampler = torchaudio.transforms.Resample(
            48000, 16000, resampling_method="sinc_interp_kaiser"
        )
        
        # Create LR
        self.lr = self.resampler(self.lpf(self.hr))
        
        # Crop to exact training size
        seg_len = config.audio.segment_length
        lr_len = seg_len // 3  # Assuming 3x upsampling for this test
        
        self.hr = self.hr[:, :seg_len]
        self.lr = self.lr[:, :lr_len]
        
        # Dummy band_id (assuming standard single band or band 0)
        self.band_id = torch.tensor(0, dtype=torch.long)
        
    def __len__(self):
        return 100  # Pretend we have 100 samples per epoch
    
    def __getitem__(self, idx):
        # Return triplet: (LR, HR, BAND_ID)
        return self.lr, self.hr, self.band_id

def train_overfit():
    print("\n" + "=" * 60)
    print("🔬 FASS-MoE Overfitting Test (Generator Only)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Setup Config & Data
    config = get_default_config()
    # Use small model for fast iteration check
    config.model.hidden_channels = 32 
    config.model.num_moe_layers = 4
    # Ensure correct upsampling ratio for test (16k -> 48k = 3x)
    config.audio.input_sr = 16000
    config.audio.target_sr = 48000
    
    temp_dir = Path("temp_overfit_data")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    sample_path = create_single_sample_dataset(temp_dir)
    
    dataset = SingleSampleDataset(sample_path, config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Setup Model & Optimizer
    # Note: Generator forward now might expect band_id if you modified it to accept it
    # We build it standard way
    generator = build_generator(config).to(device)
    optimizer = AdamW(generator.parameters(), lr=1e-3) 
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
                lr = lr.to(device)
                hr = hr.to(device)
                band_id = band_id.to(device)
                
                optimizer.zero_grad()
                
                # Check if generator accepts band_id
                import inspect
                sig = inspect.signature(generator.forward)
                if 'band_id' in sig.parameters:
                    fake, _ = generator(lr, band_id=band_id)
                else:
                    fake, _ = generator(lr)
                
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
                torchaudio.save(str(out_path), fake[0].cpu().detach(), 48000)
                print(f"   Saved output to {out_path}")
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if temp_dir.exists(): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    train_overfit()
