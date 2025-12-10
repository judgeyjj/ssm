"""
Integration Test Script: Data Loading & Training Loop Dry-Run.

验证:
1. Dataset 能否正确加载、滤波、下采样
2. Generator 和 Discriminator 能否在 Trainer 中协同工作
3. Loss 能否计算，梯度能否反向传播
"""

import os
import shutil
import tempfile
import traceback
from pathlib import Path

import torch
import torchaudio
import numpy as np

# Force CPU for this check if CUDA not available, or use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running integration test on: {device}")


def create_dummy_dataset(root_dir: Path, num_files: int = 10, duration_sec: float = 2.0):
    """Create dummy 48kHz wav files for testing."""
    root_dir.mkdir(exist_ok=True, parents=True)
    sr = 48000
    for i in range(num_files):
        # Create random signal
        t = torch.linspace(0, duration_sec, int(sr * duration_sec))
        # Add some sine waves
        wav = torch.sin(2 * torch.pi * 440 * t) + 0.5 * torch.sin(2 * torch.pi * 880 * t)
        wav = wav.unsqueeze(0)  # (1, T)
        
        path = root_dir / f"test_audio_{i:03d}.wav"
        # Convert Path to str for compatibility with older torchaudio versions
        torchaudio.save(str(path), wav, sr)
    print(f"✅ Created {num_files} dummy audio files in {root_dir}")


def test_data_pipeline(data_dir: Path):
    print("\n" + "-" * 40)
    print("🧪 Testing Data Pipeline")
    print("-" * 40)
    
    from config import get_default_config
    from dataset import create_dataloader
    
    config = get_default_config()
    config.training.batch_size = 4
    config.audio.segment_length = 16384  # 48k segment
    
    # Manually set data dir for finding files function
    # Note: create_dataloader calls find_audio_files(config.data_dir) if paths not provided
    # But here we provide paths manually to verify
    from dataset import find_audio_files
    paths = find_audio_files(data_dir)
    
    dataloader = create_dataloader(config, audio_paths=paths, train=True)
    
    try:
        lr_batch, hr_batch = next(iter(dataloader))
        
        print(f"Batch size: {lr_batch.shape[0]}")
        print(f"LR shape:   {lr_batch.shape} (Expected: [B, 1, 16384 // 3])")
        print(f"HR shape:   {hr_batch.shape} (Expected: [B, 1, 16384])")
        
        expected_lr_len = config.audio.segment_length // 3
        # Depending on rounding in dataset, might be off by 1-2 samples, checking approx
        assert abs(lr_batch.shape[-1] - expected_lr_len) <= 5, "LR length mismatch"
        assert hr_batch.shape[-1] == config.audio.segment_length, "HR length mismatch"
        
        print("✅ Data loading successful")
        return dataloader
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        traceback.print_exc()
        raise e


def test_training_loop(dataloader):
    print("\n" + "-" * 40)
    print("🧪 Testing Training Loop (Dry Run)")
    print("-" * 40)
    
    from config import get_default_config
    from trainer import create_trainer
    
    config = get_default_config()
    config.training.num_epochs = 1
    # Reduce size for speed check
    config.model.hidden_channels = 32
    config.model.num_moe_layers = 2
    
    # Initialize Trainer using create_trainer helper
    trainer = create_trainer(config, train_loader=dataloader, device=device)
    print("✅ Trainer initialized")
    
    # Run one step manually
    try:
        lr, hr = next(iter(dataloader))
        lr = lr.to(device)
        hr = hr.to(device)
        
        # 1. Generator Step
        g_loss, g_logs = trainer.train_generator_step(lr, hr)
        print(f"✅ G Step passed | Loss: {g_loss:.4f}")
        print(f"   Logs: {g_logs}")
        
        # 2. Discriminator Step
        # Need fake audio from generator first (detached)
        with torch.no_grad():
            fake, _ = trainer.generator(lr)
            
        d_loss, d_logs = trainer.train_discriminator_step(hr, fake.detach())
        print(f"✅ D Step passed | Loss: {d_loss:.4f}")
        print(f"   Logs: {d_logs}")
        
        print("✅ Full training step valid")
        
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        traceback.print_exc()
        # Check if it's because of missing modules
        if "No module named" in str(e):
             print("\n⚠️  Likely missing 'discriminator' module or dependencies.")
        raise e


def main():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # 1. Setup Data
        create_dummy_dataset(temp_dir)
        
        # 2. Test Pipeline
        dataloader = test_data_pipeline(temp_dir)
        
        # 3. Test Training
        test_training_loop(dataloader)
        
        print("\n" + "=" * 60)
        print("🎉 ALL SYSTEMS GO! Ready for SOTA training.")
        print("=" * 60)
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

