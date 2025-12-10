"""
Main Training Script.

Usage:
    python train.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

from config import FASSMoEConfig, get_default_config
from dataset import create_dataloaders
from trainer import create_trainer


def main():
    parser = argparse.ArgumentParser(description="Train FASS-MoE Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    # 1. Load Configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading configuration from {config_path}")
        config = FASSMoEConfig.load(str(config_path))
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        config = get_default_config()
        
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # 2. Prepare Data
    print("\n[Data Preparation]")
    try:
        train_loader, val_loader = create_dataloaders(config)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Please check your data paths in config.yaml")
        sys.exit(1)
        
    if len(train_loader) == 0:
        print("Error: Training dataloader is empty.")
        sys.exit(1)
        
    # 3. Initialize Trainer
    print("\n[Model Initialization]")
    trainer = create_trainer(
        config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=config.data.checkpoint_dir
    )
    
    # 4. Start Training
    print("\n[Training Start]")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        trainer.save_checkpoint(Path(config.data.checkpoint_dir) / "interrupted_checkpoint.pt")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
