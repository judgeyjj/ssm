"""
Main Training Script.

Usage:
    python train.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path
import os

import torch
import torch.distributed as dist
from dataclasses import asdict

from config import FASSMoEConfig, get_default_config
from dataset import create_dataloaders
from trainer import create_trainer

try:
    import swanlab
    _SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None  # type: ignore
    _SWANLAB_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Train FASS-MoE Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use DistributedDataParallel for multi-GPU training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (used with torchrun)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
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
        
    # Distributed setup
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    if args.distributed:
        distributed = True
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            rank = 0
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            local_rank = args.local_rank if args.local_rank >= 0 else 0

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl" if dist.is_nccl_available() else "gloo"
        else:
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if rank == 0:
            print(f"Initialized distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}, backend={backend}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if _SWANLAB_AVAILABLE and (not distributed or rank == 0):
        try:
            swanlab.init(project="FASS-MoE-SSR", config=asdict(config))
        except Exception:
            pass
        
    # 2. Prepare Data
    print("\n[Data Preparation]")
    try:
        train_loader, val_loader = create_dataloaders(
            config,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
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
        checkpoint_dir=config.data.checkpoint_dir,
        distributed=distributed,
        local_rank=local_rank,
    )
    
    # 4. Start Training
    if args.resume:
        print(f"\n[Resuming Training] Loading checkpoint from {args.resume}")
        trainer.load_checkpoint(args.resume)

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
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        if _SWANLAB_AVAILABLE and (not distributed or rank == 0):
            try:
                swanlab.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
