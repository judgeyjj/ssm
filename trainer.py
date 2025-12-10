"""
Training loop for FASS-MoE Speech Super-Resolution.

Handles adversarial training with generator and discriminator.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import FASSMoEConfig
from discriminator import ProjectedDiscriminator, build_discriminator
from generator import FASSMoEGenerator, build_generator
from losses import MultiResolutionSTFTLoss, FeatureMatchingLoss, HingeLoss


class FASSMoETrainer:
    """
    Trainer for FASS-MoE model.
    
    Implements adversarial training loop with:
    - Generator: FASS-MoE for super-resolution
    - Discriminator: ViT-based Projected GAN
    
    Losses:
    - Generator: MR-STFT + Feature Matching + Adversarial Hinge + Load Balancing
    - Discriminator: Hinge Loss
    """
    
    def __init__(
        self,
        config: FASSMoEConfig,
        generator: FASSMoEGenerator,
        discriminator: ProjectedDiscriminator,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move models to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Optimizers (AdamW)
        # Weight decay defaulting to 0.01 if not specified, though config doesn't have it currently
        weight_decay = 0.01 
        
        self.optimizer_g = AdamW(
            self.generator.parameters(),
            lr=config.training.learning_rate_g,
            betas=config.training.betas,
            weight_decay=weight_decay,
        )
        
        trainable_d_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        self.optimizer_d = AdamW(
            trainable_d_params,
            lr=config.training.learning_rate_d,
            betas=config.training.betas,
            weight_decay=weight_decay,
        )
        
        self._setup_schedulers()
        
        # Loss functions
        self.mr_stft_loss = MultiResolutionSTFTLoss().to(self.device)
        self.feature_matching_loss = FeatureMatchingLoss().to(self.device)
        self.hinge_loss = HingeLoss().to(self.device)
        
        # Loss weights
        self.lambda_adv = config.training.lambda_adv
        self.lambda_fm = config.training.lambda_fm
        self.lambda_recon = config.training.lambda_mr_stft
        self.lambda_aux = config.training.lambda_aux
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_schedulers(self):
        """Setup learning rate schedulers with warmup."""
        steps_per_epoch = len(self.train_loader)
        warmup_steps = self.config.training.warmup_epochs * steps_per_epoch
        total_steps = self.config.training.num_epochs * steps_per_epoch
        
        warmup_g = LinearLR(self.optimizer_g, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        warmup_d = LinearLR(self.optimizer_d, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        
        # Avoid negative or zero duration for cosine
        cosine_steps = max(1, total_steps - warmup_steps)
        
        cosine_g = CosineAnnealingLR(self.optimizer_g, T_max=cosine_steps, eta_min=1e-6)
        cosine_d = CosineAnnealingLR(self.optimizer_d, T_max=cosine_steps, eta_min=1e-6)
        
        self.scheduler_g = SequentialLR(self.optimizer_g, [warmup_g, cosine_g], milestones=[warmup_steps])
        self.scheduler_d = SequentialLR(self.optimizer_d, [warmup_d, cosine_d], milestones=[warmup_steps])
    
    def train(self) -> None:
        """Run the full training loop."""
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            self._log_metrics(train_metrics, "train", epoch)
            
            if self.val_loader is not None and (epoch + 1) % self.config.training.val_interval == 0:
                val_metrics = self.validate()
                self._log_metrics(val_metrics, "val", epoch)
                
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
            
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        print("Training complete!")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {'g_loss': 0.0, 'd_loss': 0.0, 'mr_stft_loss': 0.0, 'fm_loss': 0.0, 'adv_loss': 0.0, 'aux_loss': 0.0}
        num_batches = 0
        
        for batch_idx, (low_res, high_res) in enumerate(self.train_loader):
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)
            
            step_metrics = self.train_step(low_res, high_res)
            
            for key in epoch_metrics:
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key]
            num_batches += 1
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.global_step += 1
            
            if (batch_idx + 1) % self.config.training.log_interval == 0:
                print(f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] G: {step_metrics['g_loss']:.4f} D: {step_metrics['d_loss']:.4f}")
            
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        return epoch_metrics
    
    def train_step(self, low_res: torch.Tensor, high_res: torch.Tensor) -> Dict[str, float]:
        """Single training step for both G and D."""
        metrics = {}
        
        # Discriminator step
        self.optimizer_d.zero_grad()
        
        with torch.no_grad():
            fake_high_res, _ = self.generator(low_res)
        
        real_logits, real_features = self.discriminator(high_res)
        fake_logits, _ = self.discriminator(fake_high_res)
        
        d_loss = self.hinge_loss.discriminator_loss(real_logits, fake_logits)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.training.grad_clip)
        self.optimizer_d.step()
        
        metrics['d_loss'] = d_loss.item()
        
        # Generator step
        self.optimizer_g.zero_grad()
        
        fake_high_res, aux_loss = self.generator(low_res)
        fake_logits, fake_features = self.discriminator(fake_high_res)
        _, real_features = self.discriminator(high_res)
        
        sc_loss, mag_loss = self.mr_stft_loss(fake_high_res, high_res)
        mr_stft_loss = sc_loss + mag_loss
        fm_loss = self.feature_matching_loss(real_features, fake_features)
        adv_loss = self.hinge_loss.generator_loss(fake_logits)
        
        g_loss = (
            self.lambda_recon * mr_stft_loss
            + self.lambda_fm * fm_loss
            + self.lambda_adv * adv_loss
            + self.lambda_aux * aux_loss
        )
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.training.grad_clip)
        self.optimizer_g.step()
        
        metrics['g_loss'] = g_loss.item()
        metrics['mr_stft_loss'] = mr_stft_loss.item()
        metrics['fm_loss'] = fm_loss.item()
        metrics['adv_loss'] = adv_loss.item()
        metrics['aux_loss'] = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {'total_loss': 0.0, 'mr_stft_loss': 0.0, 'snr': 0.0}
        num_batches = 0
        
        for low_res, high_res in self.val_loader:
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)
            
            fake_high_res, _ = self.generator(low_res)
            
            sc_loss, mag_loss = self.mr_stft_loss(fake_high_res, high_res)
            mr_stft_loss = sc_loss + mag_loss
            snr = self._compute_snr(fake_high_res, high_res)
            
            val_metrics['mr_stft_loss'] += mr_stft_loss.item()
            val_metrics['snr'] += snr.item()
            val_metrics['total_loss'] += mr_stft_loss.item()
            num_batches += 1
        
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)
        
        return val_metrics
    
    def _compute_snr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Signal-to-Noise Ratio in dB."""
        noise = pred - target
        signal_power = (target ** 2).mean()
        noise_power = (noise ** 2).mean()
        return 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str, epoch: int) -> None:
        """Log metrics to console."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[{prefix.upper()}] Epoch {epoch + 1}: {metric_str}")
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        print(f"Checkpoint loaded from {path}")


def create_trainer(
    config: FASSMoEConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "checkpoints",
) -> FASSMoETrainer:
    """Create a trainer with generator and discriminator."""
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    
    return FASSMoETrainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
