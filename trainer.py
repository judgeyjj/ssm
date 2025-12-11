"""
Training loop for FASS-MoE Speech Super-Resolution (HiFi-GAN style).
Supports Generator Warmup, DistributedDataParallel, Tqdm, SwanLab logging,
LSD/Si-SNR metrics, and Audio Validation.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

from config import FASSMoEConfig
from discriminator import build_discriminator, HiFiDiscriminator
from generator import FASSMoEGenerator, build_generator
from losses import CombinedGeneratorLoss, DiscriminatorLoss
from metrics import compute_lsd, compute_sisnr


class FASSMoETrainer:
    """
    Trainer for FASS-MoE model with HiFi-GAN Discriminator.
    """
    
    def __init__(
        self,
        config: FASSMoEConfig,
        generator: FASSMoEGenerator,
        discriminator: HiFiDiscriminator,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        distributed: bool = False,
        local_rank: int = 0,
        enable_logging: bool = True,
    ):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.samples_dir = self.checkpoint_dir / "samples"
        self.distributed = distributed
        self.local_rank = local_rank
        self.enable_logging = enable_logging
        
        # Only main process does logging and checkpointing
        self.is_main_process = (not self.distributed) or (self.local_rank == 0)
        
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        if self.distributed:
            self.generator = DDP(self.generator, device_ids=[local_rank], output_device=local_rank)
            self.discriminator = DDP(self.discriminator, device_ids=[local_rank], output_device=local_rank)
        
        self.grad_accum_steps = config.training.grad_accum_steps
        self.gan_start_epoch = config.training.gan_start_epoch
        
        self.optimizer_g = AdamW(
            self.generator.parameters(),
            lr=config.training.learning_rate_g,
            betas=config.training.betas,
            weight_decay=0.01,
        )
        self.optimizer_d = AdamW(
            self.discriminator.parameters(),
            lr=config.training.learning_rate_d,
            betas=config.training.betas,
            weight_decay=0.01,
        )
        
        self._setup_schedulers()
        
        # Losses (LSGAN)
        self.d_criterion = DiscriminatorLoss().to(self.device)
            
        self.g_criterion = CombinedGeneratorLoss(
            lambda_recon=config.training.lambda_mr_stft,
            lambda_fm=config.training.lambda_fm,
            lambda_adv=config.training.lambda_adv,
            lambda_aux=config.training.lambda_aux,
        ).to(self.device)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_schedulers(self):
        steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
        warmup_steps = self.config.training.warmup_epochs * steps_per_epoch
        total_steps = self.config.training.num_epochs * steps_per_epoch
        
        cosine_steps = max(1, total_steps - warmup_steps)
        
        for optimizer in [self.optimizer_g, self.optimizer_d]:
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
            if optimizer == self.optimizer_g:
                self.scheduler_g = scheduler
            else:
                self.scheduler_d = scheduler
    
    def train(self) -> None:
        if self.is_main_process:
            print(f"Starting Training | Device: {self.device} | GAN Start: {self.gan_start_epoch}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            
            if self.is_main_process:
                self._log_metrics(train_metrics, "train", epoch)
            
            # Validation
            if self.val_loader and (epoch + 1) % self.config.training.val_interval == 0:
                val_metrics = self.validate(epoch)
                if self.is_main_process:
                    self._log_metrics(val_metrics, "val", epoch)
                    if val_metrics['lsd'] < self.best_val_loss: # Use LSD as best model metric
                        self.best_val_loss = val_metrics['lsd']
                        self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
            
            # Checkpoint
            if self.is_main_process and (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        
        metrics_sum = {}
        num_batches = 0
        
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        
        # Tqdm only on main process
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}", dynamic_ncols=True)
        else:
            pbar = self.train_loader
            
        for batch_idx, (low_res, high_res, band_id) in enumerate(pbar):
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)
            band_id = band_id.to(self.device)
            
            is_update = (batch_idx + 1) % self.grad_accum_steps == 0
            step_metrics = self.train_step(low_res, high_res, band_id, is_update, epoch)
            
            # Accumulate metrics for logging
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            num_batches += 1
            
            if is_update:
                self.scheduler_g.step()
                if epoch >= self.gan_start_epoch:
                    self.scheduler_d.step()
                self.global_step += 1
            
            # Update Tqdm bar
            if self.is_main_process and isinstance(pbar, tqdm):
                postfix = {
                    'g_loss': f"{step_metrics.get('g_loss', 0):.4f}",
                    'stft': f"{step_metrics.get('mr_stft_loss', 0):.4f}"
                }
                if 'd_loss' in step_metrics:
                    postfix['d_loss'] = f"{step_metrics['d_loss']:.4f}"
                pbar.set_postfix(postfix)
                
                # Log step-level metrics to SwanLab
                if HAS_SWANLAB and self.enable_logging and self.global_step % 10 == 0:
                    swanlab.log({f"train/step_{k}": v for k, v in step_metrics.items()}, step=self.global_step)
        
        return {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    
    def train_step(self, low_res, high_res, band_id, update, epoch):
        metrics = {}
        use_gan = epoch >= self.gan_start_epoch
        
        gen_module = self.generator.module if self.distributed else self.generator
        
        # ===================================================================================
        # 1. Discriminator Step (Only if warmed up)
        # ===================================================================================
        if use_gan:
            with torch.no_grad():
                fake_high_res, _ = gen_module(low_res, band_id=band_id)
            
            y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(high_res, fake_high_res.detach())
            
            d_loss = self.d_criterion(y_d_hat_r, y_d_hat_g)
            (d_loss / self.grad_accum_steps).backward()
            
            if update:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.training.grad_clip)
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()
            
            metrics['d_loss'] = d_loss.item()
            
        # ===================================================================================
        # 2. Generator Step
        # ===================================================================================
        fake_high_res, aux_loss = gen_module(low_res, band_id=band_id)
        
        if use_gan:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(high_res, fake_high_res)
            
            g_loss, loss_dict = self.g_criterion(
                fake_high_res, high_res, y_d_hat_g, fmap_r, fmap_g, aux_loss
            )
        else:
            # Warmup: Recon + Aux only
            sc, mag = self.g_criterion.mr_stft(fake_high_res, high_res)
            recon_loss = sc + mag
            g_loss = self.config.training.lambda_mr_stft * recon_loss + \
                     self.config.training.lambda_aux * aux_loss
            
            loss_dict = {'recon': recon_loss.item(), 'aux': aux_loss.item()}
        
        (g_loss / self.grad_accum_steps).backward()
        
        if update:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.training.grad_clip)
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            
        metrics['g_loss'] = g_loss.item()
        metrics['mr_stft_loss'] = loss_dict.get('recon', 0.0)
        if use_gan:
            metrics['fm_loss'] = loss_dict.get('fm', 0.0)
            metrics['adv_loss'] = loss_dict.get('adv', 0.0)
        metrics['aux_loss'] = loss_dict.get('aux', 0.0)
        
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int):
        gen_module = self.generator.module if self.distributed else self.generator
        gen_module.eval()
        
        metrics = {'lsd': 0.0, 'sisnr': 0.0, 'mr_stft': 0.0}
        count = 0
        
        # Tqdm for validation (only main process)
        if self.is_main_process:
            loader = tqdm(self.val_loader, desc="Validating", leave=False)
        else:
            loader = self.val_loader
            
        for i, (low, high, band_id) in enumerate(loader):
            low, high = low.to(self.device), high.to(self.device)
            band_id = band_id.to(self.device)
            
            fake, _ = gen_module(low, band_id=band_id)
            
            # Compute Metrics
            sc, mag = self.g_criterion.mr_stft(fake, high)
            metrics['mr_stft'] += (sc + mag).item()
            metrics['lsd'] += compute_lsd(fake, high)
            metrics['sisnr'] += compute_sisnr(fake, high)
            count += 1
            
            # Save Audio (First batch of first rank only)
            if self.is_main_process and i == 0:
                self._save_val_audio(low, high, fake, epoch)
            
        return {k: v / max(1, count) for k, v in metrics.items()}

    def _save_val_audio(self, low, high, fake, epoch):
        """Save sample audio files."""
        # Save up to 2 samples
        for j in range(min(2, low.shape[0])):
            # Save Low Res
            torchaudio.save(
                str(self.samples_dir / f"epoch_{epoch+1}_sample_{j}_lr.wav"),
                low[j].cpu().detach(), 
                self.config.audio.target_sr # It's upsampled to target_sr length already
            )
            # Save High Res (Target)
            torchaudio.save(
                str(self.samples_dir / f"epoch_{epoch+1}_sample_{j}_hr.wav"),
                high[j].cpu().detach(), 
                self.config.audio.target_sr
            )
            # Save Generated
            torchaudio.save(
                str(self.samples_dir / f"epoch_{epoch+1}_sample_{j}_gen.wav"),
                fake[j].cpu().detach(), 
                self.config.audio.target_sr
            )

    def _log_metrics(self, metrics, prefix, epoch):
        if not self.is_main_process: return
        
        # Console log
        print(f"[{prefix.upper()}] Epoch {epoch+1}: " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        
        # SwanLab log
        if HAS_SWANLAB and self.enable_logging:
            swanlab.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step)

    def save_checkpoint(self, path):
        if not self.is_main_process: return
        
        gen_module = self.generator.module if self.distributed else self.generator
        disc_module = self.discriminator.module if self.distributed else self.discriminator
        
        torch.save({
            'epoch': self.current_epoch,
            'generator': gen_module.state_dict(),
            'discriminator': disc_module.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'config': self.config,
        }, path)
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        # Map location to correct device
        ckpt = torch.load(path, map_location=self.device)
        self.current_epoch = ckpt['epoch']
        
        gen_module = self.generator.module if self.distributed else self.generator
        disc_module = self.discriminator.module if self.distributed else self.discriminator
        
        gen_module.load_state_dict(ckpt['generator'])
        disc_module.load_state_dict(ckpt['discriminator'])
        self.optimizer_g.load_state_dict(ckpt['optimizer_g'])
        self.optimizer_d.load_state_dict(ckpt['optimizer_d'])
        print(f"📂 Checkpoint loaded: {path}")


def create_trainer(config: FASSMoEConfig, train_loader, val_loader=None, device="cuda", checkpoint_dir="checkpoints", distributed=False, local_rank=0):
    gen = build_generator(config)
    disc = build_discriminator(config)
    return FASSMoETrainer(config, gen, disc, train_loader, val_loader, device, checkpoint_dir, distributed, local_rank)
