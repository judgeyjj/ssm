"""
Projected GAN Discriminator implementation.

Uses frozen ViT features with learnable random projection heads
for multi-scale discriminator outputs.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DiscriminatorConfig, FASSMoEConfig
from discriminator.mel import MelSpectrogram
from discriminator.vit import SimpleViT


class RandomProjection(nn.Module):
    """
    Learnable random projection head for Projected GAN.
    
    Projects ViT features to a lower-dimensional space for discrimination.
    Initialized with scaled random weights.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 64):
        """
        Args:
            in_channels: Number of input channels (ViT hidden dim).
            out_channels: Number of output channels.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        
        self._init_random()
    
    def _init_random(self):
        """Initialize with scaled random weights."""
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, C, H, W).
        Returns:
            Projected features of shape (B, out_channels, H, W).
        """
        return self.proj(x)


class DiscriminatorHead(nn.Module):
    """
    Discriminator head that produces real/fake logits.
    
    Applies additional convolutions before producing scalar predictions.
    """
    
    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: Number of input channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, C, H, W).
        Returns:
            Logits of shape (B, 1, H, W).
        """
        return self.conv(x)


class ProjectedDiscriminator(nn.Module):
    """
    Projected GAN Discriminator using frozen ViT features.
    
    Architecture:
    1. Convert audio to mel-spectrogram
    2. Resize to ViT input size (224x224)
    3. Extract features from ViT layers [3, 6, 9, 12]
    4. Apply learnable random projections
    5. Produce multi-scale discriminator outputs
    
    The ViT backbone is frozen; only projections and heads are trained.
    """
    
    def __init__(
        self,
        config: DiscriminatorConfig,
        sample_rate: int = 48000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        vit_model: str = 'vit_base_patch16_224',
        feature_layers: List[int] = None,
        proj_channels: int = 64,
    ):
        """
        Args:
            config: Discriminator configuration.
            sample_rate: Audio sample rate.
            n_fft: FFT window size for MelSpectrogram.
            hop_length: Hop length for MelSpectrogram.
            n_mels: Number of mel filterbanks.
            vit_model: Name of ViT model to load from timm.
            feature_layers: Which ViT layers to extract features from.
            proj_channels: Number of channels for projection heads.
        """
        super().__init__()
        self.config = config
        self.feature_layers = feature_layers or [3, 6, 9, 12]
        
        # Mel-spectrogram transform
        # Note: We use the passed n_mels (usually 80 or 128) and interpolate to 224 later
        # OR we can compute 224 mels directly. The original paper often interpolates.
        # Here we compute the audio-specific mel first.
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
        # Load ViT backbone
        self.vit = self._load_vit(vit_model)
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get ViT hidden dimension
        vit_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else 768
        
        # Random projection heads for each feature layer
        self.projections = nn.ModuleList([
            RandomProjection(vit_dim, proj_channels)
            for _ in self.feature_layers
        ])
        
        # Discriminator heads
        self.heads = nn.ModuleList([
            DiscriminatorHead(proj_channels)
            for _ in self.feature_layers
        ])
    
    def _load_vit(self, model_name: str) -> nn.Module:
        """Load ViT model from timm or use fallback."""
        try:
            import timm
            vit = timm.create_model(model_name, pretrained=True)
            return vit
        except ImportError:
            print("Warning: timm not installed, using SimpleViT fallback")
            return SimpleViT(
                image_size=224,
                patch_size=16, # Default for base_patch16
                dim=768,       # Default for base
                depth=12,
                heads=12,
            )
    
    def _extract_vit_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract intermediate features from ViT.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            List of feature tensors from specified layers.
        """
        features = []
        
        # Check if using timm model or SimpleViT
        if hasattr(self.vit, 'patch_embed') and hasattr(self.vit, 'blocks'):
            # timm ViT
            B = x.shape[0]
            
            # Patch embedding
            x = self.vit.patch_embed(x)
            
            # Handle case where patch_embed returns (B, C, H, W) instead of (B, N, C)
            if x.dim() == 4:
                x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, C, N) -> (B, N, C)
            
            # Add class token if present
            if hasattr(self.vit, 'cls_token'):
                cls_token = self.vit.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
            
            # Add position embedding
            if hasattr(self.vit, 'pos_embed'):
                x = x + self.vit.pos_embed
            
            # Apply dropout if present
            if hasattr(self.vit, 'pos_drop'):
                x = self.vit.pos_drop(x)
            
            # Extract features from specified layers
            for i, block in enumerate(self.vit.blocks):
                x = block(x)
                if (i + 1) in self.feature_layers:
                    feat = self._reshape_to_spatial(x, B)
                    features.append(feat)
        else:
            # SimpleViT fallback
            features = self.vit.get_intermediate_features(x, self.feature_layers)
        
        return features
    
    def _reshape_to_spatial(
        self, x: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Reshape ViT sequence output to spatial feature map.
        
        Args:
            x: Sequence tensor of shape (B, num_patches + 1, dim).
            batch_size: Batch size.
            
        Returns:
            Spatial tensor of shape (B, dim, H, W).
        """
        # Remove class token
        x = x[:, 1:, :]  # (B, num_patches, dim)
        
        # Compute spatial dimensions
        num_patches = x.shape[1]
        h = w = int(num_patches ** 0.5)
        
        # Reshape to spatial format
        x = x.transpose(1, 2)  # (B, dim, num_patches)
        x = x.reshape(batch_size, -1, h, w)  # (B, dim, H, W)
        
        return x
    
    def forward(
        self, audio: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass producing multi-scale discriminator outputs.
        
        Args:
            audio: Audio tensor of shape (B, 1, T).
            
        Returns:
            Tuple of (logits_list, features_list):
            - logits_list: List of discriminator logits for each scale
            - features_list: List of projected features for feature matching
        """
        B = audio.shape[0]
        
        # Convert audio to mel-spectrogram
        mel = self.mel_transform(audio)  # (B, 1, n_mels, time)
        
        # Resize to ViT input size (224x224)
        mel = F.interpolate(
            mel, size=(224, 224), mode='bilinear', align_corners=False
        )
        
        # Expand to 3 channels (ViT expects RGB)
        mel = mel.expand(-1, 3, -1, -1)
        
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406], device=mel.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=mel.device).view(1, 3, 1, 1)
        mel = (mel - mean) / std
        
        # Extract ViT features (frozen)
        with torch.no_grad():
            vit_features = self._extract_vit_features(mel)
        
        # Apply projections and heads (trainable)
        logits_list = []
        features_list = []
        
        for feat, proj, head in zip(vit_features, self.projections, self.heads):
            # Project features
            proj_feat = proj(feat)
            features_list.append(proj_feat)
            
            # Discriminator output
            logits = head(proj_feat)
            logits_list.append(logits)
        
        return logits_list, features_list


def build_discriminator(config: FASSMoEConfig) -> ProjectedDiscriminator:
    """
    Build and initialize the discriminator.
    
    Args:
        config: Full FASS-MoE configuration.
        
    Returns:
        Initialized ProjectedDiscriminator model.
    """
    return ProjectedDiscriminator(
        config=config.discriminator,
        sample_rate=config.audio.target_sr,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.mel_channels,
        vit_model=config.discriminator.vit_name,
        feature_layers=config.discriminator.feature_levels,
    )
