"""
Vision Transformer (ViT) implementation for discriminator.

Provides a simple ViT as fallback when timm is not available,
and a standard TransformerBlock for both implementations.
"""

from typing import List

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP.
    
    Follows the standard pre-norm architecture:
        x -> LayerNorm -> Attention -> + x -> LayerNorm -> MLP -> + x
    """
    
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        """
        Args:
            dim: Hidden dimension.
            heads: Number of attention heads.
            mlp_dim: Hidden dimension of MLP.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).
        Returns:
            Output tensor of shape (B, N, D).
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x


class SimpleViT(nn.Module):
    """
    Simple Vision Transformer implementation.
    
    Used as fallback when timm library is not available.
    Implements standard ViT architecture with:
    - Patch embedding via convolution
    - Learnable class token and position embeddings
    - Stack of Transformer blocks
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
    ):
        """
        Args:
            image_size: Input image size (assumes square).
            patch_size: Size of each patch.
            dim: Hidden dimension.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            mlp_dim: MLP hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = dim
        self.patch_size = patch_size
        self.depth = depth
        
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize learnable parameters."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding like a linear layer
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    
    def get_intermediate_features(
        self, x: torch.Tensor, layer_indices: List[int]
    ) -> List[torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            layer_indices: List of layer indices (1-indexed) to extract.
            
        Returns:
            List of feature tensors, each of shape (B, dim, h, w).
        """
        B = x.shape[0]
        features = []
        
        # Patch embedding: (B, 3, H, W) -> (B, dim, H/p, W/p)
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        
        # Flatten: (B, dim, H/p, W/p) -> (B, num_patches, dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Process through blocks and extract features
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Check if this layer should be extracted (1-indexed)
            if (i + 1) in layer_indices:
                # Remove class token and reshape to spatial
                feat = x[:, 1:, :]  # (B, num_patches, dim)
                feat = feat.transpose(1, 2).reshape(B, -1, h, w)
                features.append(feat)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass returning class token.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Class token embedding of shape (B, dim).
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token and position embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm and return class token
        x = self.norm(x)
        return x[:, 0]

