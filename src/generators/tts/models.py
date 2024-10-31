"""Local implementations of F5-TTS models"""
import torch
from torch import nn
import torch.nn.functional as F

class DiT(nn.Module):
    """Diffusion Transformer model"""
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        ff_mult: int = 2,
        text_dim: int = 512,
        conv_layers: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.text_dim = text_dim
        
        # Text embedding
        self.text_embedding = nn.Embedding(text_dim, dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                ff_mult=ff_mult
            ) for _ in range(depth)
        ])
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(dim, dim, 3, padding=1)
            for _ in range(conv_layers)
        ])
        
        # Output projection
        self.to_out = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor = None,
        time: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        b, n, d = x.shape
        
        # Add positional embeddings
        pos = self.pos_embedding[:, :n]
        x = x + pos
        
        # Add text embeddings if provided
        if text is not None:
            text_emb = self.text_embedding(text)
            x = x + text_emb
            
        # Add time embeddings if provided
        if time is not None:
            time = time.view(-1, 1, 1)
            x = x * time
            
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        # Apply convolutional layers
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = F.gelu(conv(x))
        x = x.transpose(1, 2)
        
        # Output projection
        x = self.to_out(x)
        
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            
        return x

class TransformerBlock(nn.Module):
    """Basic transformer block"""
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self attention
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, attn_mask=mask)[0]
        
        # Feedforward
        x = x + self.ff(self.norm2(x))
        return x

class UNetT(nn.Module):
    """U-Net Transformer model"""
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 24,
        heads: int = 16,
        ff_mult: int = 4
    ):
        super().__init__()
        self.dim = dim
        
        # Encoder blocks
        self.encoder = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, ff_mult=ff_mult)
            for _ in range(depth // 2)
        ])
        
        # Decoder blocks with skip connections
        self.decoder = nn.ModuleList([
            TransformerBlock(dim=dim * 2, heads=heads, ff_mult=ff_mult)
            for _ in range(depth // 2)
        ])
        
        # Output projection
        self.to_out = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor = None,
        time: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder
        for block in self.encoder:
            x = block(x, mask=mask)
            encoder_outputs.append(x)
            
        # Decoder with skip connections
        for block, skip in zip(self.decoder, reversed(encoder_outputs)):
            x = torch.cat([x, skip], dim=-1)
            x = block(x, mask=mask)
            
        # Output projection
        x = self.to_out(x)
        
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            
        return x
