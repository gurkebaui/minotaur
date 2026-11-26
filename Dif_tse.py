import math

import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    """Wandelt das skalare Noise-Level c_noise in einen dim-dimensionalen Vektor um (wie in DiT)."""

    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freq_emb = self.timestep_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype)
        return self.mlp(freq_emb)


def compute_rope_freqs(
    head_dim: int, max_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Berechnet Sinus- und Kosinus-Tabellen für Rotary Position Embeddings (RoPE)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Wendet RoPE auf Query- oder Key-Tensor `x` der Form (B, num_heads, seq_len, head_dim) an.
    `cos` und `sin` haben die Form (seq_len, head_dim // 2).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_b = cos[None, None, :, :].to(x.dtype)
    sin_b = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], dim=-1)
