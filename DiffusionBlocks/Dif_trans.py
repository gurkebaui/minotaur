import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from DiffusionBlocks.Dif_tse import apply_rope



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Modulation: x * (1 + scale) + shift, gebroadcastet über die Sequenzlänge."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (wie in Llama-2)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight



class SwiGLU(nn.Module):
    """Feed-Forward-Netzwerk mit SwiGLU-Aktivierung (wie in Llama-2)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention mit RoPE und flexibler Kausal- oder Trainings-Maske."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Wenn keine explizite Maske übergeben wird (Inferenz), nutzen wir is_causal=True.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(out)


class DiTTransformerLayer(nn.Module):
    """
    Ein Llama-2-Transformer-Layer, erweitert um DiT AdaLN-Zero für Noise-Conditioning.
    """

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim, dropout)
        # 6 Parameter für Attention (shift, scale, gate) + MLP (shift, scale, gate)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(
            6, dim=-1
        )
        h_attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cos, sin, attn_mask)
        x = x + gate_msa.unsqueeze(1) * h_attn

        h_mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h_mlp
        return x


class DiffusionBlock(nn.Module):
    """Ein unabhängig trainierbarer Block aus L/B Transformer-Layern."""

    def __init__(
        self, num_layers: int, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DiTTransformerLayer(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond, cos, sin, attn_mask=attn_mask)
        return x


def build_training_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Erzeugt die (2S, 2S) Attention-Maske für das AR-Training (Appendix E.4).

    Eingabe-Sequenz ist konkateniert als: [clean_0..clean_{S-1}, noisy_0..noisy_{S-1}].
    Jedes noisy_i soll Token t_{i+1} vorhersagen und darf NUR auf clean_0..clean_i
    sowie auf sich selbst (noisy_i) achten:

                     Keys: clean (0..S-1) | noisy (0..S-1)
        Queries:     ---------------------+----------------
        clean (0..S) |  untere Dreiecks-  |   alles False
                     |  matrix (kausal)   |  (kein Leakage!)
        -------------+--------------------+----------------
        noisy (0..S) |  untere Dreiecks-  |  Einheitsmatrix
                     |  matrix (<= i)     |   (nur i == j)
    """
    tril = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    zeros = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)

    top = torch.cat([tril, zeros], dim=1)
    bottom = torch.cat([tril, eye], dim=1)
    return torch.cat([top, bottom], dim=0)


