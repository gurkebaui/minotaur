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


class LatentMoE(nn.Module):
    """Latent Mixture-of-Experts: Router in dim, Top-K SwiGLU-Experten in latent_dim + Shared Expert."""

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        expert_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        shared_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.down_proj = nn.Linear(dim, latent_dim, bias=False)
        self.up_proj = nn.Linear(latent_dim, dim, bias=False)

        self.w1 = nn.Parameter(torch.randn(num_experts, latent_dim, expert_dim) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, latent_dim, expert_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, expert_dim, latent_dim) * 0.02)

        self.shared_expert = SwiGLU(dim, shared_dim or expert_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        router_probs = F.softmax(self.router(x), dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        if self.training:
            one_hot = F.one_hot(topk_idx, self.num_experts).float().mean(dim=-2)
            f = one_hot.mean(dim=(0, 1))
            p = router_probs.mean(dim=(0, 1))
            self.aux_loss = self.num_experts * (f * p).sum()
        else:
            self.aux_loss = None

        z_flat = self.down_proj(x).reshape(-1, self.latent_dim)
        topk_idx_flat = topk_idx.reshape(-1, self.top_k)
        topk_w_flat = topk_weights.reshape(-1, self.top_k)

        out_latent = torch.zeros_like(z_flat)
        for e in range(self.num_experts):
            tok_idx, k_pos = torch.where(topk_idx_flat == e)
            if tok_idx.numel() == 0:
                continue
            z_e = z_flat[tok_idx]
            h_e = F.silu(z_e @ self.w1[e]) * (z_e @ self.w3[e])
            y_e = h_e @ self.w2[e]
            w_e = topk_w_flat[tok_idx, k_pos].unsqueeze(-1).to(y_e.dtype)
            out_latent.index_add_(0, tok_idx, (y_e * w_e).to(out_latent.dtype))

        out = self.up_proj(out_latent.view(bsz, seq_len, self.latent_dim))
        return self.dropout(out) + self.shared_expert(x)


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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        latent_dim: int | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        shared_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.mlp = LatentMoE(
            dim,
            latent_dim or max(16, dim // 2),
            mlp_dim,
            num_experts=num_experts,
            top_k=top_k,
            shared_dim=shared_dim,
            dropout=dropout,
        )
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
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        latent_dim: int | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        shared_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DiTTransformerLayer(
                    dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    rms_norm_eps=rms_norm_eps,
                    latent_dim=latent_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    shared_dim=shared_dim,
                )
                for _ in range(num_layers)
            ]
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


