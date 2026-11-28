"""Attention-Layer: FA-2/FA-3, FlexAttention, SWA, MLA, QK-Norm, Head-Wise Gating, Partial RoPE & YOCO."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:
    _flash_attn_func = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:
    flex_attention = None
    create_block_mask = None

try:
    from liger_kernel.transformers import LigerRMSNorm as _LigerRMSNorm
except Exception:
    _LigerRMSNorm = None


class RMSNorm(nn.Module):
    """RMSNorm mit Auto-Dispatch auf Liger / F.rms_norm / Manual (für Exp 0_sweep)."""

    def __init__(self, dim: int, eps: float = 1e-6, use_fused: bool = True):
        super().__init__()
        self.eps = eps
        self.use_fused = use_fused
        self.weight = nn.Parameter(torch.ones(dim))
        self._liger = _LigerRMSNorm(dim, eps=eps) if (_LigerRMSNorm is not None and use_fused) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._liger is not None:
            return self._liger(x)
        if self.use_fused and hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Partial/Full Rotary Position Embedding."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0, partial_frac: float = 1.0):
        super().__init__()
        rot_dim = max(2, int(head_dim * partial_frac))
        rot_dim = (rot_dim // 2) * 2
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, H, T, D)
        d = self.rot_dim
        if positions is None:
            t = x.shape[-2]
            cos = self.cos_cached[:t].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(1)
            sin = self.sin_cached[:t].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(1)
        else:
            # positions: (B, T)
            cos = self.cos_cached[positions].to(device=x.device, dtype=x.dtype).unsqueeze(1)
            sin = self.sin_cached[positions].to(device=x.device, dtype=x.dtype).unsqueeze(1)

        x_rot = x[..., :d]
        x_pass = x[..., d:]
        half = d // 2
        rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
        out_rot = x_rot * cos + rotated * sin
        if x_pass.shape[-1] == 0:
            return out_rot
        return torch.cat((out_rot, x_pass), dim=-1)


def _build_causal_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int | None = None,
    doc_ids: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Erstellt kompakte Bool-/Float-Maske für SWA & Document Masking."""
    if q_positions is None:
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)  # (T_q, 1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)  # (1, T_k)
        causal = k_pos <= q_pos
        if window_size is not None:
            causal = causal & ((q_pos - k_pos) < window_size)
        mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
    else:
        # q_positions: (B, T_q), k_pos: (1, 1, T_k)
        q_p = q_positions.unsqueeze(-1)  # (B, T_q, 1)
        k_p = torch.arange(k_len, device=device).view(1, 1, k_len)
        causal = k_p <= q_p
        if window_size is not None:
            causal = causal & ((q_p - k_p) < window_size)
        mask = causal.unsqueeze(1)  # (B, 1, T_q, T_k)

    if doc_ids is not None and q_positions is None and q_len == k_len:
        # doc_ids: (B, T)
        same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
        mask = mask & same_doc.unsqueeze(1)
    elif doc_ids is not None and q_positions is not None:
        q_docs = torch.gather(doc_ids, 1, q_positions)
        same_doc = q_docs.unsqueeze(-1) == doc_ids.unsqueeze(1)
        mask = mask & same_doc.unsqueeze(1)

    return mask


class CausalAttention(nn.Module):
    """Vereinheitlichte Attention: GQA, MLA, QK-Norm, Head-Wise Gating, SWA, Partial RoPE & YOCO."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
        partial_rope_frac: float = 1.0,
        use_qk_norm: bool = True,
        use_fused_ops: bool = True,
        use_flash_attn: bool = True,
        use_flex_attn: bool = False,
        use_head_gating: bool = False,
        use_mla: bool = False,
        mla_kv_lora_rank: int = 192,
        window_size: int | None = None,
        is_yoco_cross: bool = False,
        linear_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        self.use_flex_attn = use_flex_attn
        self.use_head_gating = use_head_gating
        self.use_mla = use_mla and (not is_yoco_cross)
        self.window_size = window_size
        self.is_yoco_cross = is_yoco_cross

        mk_lin = linear_factory or (lambda i, o: nn.Linear(i, o, bias=False))

        self.q_proj = mk_lin(d_model, n_heads * self.head_dim)
        if not is_yoco_cross:
            if self.use_mla:
                self.kv_down = nn.Linear(d_model, mla_kv_lora_rank, bias=False)
                self.k_up = nn.Linear(mla_kv_lora_rank, n_kv_heads * self.head_dim, bias=False)
                self.v_up = nn.Linear(mla_kv_lora_rank, n_kv_heads * self.head_dim, bias=False)
            else:
                self.k_proj = mk_lin(d_model, n_kv_heads * self.head_dim)
                self.v_proj = mk_lin(d_model, n_kv_heads * self.head_dim)

        self.out_proj = mk_lin(n_heads * self.head_dim, d_model)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, use_fused=use_fused_ops)
            self.k_norm = RMSNorm(self.head_dim, use_fused=use_fused_ops)

        if use_head_gating:
            self.head_gate = nn.Linear(d_model, n_heads, bias=True)
            nn.init.constant_(self.head_gate.bias, 1.0)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, base=rope_base, partial_frac=partial_rope_frac)

    def forward(
        self,
        x: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        external_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        q_positions: torch.Tensor | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T_q, _ = x.shape

        q = self.q_proj(x).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
        q = self.rope.apply(q, positions=q_positions)

        if self.is_yoco_cross:
            assert external_kv is not None, "YOCO Cross-Attention benötigt external_kv aus Prelude!"
            k, v = external_kv
        else:
            if self.use_mla:
                c_kv = self.kv_down(x)
                k = self.k_up(c_kv).view(B, T_q, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_up(c_kv).view(B, T_q, self.n_kv_heads, self.head_dim).transpose(1, 2)
            else:
                k = self.k_proj(x).view(B, T_q, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T_q, self.n_kv_heads, self.head_dim).transpose(1, 2)

            if self.use_qk_norm:
                k = self.k_norm(k)
            k = self.rope.apply(k, positions=q_positions)

        kv_out = (k, v) if return_kv else None

        # GQA Repeat
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_attn = k.repeat_interleave(rep, dim=1)
            v_attn = v.repeat_interleave(rep, dim=1)
        else:
            k_attn, v_attn = k, v

        T_k = k_attn.shape[-2]
        needs_custom_mask = (self.window_size is not None) or (doc_ids is not None) or (q_positions is not None)

        if (
            _flash_attn_func is not None
            and self.use_flash_attn
            and not needs_custom_mask
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            out = _flash_attn_func(
                q.transpose(1, 2),
                k_attn.transpose(1, 2),
                v_attn.transpose(1, 2),
                causal=True,
                window_size=(self.window_size, 0) if self.window_size else (-1, -1),
            )
        elif not needs_custom_mask:
            out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=True)
            out = out.transpose(1, 2)
        else:
            attn_mask = _build_causal_mask(
                T_q,
                T_k,
                device=x.device,
                dtype=q.dtype,
                window_size=self.window_size,
                doc_ids=doc_ids,
                q_positions=q_positions,
            )
            out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=attn_mask, is_causal=False)
            out = out.transpose(1, 2)

        if self.use_head_gating:
            # LFM3 Head-Wise Gating: verhindert Outlier/Activation-Explosion in FP8/INT8
            gates = torch.sigmoid(self.head_gate(x)).unsqueeze(-1).to(out.dtype)
            out = out * gates

        out = out.contiguous().view(B, T_q, self.n_heads * self.head_dim)
        return self.out_proj(out), kv_out
