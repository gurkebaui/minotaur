"""A dependency-light PyTorch re-implementation of the LFM3 blocks.

Why this file exists
--------------------
``transformers`` gives you the *original* LFM3 modules, and :class:`lfm3.access.ModelAccess`
lets you replace or edit them.  This module goes one step further: it re-implements the
same architecture in plain PyTorch (~250 lines, no transformers internals, no caches), so
you can

* read the maths next to the code that runs it (see ``docs/LFM3_1.7B_BASE.md``),
* change a block without fighting the masking/cache machinery of ``transformers``,
* prototype a new operator and *verify* it against the original with
  :func:`lfm3.experiments.demo_reference_parity`.

The module and parameter names are identical to the Hugging Face implementation, so a
reference model can be filled directly from a checkpoint::

    hf_backbone = bundle.access.backbone
    ref = ReferenceBackbone.from_hf(hf_backbone)   # loads the weights, no copy of the file

What is intentionally *not* implemented
---------------------------------------
* KV / conv caches (generation is done by the HF model).
* packed documents (``position_ids`` restarting) and padding masks.
* the omni towers (vision/audio) -- they are documented but not re-implemented.

Everything here is inference-oriented: no dropout, no training loop.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _attr(config: Any, name: str, default: Any = None) -> Any:
    """Read a field from an HF config, a lab config or a plain dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _rope_parameters(config: Any, layer_type: str) -> tuple[float, float]:
    """(theta, partial_rotary_factor) for a layer type."""
    rope = _attr(config, "rope_parameters", {}) or {}
    params = rope.get(layer_type, {}) if isinstance(rope, dict) else {}
    theta = float(params.get("rope_theta", _attr(config, "rope_theta", 10_000.0)))
    partial = float(params.get("partial_rotary_factor", 1.0))
    return theta, partial


def _rope_tables(
    head_dim: int, theta: float, partial: float, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape ``(batch, seq, head_dim)``.

    LFM3 uses a *zero-tail* partial rotation: the first
    ``int(partial * head_dim) // 2`` frequencies follow the full-dimension progression,
    the remaining frequencies are zero and those dimensions stay unrotated.
    """
    rope_angles = int(partial * head_dim) // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim))
    inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - rope_angles, dtype=torch.float32)])
    freqs = position_ids[..., None].float() * inv_freq.to(position_ids.device)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE in fp32 and cast back, exactly like the HF implementation."""
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


class ReferenceRMSNorm(nn.Module):
    """RMSNorm with a fp32 variance (LFM3 / T5 style)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ReferenceMLP(nn.Module):
    """SwiGLU feed-forward: ``w2(silu(w1(x)) * w3(x))`` (no biases)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ReferenceShortConv(nn.Module):
    """Gated causal depthwise convolution (the ``conv`` operator).

    ``in_proj`` produces three streams ``(B, C, x)``; the hidden stream is convolved and
    the result is gated by the third one::

        h = conv1d(B * x)          # depthwise, kernel = conv_L_cache, causal
        y = out_proj(C * h)

    This is the whole "local mixing" operator: it looks at the last ``kernel`` tokens and
    nothing else.
    """

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, groups=hidden_size, bias=bias,
            padding=kernel_size - 1,
        )
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        projected = self.in_proj(hidden_states).transpose(-1, -2)  # (B, 3H, T)
        gate, value, x = projected.chunk(3, dim=-2)
        hidden_states = gate * x
        conv_out = self.conv(hidden_states)[..., :seq_len]         # causal: drop the right pad
        out = value * conv_out
        return self.out_proj(out.transpose(-1, -2).contiguous())


class ReferenceAttention(nn.Module):
    """Grouped-query attention with QK-norm, partial RoPE and a per-head output gate.

    Differences to vanilla Llama attention, all of them present in LFM3:

    * ``q_norm`` / ``k_norm`` RMSNorm over ``head_dim`` before the RoPE,
    * ``partial_rotary_factor`` (0.5 on full-attention layers) -- half of every head is
      rotated, the rest carries no positional information,
    * a sigmoid gate per head, computed from the *input* hidden state, multiplied onto
      the attention output before ``o_proj``,
    * ``sliding_attention`` layers only see the last ``sliding_window`` positions.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_type: str,
        sliding_window: int | None = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.layer_type = layer_type
        self.sliding_window = sliding_window if layer_type == "sliding_attention" else None

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = ReferenceRMSNorm(head_dim, eps=norm_eps)
        self.k_norm = ReferenceRMSNorm(head_dim, eps=norm_eps)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        shape = (batch, seq, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(*shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(*shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(*shape).transpose(1, 2)

        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)

        # grouped-query: repeat the kv heads to match the query heads
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        mask = self._causal_mask(seq, hidden_states.device, hidden_states.dtype)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(weights, value)                       # (B, H, T, D)

        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)  # (B, T, H, 1)
        out = (out.transpose(1, 2) * gate).reshape(batch, seq, -1)
        return self.o_proj(out)

    def _causal_mask(self, seq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = torch.arange(seq, device=device)[:, None]
        k = torch.arange(seq, device=device)[None, :]
        allowed = k <= q
        if self.sliding_window is not None:
            allowed = allowed & (k > q - self.sliding_window)
        mask = torch.zeros(seq, seq, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, float("-inf"))


class ReferenceDecoderLayer(nn.Module):
    """One LFM3 block: operator + FFN, both pre-normed and (optionally) sandwich-normed.

    Residual stream::

        x = x + post_operator_norm(operator(operator_norm(x)))
        x = x + post_ffn_norm(ffn(ffn_norm(x)))

    With ``use_sandwich_norm=False`` the post norms are identities (the LFM2 layout).
    """

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        layer_types = list(_attr(config, "layer_types"))
        self.layer_type = layer_types[layer_idx]
        self.use_sandwich_norm = bool(_attr(config, "use_sandwich_norm", True))

        hidden = _attr(config, "hidden_size")
        head_dim = _attr(config, "head_dim") or hidden // _attr(config, "num_attention_heads")
        norm_eps = _attr(config, "norm_eps", 1e-5)

        if self.layer_type == "conv":
            self.conv = ReferenceShortConv(hidden, _attr(config, "conv_L_cache", 7), _attr(config, "conv_bias", False))
        else:
            self.self_attn = ReferenceAttention(
                hidden_size=hidden,
                num_heads=_attr(config, "num_attention_heads"),
                num_kv_heads=_attr(config, "num_key_value_heads"),
                head_dim=head_dim,
                layer_type=self.layer_type,
                sliding_window=_attr(config, "sliding_window"),
                norm_eps=norm_eps,
            )
        self.feed_forward = ReferenceMLP(hidden, _attr(config, "intermediate_size"))
        self.operator_norm = ReferenceRMSNorm(hidden, eps=norm_eps)
        self.ffn_norm = ReferenceRMSNorm(hidden, eps=norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = ReferenceRMSNorm(hidden, eps=norm_eps)
            self.post_ffn_norm = ReferenceRMSNorm(hidden, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.layer_type == "conv":
            operator_out = self.conv(self.operator_norm(hidden_states))
        else:
            operator_out = self.self_attn(self.operator_norm(hidden_states), cos, sin)
        if self.use_sandwich_norm:
            operator_out = self.post_operator_norm(operator_out)
        hidden_states = residual + operator_out

        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        return hidden_states + ffn_out


# ---------------------------------------------------------------------------
# backbone
# ---------------------------------------------------------------------------


class ReferenceBackbone(nn.Module):
    """Embeddings + decoder layers + final RMSNorm.

    Module names match the Hugging Face ``Lfm3Model``, so ``from_hf`` is a plain
    ``load_state_dict``.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.layer_types = list(_attr(config, "layer_types"))
        self.embed_tokens = nn.Embedding(
            _attr(config, "vocab_size"), _attr(config, "hidden_size"), _attr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList([ReferenceDecoderLayer(config, i) for i in range(len(self.layer_types))])
        self.embedding_norm = ReferenceRMSNorm(_attr(config, "hidden_size"), eps=_attr(config, "norm_eps", 1e-5))

    # -- construction -------------------------------------------------------
    @classmethod
    def from_hf(cls, hf_backbone: nn.Module) -> "ReferenceBackbone":
        """Build from an HF backbone and copy its weights (no file access).

        The backbone must be *pristine*: LoRA (``lora_A`` / ``lora_B``) or an
        ``EmbeddingAdapter`` change the parameter names and cannot be loaded by name.
        """
        model = cls(hf_backbone.config)
        try:
            model.load_state_dict(hf_backbone.state_dict(), strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "could not copy the HF weights by name; LoRA and embedding adapters change "
                f"the parameter names. Original error: {exc}"
            ) from exc
        return model

    @classmethod
    def from_lab_config(cls, lab_config: Any) -> "ReferenceBackbone":
        from .config import Lfm3Config

        assert isinstance(lab_config, Lfm3Config)
        return cls(lab_config.to_hf_overrides())

    # -- forward ------------------------------------------------------------
    def rope_tables(self, position_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """cos/sin per layer type (computed once per forward pass)."""
        head_dim = _attr(self.config, "head_dim")
        tables = {}
        for layer_type in sorted(set(self.layer_types)):
            if layer_type == "conv":
                continue
            theta, partial = _rope_parameters(self.config, layer_type)
            tables[layer_type] = _rope_tables(head_dim, theta, partial, position_ids)
        return tables

    def forward_hidden(
        self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        seq = hidden.shape[1]
        position_ids = torch.arange(seq, device=hidden.device)[None, :]
        tables = self.rope_tables(position_ids)
        for layer in self.layers:
            cos, sin = tables.get(layer.layer_type, (None, None))
            hidden = layer(hidden, cos, sin)
        return self.embedding_norm(hidden)

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None):
        """Return a dict with ``hidden_states`` and ``logits`` when a head is attached."""
        hidden = self.forward_hidden(input_ids, inputs_embeds)
        out = {"hidden_states": hidden}
        head = getattr(self, "lm_head", None)
        if head is not None:
            out["logits"] = head(hidden)
        return out

    def attach_lm_head(self, head: nn.Module) -> nn.Module:
        self.lm_head = head
        return head

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "ReferenceRMSNorm",
    "ReferenceMLP",
    "ReferenceShortConv",
    "ReferenceAttention",
    "ReferenceDecoderLayer",
    "ReferenceBackbone",
]