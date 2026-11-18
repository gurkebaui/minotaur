"""Human-readable architecture and parameter reports.

``describe(bundle)`` prints what is *actually loaded* (dtypes, parameter groups, tied
heads).  ``describe_architecture(cfg)`` prints what a lab config *would* build, with the
analytic parameter budget -- no checkpoint required, which makes it useful for docs and
for checking a config before a long download.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import Lfm3Config, OmniConfig

logger = logging.getLogger(__name__)


def _fmt(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n/1e9:.3f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(int(n))


def _rule(title: str, width: int = 78) -> str:
    return f"{title}\n" + "-" * width


def describe_architecture(lab_cfg: Lfm3Config | OmniConfig) -> str:
    """Architecture summary derived purely from a lab config."""
    lines: list[str] = []
    if isinstance(lab_cfg, OmniConfig):
        lines.append(_rule(f"{lab_cfg.model_name} (LFM3-Omni)"))
        lines.append(f"repo            : {lab_cfg.model_id}")
        lines.append(f"text tower      : {lab_cfg.text_config.model_name}")
        lines.append(f"vision tower    : {lab_cfg.vision_config.num_hidden_layers} layers, "
                     f"hidden={lab_cfg.vision_config.hidden_size}, patch={lab_cfg.vision_config.patch_size}")
        lines.append(f"audio encoder   : {lab_cfg.audio_encoder_config.num_hidden_layers} layers, "
                     f"hidden={lab_cfg.audio_encoder_config.hidden_size}, window={lab_cfg.audio_encoder_config.sliding_window}")
        lines.append(f"depth decoder   : {lab_cfg.depth_config.num_hidden_layers} layers, "
                     f"hidden={lab_cfg.depth_config.hidden_size}, codebooks={lab_cfg.num_codebooks}")
        text_cfg: Lfm3Config = lab_cfg.text_config
        lines.append("")
        lines.append(_rule("text tower"))
    else:
        text_cfg = lab_cfg
        lines.append(_rule(f"{lab_cfg.model_name}"))
        lines.append(f"repo            : {lab_cfg.model_id}")

    counts = text_cfg.counts()
    lines += [
        f"parameters      : ~{_fmt(text_cfg.parameter_estimate()['total'])} (analytic)",
        f"vocab           : {text_cfg.vocab_size}",
        f"hidden          : {text_cfg.hidden_size}",
        f"intermediate    : {text_cfg.intermediate_size}",
        f"layers          : {text_cfg.num_hidden_layers}  {counts}",
        f"attention       : {text_cfg.num_attention_heads} heads / {text_cfg.num_key_value_heads} kv heads, "
        f"head_dim={text_cfg.head_dim}",
        f"rope            : full theta={text_cfg.rope_theta_full:g}, partial={text_cfg.partial_rotary_factor}; "
        f"sliding theta={text_cfg.rope_theta_sliding:g}",
        f"sliding window  : {text_cfg.sliding_window}",
        f"short conv      : kernel={text_cfg.conv_kernel_size}, bias={text_cfg.conv_bias}, gated",
        f"sandwich norm   : {text_cfg.use_sandwich_norm}",
        f"norm eps        : {text_cfg.norm_eps:g}",
        f"tied embeddings : {text_cfg.tie_word_embeddings}",
        f"context         : {text_cfg.max_position_embeddings}",
        "",
        _rule("layer pattern (0 = first block)"),
    ]
    for i, layer_type in enumerate(text_cfg.resolved_layer_types()):
        lines.append(f"  [{i:2d}] {layer_type}")
    return "\n".join(lines)


def describe(bundle: Any) -> str:
    """Full report of a loaded :class:`~lfm3.loader.ModelBundle`."""
    model = bundle.model
    cfg = bundle.config
    lines: list[str] = []
    lines.append(_rule(f"{bundle.name}"))
    lines.append(f"class           : {type(model).__name__}")
    lines.append(f"device / dtype  : {bundle.device} / {bundle.dtype}")
    lines.append(f"parameters      : {_fmt(bundle.parameters())}")

    text_cfg = getattr(cfg, "text_config", cfg)
    lines += [
        f"vocab           : {getattr(text_cfg, 'vocab_size', '?')}",
        f"hidden          : {getattr(text_cfg, 'hidden_size', '?')}",
        f"layers          : {getattr(text_cfg, 'num_hidden_layers', '?')}",
        f"context         : {getattr(text_cfg, 'max_position_embeddings', '?')}",
        "",
        _rule("layer pattern"),
    ]
    lines.append(bundle.access.describe_layers())

    groups = bundle.access.parameter_groups()
    lines += ["", _rule("parameters by group")]
    for name, info in sorted(groups.items(), key=lambda kv: -kv[1]["total"])[:12]:
        lines.append(
            f"  {name:40s} {_fmt(info['total']):>10s}  ({info['tensors']} tensors, "
            f"{_fmt(info['trainable'])} trainable)"
        )

    emb = bundle.access.embedding_report()
    lines += [
        "",
        _rule("embedding"),
        f"  matrix          : {emb['shape']} {emb['dtype']}",
        f"  tied to head    : {emb['tied_to_head']}",
        f"  row norm        : mean {emb['mean_norm']:.4f}, std {emb['std_norm']:.4f}, zero rows {emb['zero_rows']}",
        f"  pad token id    : {emb['padding_idx']}",
    ]

    if bundle.tokenizer is not None:
        specials = getattr(bundle.tokenizer, "all_special_tokens", [])
        lines += [
            "",
            _rule("tokenizer"),
            f"  class           : {type(bundle.tokenizer).__name__}",
            f"  vocab size      : {len(bundle.tokenizer)}",
            f"  special tokens  : {specials[:6]}{' ...' if len(specials) > 6 else ''}",
        ]
    if bundle.processor is not None:
        lines.append(f"  processor       : {type(bundle.processor).__name__}")

    lines += ["", _rule("trainable"), f"  {bundle.access.trainable_summary()}"]
    return "\n".join(lines)


def parameter_budget(lab_cfg: Lfm3Config) -> str:
    """Per-component parameter table (analytic), useful in docs and reviews."""
    est = lab_cfg.parameter_estimate()
    counts = lab_cfg.counts()
    rows = [
        ("embed_tokens", est["embeddings"]),
        ("lm_head", 0 if lab_cfg.tie_word_embeddings else est["embeddings"]),
        ("final norm", est["final_norm"]),
    ]
    for layer_type, per_layer in est["per_layer"].items():
        if counts.get(layer_type):
            rows.append((f"{layer_type} x{counts[layer_type]}", per_layer * counts[layer_type]))
    lines = [f"parameter budget for {lab_cfg.model_name} (analytic)"]
    lines.append(f"{'component':28s}{'params':>14s}{'share':>8s}")
    total = est["total"]
    for name, value in rows:
        if value:
            lines.append(f"{name:28s}{_fmt(value):>14s}{100*value/total:7.2f}%")
    lines.append(f"{'TOTAL':28s}{_fmt(total):>14s}{100.0:7.2f}%")
    return "\n".join(lines)


__all__ = ["describe", "describe_architecture", "parameter_budget"]