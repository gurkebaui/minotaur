"""Plug-in modules for LFM3 experiments.

Everything here is a plain ``torch.nn.Module`` with no dependency on the transformers
internals, so it can be unit-tested on its own and reused in a training script.

| Module                  | Purpose                                                        |
|-------------------------|----------------------------------------------------------------|
| :class:`NgramLayer`     | hashed n-gram features added to the logits or the hidden state  |
| :class:`BottleneckAdapter` | small MLP between final norm and LM head (the "attached model") |
| :class:`TinyTransformerAdapter` | one or two transformer layers in the same slot         |
| :class:`EmbeddingAdapter` | trainable scale/bias/low-rank correction of the input embedding |
| :class:`ProbeHead`      | small classifier on hidden states (interpretability)            |
| :class:`LoRALinear`     | low-rank weight update for fine-tuning                         |
| :func:`inject_lora`     | swap selected ``nn.Linear`` layers for :class:`LoRALinear`      |
| :class:`BlockWrapper`   | adapt a ``hidden -> hidden`` module to the HF block signature   |
| :class:`BlockBypass`    | identity / zero / scaled block for ablations                    |
| :class:`HeadAdapter`    | ``hidden -> adapter -> head`` used by ``attach_after_final_norm`` |

All constructors take an explicit config dataclass where it makes sense, matching the
project's config-driven style.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# A large odd prime; the rolling hash is computed modulo 2**64 by int64 wraparound.
_HASH_PRIME = 1_000_003


def ngram_ids(input_ids: torch.Tensor, n: int) -> torch.Tensor:
    """Hash the last ``n`` token ids of every position into a bucket index.

    ``input_ids`` has shape ``(batch, seq)``.  Position ``t`` hashes
    ``(x[t-n+1], ..., x[t])``; positions without a full window are padded with zeros on
    the left, so the first ``n-1`` positions hash a shorter prefix.

    The hash is a polynomial rolling hash with int64 wraparound, vectorised over the
    batch.  It is *not* collision free: ``num_buckets`` controls the trade-off between
    memory and collisions, which is exactly the knob you want when experimenting.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    ids = input_ids.long()
    batch, seq = ids.shape
    if n == 1:
        return ids
    padded = F.pad(ids, (n - 1, 0), value=0)
    windows = padded.unfold(-1, n, 1)  # (B, T, n)
    hash_value = torch.zeros(batch, seq, dtype=torch.int64, device=ids.device)
    for k in range(n):
        hash_value = hash_value * _HASH_PRIME + windows[..., k]
    return hash_value


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu, "tanh": torch.tanh}[name]


class _Activation(nn.Module):
    """Wrap a functional activation so it can live in an ``nn.Sequential``."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


# ---------------------------------------------------------------------------
# n-gram layer
# ---------------------------------------------------------------------------


@dataclass
class NgramConfig:
    """Configuration of :class:`NgramLayer`.

    ``mode="logit_bias"`` adds a learned vector per hashed n-gram directly to the
    logits (shape ``(batch, seq, vocab)``).  ``mode="hidden_add"`` adds a vector to the
    hidden state before the LM head instead (shape ``(batch, seq, hidden)``), which is
    cheaper in memory and can be trained jointly with a small adapter.

    The output projection is zero-initialised, so a freshly attached layer is a no-op
    and training starts from the unmodified model.
    """

    n: int = 3
    num_buckets: int = 16_384
    rank: int = 64
    mode: str = "logit_bias"
    vocab_size: int = 128_000
    hidden_size: int = 2_048
    scale: float = 1.0
    init_std: float = 0.02
    output_init_std: float = 0.0     # 0 -> zero init (no-op at start)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("logit_bias", "hidden_add"):
            raise ValueError("mode must be 'logit_bias' or 'hidden_add'")
        self.out_features = self.vocab_size if self.mode == "logit_bias" else self.hidden_size


class NgramLayer(nn.Module):
    """Hashed n-gram features, added to logits or hidden states.

    Shape contract::

        forward(input_ids, hidden=None) -> (batch, seq, out_features)

    ``input_ids`` is always required (the layer reads tokens); ``hidden`` is only used
    in ``hidden_add`` mode, where its dtype/device are matched.
    """

    def __init__(self, cfg: NgramConfig):
        super().__init__()
        self.cfg = cfg
        self.bucket_embedding = nn.Embedding(cfg.num_buckets, cfg.rank)
        self.out_proj = nn.Linear(cfg.rank, cfg.out_features, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bucket_embedding.weight, std=self.cfg.init_std)
        nn.init.normal_(self.out_proj.weight, std=self.cfg.output_init_std)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq) token ids -> (batch, seq, rank) n-gram features."""
        buckets = ngram_ids(input_ids, self.cfg.n) % self.cfg.num_buckets
        return self.dropout(self.bucket_embedding(buckets))

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.features(input_ids)
        out = self.out_proj(feat) * self.cfg.scale
        if hidden is not None:
            out = out.to(hidden.dtype)
        return out

    def extra_repr(self) -> str:
        return (
            f"n={self.cfg.n}, buckets={self.cfg.num_buckets}, rank={self.cfg.rank}, "
            f"mode={self.cfg.mode}, out={self.cfg.out_features}"
        )


def make_ngram_layer(bundle: Any, cfg: NgramConfig | None = None) -> NgramLayer:
    """Build an :class:`NgramLayer` sized for ``bundle``."""
    cfg = cfg or NgramConfig()
    cfg.vocab_size = bundle.config.text_config.vocab_size if hasattr(bundle.config, "text_config") else bundle.config.vocab_size
    cfg.hidden_size = bundle.config.text_config.hidden_size if hasattr(bundle.config, "text_config") else bundle.config.hidden_size
    return NgramLayer(cfg)


# ---------------------------------------------------------------------------
# adapters in front of the LM head
# ---------------------------------------------------------------------------


@dataclass
class BottleneckConfig:
    """Small MLP inserted between the final norm and the LM head."""

    bottleneck: int = 512
    layers: int = 2
    activation: str = "gelu"
    residual: bool = True
    scale: float = 1.0
    dropout: float = 0.0
    # zero-init the output projection: with ``residual=True`` the adapter starts as the
    # identity, so attaching it does not change the model until it is trained.
    zero_init_output: bool = True


class BottleneckAdapter(nn.Module):
    """``hidden -> down -> act -> up -> (+ hidden)``.

    This is the "attach a small model before the output embedding" building block.  It
    is shape preserving, so it can be dropped in front of the untouched LM head; give
    ``residual=False`` for a pure bottleneck that actually changes the representation.
    """

    def __init__(self, hidden_size: int, cfg: BottleneckConfig | None = None):
        super().__init__()
        cfg = cfg or BottleneckConfig()
        self.cfg = cfg
        act = _activation(cfg.activation)
        self.down = nn.Linear(hidden_size, cfg.bottleneck)
        self.middle = nn.ModuleList(
            nn.Sequential(nn.Linear(cfg.bottleneck, cfg.bottleneck), nn.LayerNorm(cfg.bottleneck), _Activation(act))
            for _ in range(max(0, cfg.layers - 1))
        )
        self.up = nn.Linear(cfg.bottleneck, hidden_size)
        self.norm = nn.LayerNorm(cfg.bottleneck)
        self.act = _Activation(act)
        self.dropout = nn.Dropout(cfg.dropout)
        self.hidden_size = hidden_size
        if cfg.zero_init_output:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.down(hidden)))
        for layer in self.middle:
            out = layer(out)
        out = self.dropout(self.up(out))
        if self.cfg.residual:
            out = out + hidden
        return out * self.cfg.scale


@dataclass
class TinyTransformerConfig:
    """A one-to-few layer transformer inserted before the LM head."""

    layers: int = 1
    num_heads: int = 8
    ff_multiplier: int = 2
    dropout: float = 0.0
    residual: bool = True
    norm_first: bool = True


class TinyTransformerAdapter(nn.Module):
    """A small ``nn.TransformerEncoder`` in front of the LM head.

    The sequence axis is kept, so the adapter can mix information across positions
    (unlike :class:`BottleneckAdapter`, which is position-wise).
    """

    def __init__(self, hidden_size: int, cfg: TinyTransformerConfig | None = None):
        super().__init__()
        cfg = cfg or TinyTransformerConfig()
        if hidden_size % cfg.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=hidden_size * cfg.ff_multiplier,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers, enable_nested_tensor=False)
        self.cfg = cfg

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder(hidden)


class HeadAdapter(nn.Module):
    """``hidden -> adapter -> head``; used by ``ModelAccess.attach_after_final_norm``."""

    def __init__(self, adapter: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.head = head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(hidden))


# ---------------------------------------------------------------------------
# embedding adapter
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingConfig:
    """Trainable correction of the input embedding."""

    rank: int = 0            # 0 -> only scale and bias
    scale_init: float = 1.0
    bias_init: float = 0.0
    std: float = 0.02


class EmbeddingAdapter(nn.Module):
    """Wrap an embedding matrix and learn ``scale * E[x] + bias + low_rank[x]``.

    The wrapped matrix is kept as-is (``base``), so the adapter is a drop-in
    replacement for ``embed_tokens`` and the LM head can stay tied to the *base*
    weight or to the adapted output -- your choice via ``set_input_embeddings``.
    """

    def __init__(self, base: nn.Embedding, cfg: EmbeddingConfig | None = None):
        super().__init__()
        cfg = cfg or EmbeddingConfig()
        self.base = base
        self.cfg = cfg
        self.scale = nn.Parameter(torch.tensor(cfg.scale_init))
        self.bias = nn.Parameter(torch.full((base.embedding_dim,), cfg.bias_init))
        if cfg.rank > 0:
            self.low_rank = nn.Parameter(torch.empty(cfg.rank, base.embedding_dim))
            nn.init.normal_(self.low_rank, std=cfg.std)
            self.bottleneck = nn.Parameter(torch.empty(base.num_embeddings, cfg.rank))
            nn.init.normal_(self.bottleneck, std=cfg.std)
        else:
            self.low_rank = None
            self.bottleneck = None

    @property
    def num_embeddings(self) -> int:
        return self.base.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        """The effective ``(vocab, hidden)`` matrix (base + low-rank correction)."""
        weight = self.base.weight
        if self.low_rank is not None:
            weight = weight + self.bottleneck @ self.low_rank
        return weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids) * self.scale + self.bias
        if self.low_rank is not None:
            out = out + self.bottleneck[input_ids] @ self.low_rank
        return out


# ---------------------------------------------------------------------------
# probe head
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    """Small classifier on hidden states (linear or MLP)."""

    hidden: int = 0            # 0 -> linear
    activation: str = "gelu"
    dropout: float = 0.0


class ProbeHead(nn.Module):
    """Map hidden states to vocabulary logits; a lightweight alternative LM head."""

    def __init__(self, hidden_size: int, vocab_size: int, cfg: ProbeConfig | None = None):
        super().__init__()
        cfg = cfg or ProbeConfig()
        layers: list[nn.Module] = []
        if cfg.hidden > 0:
            layers += [
                nn.Linear(hidden_size, cfg.hidden),
                _Activation(_activation(cfg.activation)),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, vocab_size, bias=False),
            ]
        else:
            layers += [nn.Linear(hidden_size, vocab_size, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """``y = base(x) + (x @ A^T) @ B^T * alpha/rank``.

    ``base`` is frozen.  The state dict keys change (``...base.weight``), so use
    :func:`lora_state_dict` / :func:`load_lora_state_dict` for LoRA checkpoints.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # match the base layer's device/dtype: injecting LoRA after ``from_pretrained``
        # must not leave the update on the CPU while the model sits on the GPU
        device, dtype = base.weight.device, base.weight.dtype
        lora_a = torch.empty(rank, base.in_features)
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(lora_a.to(device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.lora_A.copy_(
                torch.empty_like(self.lora_A).uniform_(-1.0 / self.lora_A.shape[1], 1.0 / self.lora_A.shape[1])
            )
            self.lora_B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        update = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + update * self.scaling

    def merged(self) -> nn.Linear:
        """Return a copy of ``base`` with the update folded into the weight."""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.data.copy_(self.base.weight.data + (self.lora_B @ self.lora_A) * self.scaling)
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged.to(self.base.weight.device, self.base.weight.dtype)


@dataclass
class LoRAConfig:
    """Which layers get a LoRA update."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # matched against the *end* of parameter names, e.g. "q_proj" matches
    # "model.layers.3.self_attn.q_proj"
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # never wrap these even if they match
    exclude: tuple[str, ...] = ("lm_head",)


def inject_lora(model: nn.Module, cfg: LoRAConfig | None = None) -> dict[str, int]:
    """Replace matching ``nn.Linear`` layers with :class:`LoRALinear`, in place.

    Returns ``{"wrapped": k, "trainable": n, "total": m}`` so a training script can log
    the parameter budget.  The base weights are loaded already (inject *after*
    ``from_pretrained``), which is why the keys of the returned model differ from the
    checkpoint only by the ``.base`` infix and the ``lora_*`` parameters.
    """
    cfg = cfg or LoRAConfig()
    targets = tuple(cfg.targets)
    wrapped = 0

    def matches(name: str) -> bool:
        return name.rsplit(".", 1)[-1] in targets and not any(name.endswith(ex) for ex in cfg.exclude)

    def walk(module: nn.Module, prefix: str = "") -> None:
        nonlocal wrapped
        for child_name, child in list(module.named_children()):
            path = f"{prefix}{child_name}"
            if isinstance(child, nn.Linear) and matches(path):
                setattr(module, child_name, LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout))
                wrapped += 1
            else:
                walk(child, path + ".")

    walk(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"wrapped": wrapped, "trainable": trainable, "total": total}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the LoRA parameters (small checkpoints)."""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if "lora_" in name}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], strict: bool = True) -> Any:
    """Load a LoRA state dict (from :func:`lora_state_dict`).

    ``strict=True`` additionally refuses keys that do not exist in the model, which
    catches a rank or target mismatch between save and load.
    """
    incompatible = model.load_state_dict(state, strict=False)
    if strict and incompatible.unexpected_keys:
        raise KeyError(f"lora state has unknown keys: {incompatible.unexpected_keys[:5]}")
    return incompatible


def mark_only_lora_trainable(model: nn.Module, train_embeddings: bool = False) -> None:
    """Freeze everything except LoRA parameters (and optionally embeddings/head)."""
    for name, param in model.named_parameters():
        param.requires_grad_(("lora_" in name) or (train_embeddings and "embed_tokens" in name))


# ---------------------------------------------------------------------------
# block surgery helpers
# ---------------------------------------------------------------------------


class BlockWrapper(nn.Module):
    """Adapt a ``hidden -> hidden`` module to the Hugging Face block signature.

    Hugging Face calls every decoder layer as::

        layer(hidden_states, position_embeddings=..., attention_mask=...,
              position_ids=..., past_key_values=..., seq_idx=..., **kwargs)

    Anything that only wants the hidden state can be wrapped in this class, e.g.::

        acc.replace_block(4, BlockWrapper(my_mlp))
    """

    def __init__(self, inner: nn.Module | Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.inner = inner

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        seq_idx: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.inner(hidden_states)

    def extra_repr(self) -> str:
        return f"inner={type(self.inner).__name__}"


class BlockBypass(nn.Module):
    """Identity / zero / scaled block for ablations.

    ``mode="identity"`` returns the input unchanged, ``"zero"`` returns zeros, and
    ``"scale"`` returns ``alpha * hidden_states``.
    """

    def __init__(self, mode: str = "identity", alpha: float = 1.0):
        super().__init__()
        if mode not in ("identity", "zero", "scale"):
            raise ValueError("mode must be identity, zero or scale")
        self.mode = mode
        self.alpha = alpha

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        seq_idx: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.mode == "identity":
            return hidden_states
        if self.mode == "zero":
            return torch.zeros_like(hidden_states)
        return hidden_states * self.alpha


def clone_block(block: nn.Module) -> nn.Module:
    """Deep-copy a decoder block so it can be modified without touching the original."""
    import copy

    return copy.deepcopy(block)


class LambdaAdapter(nn.Module):
    """Wrap a plain function as a module (``hidden -> hidden``).

    Handy for quick experiments that do not deserve their own class::

        acc.attach_after_final_norm(LambdaAdapter(lambda h: h * 1.1))
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fn(hidden)


# ---------------------------------------------------------------------------
# small factory used by the CLI / experiments
# ---------------------------------------------------------------------------


def build_n_gram(bundle: Any, cfg: NgramConfig | None = None) -> NgramLayer:
    """Create an :class:`NgramLayer` sized for ``bundle`` and remember it on the bundle.

    The layer is *not* wired into the model automatically: an n-gram bias needs the token
    ids, which the LM head does not receive.  Use it through
    ``bundle.access.forward_with_ngram(input_ids, ngram)`` (works for training and
    teacher forcing) or add it to the hidden state with a normal adapter::

        ngram = build_n_gram(bundle, NgramConfig(n=3, mode="hidden_add"))
        acc.attach_after_final_norm(LambdaAdapter(lambda h: h + ngram(ids, h)))

    ``bundle.ngram`` is set to the new layer so experiments can find it again.
    """
    layer = make_ngram_layer(bundle, cfg)
    bundle.ngram = layer
    return layer


__all__ = [
    "NgramConfig",
    "NgramLayer",
    "make_ngram_layer",
    "BottleneckConfig",
    "BottleneckAdapter",
    "TinyTransformerConfig",
    "TinyTransformerAdapter",
    "HeadAdapter",
    "EmbeddingConfig",
    "EmbeddingAdapter",
    "ProbeConfig",
    "ProbeHead",
    "LoRALinear",
    "LoRAConfig",
    "inject_lora",
    "lora_state_dict",
    "load_lora_state_dict",
    "mark_only_lora_trainable",
    "BlockWrapper",
    "BlockBypass",
    "LambdaAdapter",
    "clone_block",
    "ngram_ids",
    "build_n_gram",
]
