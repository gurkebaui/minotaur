"""Central configuration for the LFM3 lab.

The project already uses `dataclass` configs (`config.py` at the repo root), so the
LFM3 lab follows the same pattern: every knob is a plain dataclass field, presets are
module level constants, and `to_hf_overrides()` converts a lab config into the keyword
arguments that the Hugging Face config objects understand.

Two layers of configuration exist on purpose:

* :class:`Lfm3Config` / :class:`OmniConfig` describe the *architecture* (widths, depths,
  layer pattern, rope settings).  They are lab-owned: you can edit them, save them next
  to a checkpoint, or print them in a paper.
* :class:`LoadOptions` describes *how* a checkpoint is loaded (device, dtype, attention
  backend, cache).  It never changes the architecture.

Typical use::

    from lfm3.config import LFM3_1_7B_BASE, LoadOptions, TinyLFM3

    cfg  = LFM3_1_7B_BASE                       # architecture preset
    opts = LoadOptions(dtype="bfloat16")        # runtime preset
    # cfg / opts are passed to lfm3.loader.load_base(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def canonical_layer_types(num_layers: int) -> list[str]:
    """The LFM3 layer pattern.

    LFM3 is *tri-hybrid*: every fourth layer is global attention, the layer right after
    a global one (every fourth, starting at index 4) is sliding-window attention, and all
    remaining layers are gated short convolutions.  For the 24-layer model that gives

    * 13 x ``conv``
    *  5 x ``sliding_attention``  (indices 4, 8, 12, 16, 20)
    *  6 x ``full_attention``     (indices 2, 6, 10, 14, 18, 22)

    This is exactly the rule used by ``Lfm3Config.__post_init__`` upstream::

        "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
    """
    return [
        "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
        for i in range(num_layers)
    ]


def _numel(*shape: int) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


# ---------------------------------------------------------------------------
# text model (LFM3-1.7B-Base and the text tower of every other LFM3 model)
# ---------------------------------------------------------------------------


@dataclass
class Lfm3Config:
    """Architecture of an LFM3 text model.

    The defaults mirror ``LiquidAI/LFM3-1.7B-Base`` (config.json, transformers 5.12+).
    Every field maps onto a field of the upstream ``Lfm3Config``; see
    ``to_hf_overrides()`` for the translation.
    """

    # -- identity -----------------------------------------------------------
    model_id: str = "LiquidAI/LFM3-1.7B-Base"
    model_name: str = "LFM3-1.7B-Base"
    # Where to fetch the (tiny) upstream config class from.  Defaults to ``model_id``;
    # random test models point at the real repo so they only need its config files.
    config_source: str | None = None

    # -- transformer shape --------------------------------------------------
    vocab_size: int = 128_000
    hidden_size: int = 2_048
    intermediate_size: int = 7_680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128

    # -- positions ----------------------------------------------------------
    max_position_embeddings: int = 262_144
    sliding_window: int = 256
    partial_rotary_factor: float = 0.5          # only *full* attention layers rotate partially
    rope_theta_full: float = 1_000_000.0
    rope_theta_sliding: float = 10_000.0

    # -- operators ----------------------------------------------------------
    conv_kernel_size: int = 7                    # width of the gated short convolution
    conv_bias: bool = False
    use_sandwich_norm: bool = True               # RMSNorm after operator *and* after FFN
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # -- tokenizer specials -------------------------------------------------
    pad_token_id: int = 124_893
    bos_token_id: int = 124_894
    eos_token_id: int = 124_900

    # -- overrides ----------------------------------------------------------
    # ``None`` means "use the canonical pattern above".  Set a list to experiment
    # with the hybrid layout (e.g. all global attention, or more conv layers).
    layer_types: list[str] | None = None

    # ------------------------------------------------------------------ API
    def resolved_layer_types(self) -> list[str]:
        """Return the per-layer operator list (canonical pattern by default)."""
        if self.layer_types is None:
            return canonical_layer_types(self.num_hidden_layers)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers "
                f"is {self.num_hidden_layers}"
            )
        return list(self.layer_types)

    def counts(self) -> dict[str, int]:
        """How many layers of each operator type the model has."""
        out = {"conv": 0, "sliding_attention": 0, "full_attention": 0}
        for t in self.resolved_layer_types():
            out[t] = out.get(t, 0) + 1
        return out

    def to_hf_overrides(self) -> dict[str, Any]:
        """Translate into kwargs for the upstream ``Lfm3Config``.

        Only architecture-relevant fields are returned; runtime options live in
        :class:`LoadOptions`.
        """
        rope = {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": self.rope_theta_full,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": self.rope_theta_sliding,
            },
        }

        # Special ids of the real checkpoint are outside a shrunken test vocabulary;
        # ``nn.Embedding`` rejects such a padding index, so drop them instead.
        def special(token_id: int | None) -> int | None:
            return token_id if token_id is not None and 0 <= token_id < self.vocab_size else None

        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_window": self.sliding_window,
            "conv_L_cache": self.conv_kernel_size,
            "conv_bias": self.conv_bias,
            "use_sandwich_norm": self.use_sandwich_norm,
            "norm_eps": self.norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "pad_token_id": special(self.pad_token_id),
            "bos_token_id": special(self.bos_token_id),
            "eos_token_id": special(self.eos_token_id),
            "rope_parameters": rope,
            "layer_types": self.resolved_layer_types(),
        }

    # -------------------------------------------------------------- budgets
    def parameter_estimate(self) -> dict[str, int]:
        """Analytic parameter count per component (no checkpoint needed).

        Useful to sanity-check a checkpoint, to size an experiment, or to fill a
        table in a paper.  Tied embeddings are counted once.
        """
        h, i, v = self.hidden_size, self.intermediate_size, self.vocab_size
        nh, nkv, hd = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        k = self.conv_kernel_size

        embed = v * h
        mlp = 3 * h * i
        norms = 4 * h if self.use_sandwich_norm else 2 * h

        attn = (
            h * nh * hd          # q_proj
            + 2 * h * nkv * hd   # k_proj + v_proj
            + nh * hd * h        # o_proj
            + nh * h             # gate_proj (one scalar per head)
            + 2 * hd             # q_norm + k_norm
            + mlp
            + norms
        )
        conv = (
            h * 3 * h            # in_proj
            + h * k              # depthwise conv
            + h * h              # out_proj
            + mlp
            + norms
        )
        per_type = {"full_attention": attn, "sliding_attention": attn, "conv": conv}
        layers = sum(per_type[t] * n for t, n in self.counts().items())
        total = embed + layers + h  # + final RMSNorm
        return {
            "embeddings": embed,
            "layers": layers,
            "final_norm": h,
            "total": total,
            "per_layer": per_type,
        }

    def with_size(self, *, hidden_size: int, num_hidden_layers: int, **kwargs: Any) -> "Lfm3Config":
        """Return a copy scaled down for fast iteration (see ``TinyLFM3``)."""
        changes: dict[str, Any] = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            # keep the width/expansion ratio of the original model when possible
            "intermediate_size": int(round(4 * hidden_size / 64) * 64) or 4 * hidden_size,
            "num_attention_heads": max(1, hidden_size // 64),
            "num_key_value_heads": max(1, hidden_size // 128),
        }
        changes.update(kwargs)
        # derive head_dim last so a custom head count stays consistent
        changes.setdefault("head_dim", changes["hidden_size"] // changes["num_attention_heads"])
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# omni model (vision + audio + depth decoder around the same text backbone)
# ---------------------------------------------------------------------------


@dataclass
class VisionConfig:
    """LFM3-VL vision tower (used by LFM3-Omni-2B)."""

    hidden_size: int = 1_024
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 2_560
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    in_channels: int = 3
    norm_eps: float = 1e-5

    def layer_types(self) -> list[str]:
        # two conv layers, then alternating conv / global attention
        return ["conv" if i < 2 or i % 2 == 0 else "full_attention" for i in range(self.num_hidden_layers)]

    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size**2


@dataclass
class AudioEncoderConfig:
    """Liquid speech encoder in front of the omni LM.

    Three stride-2 convolutions turn a (mel, time) spectrogram into one vector per 8
    frames; a 17-layer LFM2-style model with a 32-frame sliding window encodes it.
    """

    hidden_size: int = 768
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 2_560
    conv_channels: int = 256
    conv_kernel_size: int = 3
    num_mel_bins: int = 128
    sliding_window: int = 32
    norm_eps: float = 1e-5

    def layer_types(self) -> list[str]:
        """``[conv, conv, full]`` repeated, with a full-attention tail when it fits.

        The shipped 17-layer encoder is ``[conv, conv, full] * 5 + [full, full]``; the
        rule below reproduces that and stays sensible for other depths.
        """
        pattern = ["conv", "conv", "full_attention"]
        types = [pattern[i % 3] for i in range(self.num_hidden_layers)]
        if self.num_hidden_layers % 3 == 2:
            types[-2:] = ["full_attention", "full_attention"]
        return types

    def to_hf_overrides(self) -> dict[str, Any]:
        """Encoder config incl. the nested LFM2 trunk that runs over the spectrogram."""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "conv_channels": self.conv_channels,
            "conv_kernel_size": self.conv_kernel_size,
            "num_mel_bins": self.num_mel_bins,
            "sliding_window": self.sliding_window,
            "norm_eps": self.norm_eps,
            # the trunk is a small LFM2 model; it must be passed explicitly, otherwise
            # the upstream config falls back to the full 17-layer default
            "text_config": {
                "model_type": "lfm2",
                "vocab_size": 2,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_types": self.layer_types(),
                "conv_L_cache": self.conv_kernel_size,
                "max_position_embeddings": 128_000,
                "norm_eps": self.norm_eps,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": None,
                "tie_word_embeddings": True,
            },
        }


@dataclass
class DepthConfig:
    """Depth decoder: bidirectional LFM3 over the codebook axis of one speech frame."""

    hidden_size: int = 1_024
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 32
    intermediate_size: int = 2_816
    max_position_embeddings: int = 12
    norm_eps: float = 1e-5
    partial_rotary_factor: float = 1.0
    rope_theta: float = 1_000_000.0

    def to_hf_overrides(self) -> dict[str, Any]:
        """Depth decoder = bidirectional LFM3 with full attention on the codebook axis."""
        return {
            "model_type": "lfm3",
            "vocab_size": 1,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_types": ["full_attention"] * self.num_hidden_layers,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": self.rope_theta,
                    "partial_rotary_factor": self.partial_rotary_factor,
                }
            },
            "sliding_window": None,
            "use_sandwich_norm": False,
            "use_cache": False,
            "norm_eps": self.norm_eps,
            "tie_word_embeddings": True,
            "pad_token_id": None,
            "bos_token_id": None,
            "eos_token_id": None,
        }


@dataclass
class OmniConfig:
    """Architecture of LFM3-Omni-2B.

    The omni checkpoint is one LM (the LFM3-1.7B text tower) with four attachments:
    a vision tower, a speech encoder, two projectors, and a depth decoder that writes
    speech codebooks.  See ``docs/LFM3_OMNI_2B.md``.
    """

    model_id: str = "LiquidAI/LFM3-Omni-2B"
    model_name: str = "LFM3-Omni-2B"
    config_source: str | None = None

    text_config: Lfm3Config = field(default_factory=lambda: Lfm3Config(
        model_id="LiquidAI/LFM3-Omni-2B", model_name="LFM3-Omni-2B (text tower)",
        max_position_embeddings=128_000,
    ))
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    audio_encoder_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    depth_config: DepthConfig = field(default_factory=DepthConfig)

    # speech / multimodal plumbing
    num_codebooks: int = 12
    codebook_size: int = 2_017
    audio_eos_code: int = 2_016
    maskgit_schedule: list[int] = field(default_factory=lambda: [4, 4, 4])
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    downsample_factor: int = 2
    projector_hidden_size: int = 4_096
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True

    # placeholder tokens (ids inside the 128k text vocabulary)
    audio_token_id: int = 125_031
    audio_out_token_id: int = 125_032
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    audio_start_token_id: int = 125_012
    text_end_token_id: int = 125_014
    mixed_start_token_id: int = 125_015

    def to_hf_overrides(self) -> dict[str, Any]:
        """Nested kwargs for the upstream ``Lfm3OmniConfig``."""
        return {
            "text_config": self.text_config.to_hf_overrides(),
            "vision_config": {
                "hidden_size": self.vision_config.hidden_size,
                "num_hidden_layers": self.vision_config.num_hidden_layers,
                "num_attention_heads": self.vision_config.num_attention_heads,
                "num_key_value_heads": self.vision_config.num_key_value_heads,
                "intermediate_size": self.vision_config.intermediate_size,
                "patch_size": self.vision_config.patch_size,
                "temporal_patch_size": self.vision_config.temporal_patch_size,
                "spatial_merge_size": self.vision_config.spatial_merge_size,
                "conv_kernel_size": self.vision_config.conv_kernel_size,
                "in_channels": self.vision_config.in_channels,
                "norm_eps": self.vision_config.norm_eps,
                "rope_parameters": {"rope_type": "axial", "rope_theta": 1_000_000.0},
            },
            "audio_encoder_config": self.audio_encoder_config.to_hf_overrides(),
            "depth_config": self.depth_config.to_hf_overrides(),
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "audio_eos_code": self.audio_eos_code,
            "maskgit_schedule": self.maskgit_schedule,
            "interleaved_n_text": self.interleaved_n_text,
            "interleaved_n_audio": self.interleaved_n_audio,
            "downsample_factor": self.downsample_factor,
            "projector_hidden_size": self.projector_hidden_size,
            "projector_hidden_act": self.projector_hidden_act,
            "projector_bias": self.projector_bias,
            "audio_token_id": self.audio_token_id,
            "audio_out_token_id": self.audio_out_token_id,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "text_end_token_id": self.text_end_token_id,
            "mixed_start_token_id": self.mixed_start_token_id,
        }


# ---------------------------------------------------------------------------
# runtime options
# ---------------------------------------------------------------------------


@dataclass
class LoadOptions:
    """How to load a checkpoint (never changes the architecture)."""

    device: str | None = None            # None -> cuda if available, else cpu
    dtype: str = "bfloat16"              # bfloat16 / float16 / float32
    attn_implementation: str = "sdpa"    # sdpa / eager / flash_attention_2 / flex_attention
    cache_dir: str | None = None         # HF cache override
    local_files_only: bool = False       # never touch the network
    low_cpu_mem_usage: bool = True       # stream weights instead of one big copy
    random_init: bool = False            # skip the checkpoint, random weights (testing)
    load_tokenizer: bool = True          # set False for random tiny models without a tokenizer
    trust_remote_code: bool = True       # required: LFM3 ships its own modeling code

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch is a hard dependency in practice
            return "cpu"

    def torch_dtype(self):
        import torch

        dtype = DTYPES.get(str(self.dtype).lower())
        if dtype is None:
            raise ValueError(f"unknown dtype {self.dtype!r}; choose from {sorted(DTYPES)}")
        return getattr(torch, dtype)


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

# The model the lab is built around.
LFM3_1_7B_BASE = Lfm3Config()

# The text tower inside LFM3-Omni-2B (128k context instead of 256k).
LFM3_OMNI_TEXT = Lfm3Config(
    model_id="LiquidAI/LFM3-Omni-2B",
    model_name="LFM3-Omni-2B (text tower)",
    max_position_embeddings=128_000,
)

# Full omni architecture.
LFM3_OMNI_2B = OmniConfig()

# Tiny random models: same architecture, a few layers and a small width.  They load in
# milliseconds, need no download, and are perfect for testing new code paths.
TinyLFM3 = Lfm3Config(
    model_id="tiny-random",
    model_name="tiny-random",
    config_source="LiquidAI/LFM3-1.7B-Base",
    vocab_size=512,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    layer_types=canonical_layer_types(6),
)

# A slightly bigger random model for quick sanity checks of attention behaviour.
MiniLFM3 = Lfm3Config(
    model_id="mini-random",
    model_name="mini-random",
    config_source="LiquidAI/LFM3-1.7B-Base",
    vocab_size=2_000,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=32,
    max_position_embeddings=1_024,
)

# A tiny omni model: same wiring as LFM3-Omni-2B, but small enough to build in a second.
# Useful to test omni code paths without downloading 4.33 GB.
TinyOmni = OmniConfig(
    model_id="tiny-omni-random",
    model_name="tiny-omni-random",
    config_source="LiquidAI/LFM3-Omni-2B",
    text_config=replace(
        TinyLFM3,
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    ),
    vision_config=VisionConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        patch_size=4,
    ),
    audio_encoder_config=AudioEncoderConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    ),
    depth_config=DepthConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=4,
    ),
    # the placeholder ids must live inside the tiny text vocabulary
    num_codebooks=4,
    codebook_size=32,
    audio_eos_code=31,
    maskgit_schedule=[1, 1, 2],
    audio_token_id=500,
    audio_out_token_id=501,
    image_token_id=502,
    video_token_id=503,
    audio_start_token_id=504,
    text_end_token_id=505,
    mixed_start_token_id=506,
)

# Ready-made load options.
DEFAULT_LOAD = LoadOptions()
CPU_LOAD = LoadOptions(device="cpu", dtype="float32", attn_implementation="eager")
TINY_LOAD = LoadOptions(
    device="cpu", dtype="float32", attn_implementation="eager", random_init=True, load_tokenizer=False
)


__all__ = [
    "Lfm3Config",
    "OmniConfig",
    "VisionConfig",
    "AudioEncoderConfig",
    "DepthConfig",
    "LoadOptions",
    "canonical_layer_types",
    "LFM3_1_7B_BASE",
    "LFM3_OMNI_TEXT",
    "LFM3_OMNI_2B",
    "TinyLFM3",
    "MiniLFM3",
    "TinyOmni",
    "DEFAULT_LOAD",
    "CPU_LOAD",
    "TINY_LOAD",
]