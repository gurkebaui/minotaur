"""Loading LFM3 checkpoints into a research-friendly bundle.

Public entry points
-------------------
``load_base(...)``   Load ``LiquidAI/LFM3-1.7B-Base`` (or any LFM3 text checkpoint).
``load_omni(...)``   Load ``LiquidAI/LFM3-Omni-2B`` (vision + audio + speech output).
``load_config(...)`` Load only the Hugging Face config (no weights).

All three return (or build on) a :class:`ModelBundle`, which bundles

* ``model``       -- the Hugging Face model (``trust_remote_code`` is required, LFM3 is
  not part of ``transformers`` yet),
* ``tokenizer``   -- and, for omni, ``processor``,
* ``access``      -- a :class:`lfm3.access.ModelAccess` view with shortcuts for the
  embedding matrix, every decoder block, the final norm, the LM head, hooks and
  surgery helpers.

The loader is deliberately thin: it only deals with device/dtype, config overrides and
one compatibility shim.  Everything you want to *do* with the model lives in
``lfm3.access`` and ``lfm3.modules``.

Compatibility note
------------------
The checkpoints were exported with ``transformers`` 5.12.  Since 5.13 the base class
validates ``rope_parameters`` for *every* entry of ``layer_types`` -- including the
``conv`` layers, which have no rope at all.  :func:`apply_compat_shims` installs a tiny,
idempotent patch that tolerates missing entries.  Without it ``AutoConfig.from_pretrained``
raises ``KeyError: 'conv'`` on this environment (transformers 5.17).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import DEFAULT_LOAD, LFM3_1_7B_BASE, LFM3_OMNI_2B, LoadOptions, OmniConfig, Lfm3Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# compatibility shim
# ---------------------------------------------------------------------------

_SHIM_INSTALLED = False


def apply_compat_shims() -> None:
    """Make ``transformers`` >= 5.13 accept LFM3's rope config.

    LFM3 has three layer types, but only the two attention types have rope parameters.
    The upstream config therefore stores ``rope_parameters`` as a per-layer-type dict
    with no ``conv`` key.  ``RotaryEmbeddingConfigMixin.standardize_rope_params``
    iterates ``layer_types`` unconditionally and raises ``KeyError`` for ``conv``.

    The patch adds a temporary ``None`` entry for the missing types (the method already
    skips ``None``), then removes the temporary entries plus the scalar keys that
    ``convert_rope_params_to_dict`` injects at the top level.  It is applied once per
    process and only changes behaviour for configs that mix per-layer-type rope with
    rope-less layer types -- i.e. LFM3.
    """
    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return

    from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin

    original = RotaryEmbeddingConfigMixin.standardize_rope_params

    def standardize_rope_params(self):  # type: ignore[no-untyped-def]
        layer_types = list(getattr(self, "layer_types", None) or [])
        rope_parameters = getattr(self, "rope_parameters", None)
        if not isinstance(rope_parameters, dict) or not layer_types:
            return original(self)

        per_layer_type = any(key in rope_parameters for key in layer_types)
        added = [layer_type for layer_type in layer_types if layer_type not in rope_parameters]
        for layer_type in added:
            rope_parameters[layer_type] = None
        try:
            return original(self)
        finally:
            for layer_type in added:
                rope_parameters.pop(layer_type, None)
            if per_layer_type:
                # ``convert_rope_params_to_dict`` also sets a scalar "rope_theta"; that
                # key is meaningless next to per-layer entries and trips validation.
                for key in [k for k in list(rope_parameters) if k not in layer_types]:
                    rope_parameters.pop(key, None)

    RotaryEmbeddingConfigMixin.standardize_rope_params = standardize_rope_params

    # ``transformers`` >= 5.17 remaps the legacy layer-type name "conv" to
    # "linear_attention" for *custom-code* configs.  LFM3's remote modeling code was
    # written against "conv" and would build the wrong masks, so the remap is disabled
    # for this process.  Native models (LFM2) do not use the remap at all.
    try:
        from transformers import configuration_utils as _configuration_utils

        _configuration_utils._LEGACY_LAYER_TYPE_REMAP.pop("conv", None)
    except Exception:  # pragma: no cover - older/newer transformers
        logger.debug("could not disable the legacy 'conv' layer-type remap")

    _SHIM_INSTALLED = True


def _restore_lfm3_layer_types(config: Any) -> Any:
    """Undo a legacy ``conv`` -> ``linear_attention`` remap on LFM3 configs.

    Safety net for configs that were constructed before :func:`apply_compat_shims` ran
    (or by a code path that validates again).  LFM3 has no linear-attention layers, so
    every ``linear_attention`` entry in an ``lfm3`` config is a remapped ``conv``.
    """
    candidates = [config]
    for name in ("text_config", "vision_config", "audio_encoder_config", "depth_config"):
        sub = getattr(config, name, None)
        if sub is not None:
            candidates.append(sub)
    for candidate in candidates:
        if getattr(candidate, "model_type", None) != "lfm3":
            continue
        layer_types = getattr(candidate, "layer_types", None)
        if layer_types and "linear_attention" in layer_types:
            candidate.layer_types = ["conv" if t == "linear_attention" else t for t in layer_types]
    return config


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

# keys that `PreTrainedConfig` injects itself and that the LFM3 config rejects
_CONFIG_KEYS_TO_DROP = ("auto_map", "architectures", "transformers_version", "_name_or_path")


def apply_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    """Apply a flat override dict to an instantiated HF config.

    ``layer_types`` and ``num_hidden_layers`` are applied together: the upstream
    validator rejects a mismatch, and overriding one without the other is a common
    experiment (fewer layers, same pattern).
    """
    overrides = dict(overrides)
    layer_types = overrides.pop("layer_types", None)
    num_layers = overrides.pop("num_hidden_layers", None)

    for key, value in overrides.items():
        # ``None`` is a meaningful value for the special token ids (it means "unset"),
        # so overrides are applied verbatim; callers drop keys they do not want to touch.
        setattr(config, key, value)
    if num_layers is not None:
        config.num_hidden_layers = int(num_layers)
    if layer_types is not None:
        config.layer_types = list(layer_types)
    return config


def load_hf_config(
    model_id: str,
    overrides: dict[str, Any] | None = None,
    *,
    local_files_only: bool = False,
    cache_dir: str | None = None,
    trust_remote_code: bool = True,
) -> Any:
    """Load the upstream config for ``model_id`` and apply ``overrides``."""
    apply_compat_shims()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    if overrides:
        nested = {"text_config", "vision_config", "audio_encoder_config", "depth_config"} & set(overrides)
        if nested:
            # nested sub-configs must go through the constructor: their ``__post_init__``
            # turns dicts into config objects and validates them
            data = config.to_dict()
            data.update(overrides)
            for key in _CONFIG_KEYS_TO_DROP:
                data.pop(key, None)
            config = type(config)(**data)
        else:
            apply_overrides(config, overrides)
    return _restore_lfm3_layer_types(config)


_ALWAYS_APPLY_NONE = ("pad_token_id", "bos_token_id", "eos_token_id")


def _config_from_lab(lab_config: Lfm3Config, overrides: dict[str, Any] | None) -> Any:
    """Build an upstream config from a lab config (requires the small remote .py files)."""
    merged = {**lab_config.to_hf_overrides(), **(overrides or {})}
    # ``to_hf_overrides`` emits None for special ids that do not fit a shrunken
    # vocabulary; those must reach the config, everything else means "leave as is".
    merged = {k: v for k, v in merged.items() if v is not None or k in _ALWAYS_APPLY_NONE}
    return load_hf_config(lab_config.config_source or lab_config.model_id, merged)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_tokenizer(model_id: str, *, local_files_only: bool, cache_dir: str | None) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only, cache_dir=cache_dir, trust_remote_code=True
    )


def load_base(
    cfg: Lfm3Config | None = None,
    opts: LoadOptions | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    tokenizer_id: str | None = None,
    **from_pretrained_kwargs: Any,
) -> "ModelBundle":
    """Load an LFM3 text model.

    Parameters
    ----------
    cfg:
        Architecture preset, defaults to :data:`lfm3.config.LFM3_1_7B_BASE`.
    opts:
        Runtime options (device, dtype, attention backend).  Defaults to
        :data:`lfm3.config.DEFAULT_LOAD`.
    overrides:
        Extra HF config overrides applied on top of ``cfg`` (e.g.
        ``{"layer_types": [...]}`` to change the hybrid pattern).
    tokenizer_id:
        Tokenizer repo, defaults to the model repo.
    **from_pretrained_kwargs:
        Passed straight to ``AutoModelForCausalLM.from_pretrained`` (``device_map``,
        ``quantization_config``, ...).

    Returns
    -------
    ModelBundle
        ``bundle.model`` is the HF ``Lfm3ForCausalLM``; ``bundle.access`` gives the
        structural view described in ``lfm3.access``.
    """
    cfg = cfg or LFM3_1_7B_BASE
    opts = opts or DEFAULT_LOAD
    apply_compat_shims()

    from transformers import AutoModelForCausalLM

    hf_config = _config_from_lab(cfg, overrides)
    device = _resolve_device(opts.device)

    if opts.random_init:
        logger.info("random init (%s), no checkpoint is read", cfg.model_id)
        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=opts.trust_remote_code)
        model = model.to(device=device, dtype=opts.torch_dtype())
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            config=hf_config,
            trust_remote_code=opts.trust_remote_code,
            dtype=opts.torch_dtype(),
            attn_implementation=opts.attn_implementation,
            cache_dir=opts.cache_dir,
            local_files_only=opts.local_files_only,
            **from_pretrained_kwargs,
        )
        if device != "cpu" or from_pretrained_kwargs.get("device_map") is None:
            model = model.to(device)

    tokenizer = None
    if opts.load_tokenizer:
        tokenizer = _load_tokenizer(
            tokenizer_id or cfg.model_id, local_files_only=opts.local_files_only, cache_dir=opts.cache_dir
        )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=hf_config,
        lab_config=cfg,
        name=cfg.model_name,
        device=device,
        dtype=opts.torch_dtype(),
    )


def load_omni(
    cfg: OmniConfig | None = None,
    opts: LoadOptions | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    **from_pretrained_kwargs: Any,
) -> "ModelBundle":
    """Load LFM3-Omni-2B (text + vision + audio + depth decoder).

    Uses ``AutoModelForImageTextToText`` because that is the auto class the omni
    checkpoint registers.  The returned bundle also carries ``processor`` (image/audio
    preprocessing) in addition to the tokenizer.

    The text tower is available through ``bundle.text_model`` and behaves exactly like
    the model returned by :func:`load_base`; the extra towers are reachable through
    ``bundle.model.model.vision_tower``, ``.audio_encoder``, ``.depth_decoder`` etc.
    """
    cfg = cfg or LFM3_OMNI_2B
    opts = opts or DEFAULT_LOAD
    apply_compat_shims()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    merged = {**cfg.to_hf_overrides(), **(overrides or {})}
    hf_config = load_hf_config(
        cfg.config_source or cfg.model_id,
        merged,
        local_files_only=opts.local_files_only,
        cache_dir=opts.cache_dir,
        trust_remote_code=opts.trust_remote_code,
    )
    device = _resolve_device(opts.device)

    if opts.random_init:
        # ``AutoModelForImageTextToText.from_config`` does not resolve the remote-code
        # auto_map, so the class is imported directly for random test models
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_cls = get_class_from_dynamic_module(
            "modeling_lfm3_omni.Lfm3OmniForConditionalGeneration", cfg.config_source or cfg.model_id
        )
        model = model_cls(hf_config)
        model = model.to(device=device, dtype=opts.torch_dtype())
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_id,
            config=hf_config,
            trust_remote_code=opts.trust_remote_code,
            dtype=opts.torch_dtype(),
            attn_implementation=opts.attn_implementation,
            cache_dir=opts.cache_dir,
            local_files_only=opts.local_files_only,
            **from_pretrained_kwargs,
        )
        if device != "cpu" or from_pretrained_kwargs.get("device_map") is None:
            model = model.to(device)

    tokenizer = None
    if opts.load_tokenizer:
        tokenizer = _load_tokenizer(cfg.model_id, local_files_only=opts.local_files_only, cache_dir=opts.cache_dir)
    processor = None
    if not opts.random_init:
        try:
            processor = AutoProcessor.from_pretrained(
                cfg.model_id,
                local_files_only=opts.local_files_only,
                cache_dir=opts.cache_dir,
                trust_remote_code=opts.trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - processor is optional for text work
            logger.warning("could not load the omni processor: %s", exc)

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        config=hf_config,
        lab_config=cfg,
        name=cfg.model_name,
        device=device,
        dtype=opts.torch_dtype(),
    )


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """A loaded model plus everything the lab needs to work with it.

    Attributes
    ----------
    model:
        The Hugging Face model.  ``model.model`` is the LFM3 text backbone; for omni
        it is ``model.model.language_model`` (see :attr:`text_model`).
    tokenizer:
        The matching tokenizer.
    processor:
        Omni image/audio processor, ``None`` for text-only checkpoints.
    config:
        The instantiated upstream config.
    lab_config:
        The lab-owned architecture dataclass this bundle was built from.
    access:
        :class:`lfm3.access.ModelAccess` -- the structural / experimental API.
    """

    model: Any
    tokenizer: Any
    config: Any
    lab_config: Lfm3Config | OmniConfig
    name: str
    device: str
    dtype: torch.dtype
    processor: Any = None
    access: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.access is None:
            from .access import ModelAccess

            self.access = ModelAccess(self)

    # -- convenience --------------------------------------------------------
    @property
    def text_model(self) -> Any:
        """The text backbone (``model.model`` for base, ``model.model.language_model`` for omni)."""
        inner = getattr(self.model, "model", None)
        if inner is None:
            return self.model
        return getattr(inner, "language_model", inner)

    @property
    def is_omni(self) -> bool:
        return hasattr(self.text_model, "language_model") or "Omni" in type(self.model).__name__

    def to(self, device: str | None = None, dtype: torch.dtype | str | None = None) -> "ModelBundle":
        """Move/cast the model in place and return ``self``."""
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if device is not None:
            self.model.to(device)
            self.device = device
        if dtype is not None:
            self.model.to(dtype)
            self.dtype = dtype
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path: str) -> None:
        """Save the *full* model state dict (weights + any attached modules)."""
        torch.save(
            {
                "name": self.name,
                "lab_config": self.lab_config,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    def load(self, path: str, strict: bool = False) -> None:
        """Load a state dict produced by :meth:`save` (non-strict by default)."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload.get("state_dict", payload)
        self.model.load_state_dict(state, strict=strict)

    def describe(self) -> str:
        from .inspect import describe

        return describe(self)


def load_tiny(**kwargs: Any) -> ModelBundle:
    """A tiny random model for tests: ``load_base(TinyLFM3, TINY_LOAD)``."""
    from .config import TINY_LOAD, TinyLFM3

    opts = kwargs.pop("opts", TINY_LOAD)
    return load_base(kwargs.pop("cfg", TinyLFM3), opts, **kwargs)


def load_tiny_omni(**kwargs: Any) -> ModelBundle:
    """A tiny random omni model for tests: ``load_omni(TinyOmni, opts with random_init=True)``."""
    from .config import TINY_LOAD, TinyOmni

    opts = kwargs.pop("opts", LoadOptions(device="cpu", dtype="float32", attn_implementation="eager",
                                          random_init=True, load_tokenizer=False))
    return load_omni(kwargs.pop("cfg", TinyOmni), opts, **kwargs)


__all__ = [
    "ModelBundle",
    "apply_compat_shims",
    "apply_overrides",
    "load_hf_config",
    "load_base",
    "load_omni",
    "load_tiny",
    "load_tiny_omni",
]