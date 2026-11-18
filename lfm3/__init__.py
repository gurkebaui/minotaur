"""LFM3 research lab: load, inspect and modify Liquid AI's LFM3 models.

Quick start::

    from lfm3 import load_base
    bundle = load_base()                 # downloads ~3.5 GB on first use
    print(bundle.describe())

    ids = bundle.tokenizer("Once upon a time", return_tensors="pt").input_ids
    logits = bundle(ids.to(bundle.device)).logits

    acc = bundle.access
    acc.parameter_groups()               # where the parameters live
    acc.bypass_block(7)                  # ablate one block
    acc.attach_after_final_norm(adapter) # small model in front of the LM head

Sub-modules
-----------
``lfm3.config``      architecture + runtime configs and presets
``lfm3.loader``      checkpoint loading, ``ModelBundle``
``lfm3.access``      structural view, hooks, surgery
``lfm3.modules``     n-gram layer, adapters, LoRA, probes
``lfm3.inspect``     human-readable architecture reports
``lfm3.experiments`` runnable demos (also the smoke tests)

The architecture is documented in ``docs/LFM3_1.7B_BASE.md`` and
``docs/LFM3_OMNI_2B.md``; the code walkthrough is in ``docs/LFM3_LAB_GUIDE.md``.
"""

from __future__ import annotations

from typing import Any

from . import config, modules
from .config import (
    CPU_LOAD,
    DEFAULT_LOAD,
    LFM3_1_7B_BASE,
    LFM3_OMNI_2B,
    LFM3_OMNI_TEXT,
    TINY_LOAD,
    AudioEncoderConfig,
    DepthConfig,
    Lfm3Config,
    LoadOptions,
    MiniLFM3,
    OmniConfig,
    TinyLFM3,
    TinyOmni,
    VisionConfig,
    canonical_layer_types,
)

__version__ = "0.1.0"

_LAZY = {
    "ModelBundle": "loader",
    "load_base": "loader",
    "load_omni": "loader",
    "load_tiny": "loader",
    "load_tiny_omni": "loader",
    "load_hf_config": "loader",
    "apply_compat_shims": "loader",
    "ModelAccess": "access",
    "Activations": "access",
    "ParamRow": "access",
    "describe": "inspect",
    "describe_architecture": "inspect",
    "parameter_budget": "inspect",
    "DEMOS": "experiments",
    "run_all": "experiments",
    "build_tiny_bundle": "experiments",
}


def __getattr__(name: str) -> Any:
    """Import heavy sub-modules only when they are used."""
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(
    [
        "config",
        "modules",
        "Lfm3Config",
        "OmniConfig",
        "VisionConfig",
        "AudioEncoderConfig",
        "DepthConfig",
        "LoadOptions",
        "LFM3_1_7B_BASE",
        "LFM3_OMNI_TEXT",
        "LFM3_OMNI_2B",
        "TinyLFM3",
        "TinyOmni",
        "MiniLFM3",
        "DEFAULT_LOAD",
        "CPU_LOAD",
        "TINY_LOAD",
        "canonical_layer_types",
        *_LAZY.keys(),
    ]
)