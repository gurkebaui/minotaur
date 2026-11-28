"""Laden der übergeordneten Suite-YAML-Config und Registry aller 15 Sub-Experimente."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from minotaur_exp.config import ExperimentConfig, LoopConfig, SparseBackwardConfig

DEFAULT_SUITE_CONFIG: dict[str, Any] = {
    "selected_experiments": ["all"],
    "model_scale": "proxy_150m",
    "time_scale": 1.0,
    "hardware": {
        "device": "cuda",
        "compile_mode": "default",
        "allow_tf32": True,
    },
    "data": {
        "dataset_preset": "fineweb_edu",
        "cache_dir": "minotaur_exp/cache",
        "train_tokens": 50_000_000,
        "val_tokens": 2_000_000,
        "seq_len": 1024,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "teacher_model": "gpt2",
        "superbpe_vocab_size": 65536,
    },
    "logging": {
        "logs_root": "minotaur_exp/logs",
        "run_tag": "",
        "log_every_steps": 20,
        "eval_every_steps": 200,
        "save_checkpoints": True,
        "continue_on_error": True,
    },
    "checkpoint_overrides": {},
    "experiment_overrides": {},
}

GROUP_ALIASES: dict[str, list[str]] = {
    "0": ["0a", "0b", "0_sweep"],
    "exp0": ["0a", "0b", "0_sweep"],
    "1": ["1a", "1b"],
    "exp1": ["1a", "1b"],
    "2": ["2a", "2b"],
    "exp2": ["2a", "2b"],
    "3": ["3a", "3b"],
    "exp3": ["3a", "3b"],
    "4": ["4a", "4b"],
    "exp4": ["4a", "4b"],
    "5": ["5a", "5b"],
    "exp5": ["5a", "5b"],
    "6": ["6a", "6b"],
    "exp6": ["6a", "6b"],
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_suite_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lädt die übergeordnete YAML-Config und füllt Defaults sauber auf."""
    if path is None:
        path = Path(__file__).resolve().parent / "suite_config.yaml"
    else:
        path = Path(path)
    if not path.exists():
        return dict(DEFAULT_SUITE_CONFIG)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_SUITE_CONFIG, loaded)


def _apply_dict_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> None:
    for key, val in overrides.items():
        if key == "loop" and isinstance(val, dict):
            for lk, lv in val.items():
                setattr(cfg.loop, lk, lv)
        elif key == "sparse_bw" and isinstance(val, dict):
            for sk, sv in val.items():
                setattr(cfg.sparse_bw, sk, sv)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)


def build_experiment_registry(suite_cfg: dict[str, Any]) -> OrderedDict[str, ExperimentConfig]:
    """Erzeugt alle 15 Experiment-Konfigurationen gemäß Experiments.md."""
    data_cfg = suite_cfg.get("data", {})
    hw_cfg = suite_cfg.get("hardware", {})
    log_cfg = suite_cfg.get("logging", {})
    scale = suite_cfg.get("model_scale", "proxy_150m")
    time_scale = float(suite_cfg.get("time_scale", 1.0))

    def make_base(**kwargs: Any) -> ExperimentConfig:
        cfg = ExperimentConfig(
            seq_len=int(data_cfg.get("seq_len", 1024)),
            max_seq_len=max(1024, int(data_cfg.get("seq_len", 1024))),
            batch_size=int(data_cfg.get("batch_size", 8)),
            grad_accum_steps=int(data_cfg.get("grad_accum_steps", 1)),
            compile_mode=str(hw_cfg.get("compile_mode", "default")),
            log_every_steps=int(log_cfg.get("log_every_steps", 20)),
            eval_every_steps=int(log_cfg.get("eval_every_steps", 200)),
            teacher_model_name=str(data_cfg.get("teacher_model", "gpt2")),
            superbpe_vocab_size=int(data_cfg.get("superbpe_vocab_size", 65536)),
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        cfg.apply_scale_preset(scale)
        cfg.max_minutes = max(0.05, cfg.max_minutes * time_scale)
        return cfg

    reg: OrderedDict[str, ExperimentConfig] = OrderedDict()

    # --- Exp 0: Baseline & Systems-Bundle ---
    reg["0a"] = make_base(
        exp_id="0a",
        group_id="exp0",
        name="Exp 0a: Standard Dense Baseline",
        description="PyTorch SDPA, Standard Cross-Entropy, BF16+FP32 Master, AdamW (ohne Systems-Stack).",
        use_flash_attn=False,
        use_fused_ops=False,
        use_compile=False,
        use_cce=False,
        use_doc_masking=False,
        use_selective_ckpt=False,
        use_qk_norm=False,
        activation="swiglu",
        max_minutes=25.0,
    )
    reg["0b"] = make_base(
        exp_id="0b",
        group_id="exp0",
        name="Exp 0b: Systems-Bundle B0",
        description="FA-2 + Fused Ops + torch.compile + CCE + Sequence Packing + Selective Ckpt + QK-Norm + ReLU2.",
        baseline_ref="0a",
        max_minutes=45.0,
    )
    reg["0_sweep"] = make_base(
        exp_id="0_sweep",
        group_id="exp0",
        name="Exp 0 Sweep: Delta-Ablation",
        description="Kurzläufe ohne Fused-Ops, CCE oder torch.compile zur exakten Delta-Isolierung gegen 0b.",
        baseline_ref="0b",
        max_minutes=20.0,
    )

    # --- Exp 1: Ultra-Low-VRAM & Präzisions-Bundle ---
    reg["1a"] = make_base(
        exp_id="1a",
        group_id="exp1",
        name="Exp 1A: Pure-BF16+SR + 8-bit Muon + LOMO + WSD+EMA",
        description="Keine FP32 Master Weights, 8-bit Muon (2D) + 8-bit AdamW, LOMO Fused Hook, WSD + EMA.",
        baseline_ref="0b",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_wsd=True,
        use_ema=True,
        max_minutes=72.0,
    )
    reg["1b"] = make_base(
        exp_id="1b",
        group_id="exp1",
        name="Exp 1B: Ada-FP8 + COAT + APOLLO-Mini + LOMO + Head-Gating",
        description="Maximale Kompression für 7B-Skalierung: FP8 Scaled-MM, COAT Saved-Tensors, APOLLO-Mini, Head-Wise Gating.",
        baseline_ref="1a",
        precision_mode="ada_fp8",
        use_coat=True,
        optimizer_type="apollo_mini",
        use_lomo_hook=True,
        use_head_gating=True,
        use_wsd=True,
        max_minutes=78.0,
    )

    # --- Exp 2: Architektur- & FLOP-Killer-Bundle ---
    reg["2a"] = make_base(
        exp_id="2a",
        group_id="exp2",
        name="Exp 2A: CoLA + LFM3 Tri-Hybrid",
        description="1A + CoLA Low-Rank Bottleneck + LFM3 Tri-Hybrid (50% Conv k=7, 25% SWA, 25% Global MLA + Head-Gating + Partial RoPE).",
        baseline_ref="1a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_lfm3_hybrid=True,
        use_head_gating=True,
        use_mla=True,
        partial_rope_frac=0.5,
        max_minutes=90.0,
    )
    reg["2b"] = make_base(
        exp_id="2b",
        group_id="exp2",
        name="Exp 2B: 2A + SuperBPE + MTP (k=2)",
        description="2A + SuperBPE Cross-Whitespace Tokenizer + Multi-Token Prediction (k=2).",
        baseline_ref="2a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_lfm3_hybrid=True,
        use_head_gating=True,
        use_mla=True,
        partial_rope_frac=0.5,
        use_superbpe=True,
        use_mtp=True,
        mtp_k=2,
        max_minutes=90.0,
    )

    # --- Exp 3: Parameter-Entkopplung & Self-Growth ---
    reg["3a"] = make_base(
        exp_id="3a",
        group_id="exp3",
        name="Exp 3A: CoLA-Backbone + UltraMemV2 (1:4)",
        description="2A CoLA-Backbone + UltraMemV2 PKM-Lookup-Tabellen im 1:4 Parameter-Verhältnis.",
        baseline_ref="2a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_lfm3_hybrid=True,
        use_head_gating=True,
        use_mla=True,
        partial_rope_frac=0.5,
        use_ultramem=True,
        max_minutes=90.0,
    )
    reg["3b"] = make_base(
        exp_id="3b",
        group_id="exp3",
        name="Exp 3B: Self-Growth (HyperCloning + GStack + Zero-Init UltraMemV2)",
        description="Start bei halber Breite/Tiefe -> funktions-erhaltendes Wachstum + Zero-Init UltraMemV2 Sparse Upcycling.",
        baseline_ref="3a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_lfm3_hybrid=True,
        use_head_gating=True,
        use_mla=True,
        partial_rope_frac=0.5,
        use_ultramem=True,
        use_self_growth=True,
        max_minutes=90.0,
    )

    # --- Exp 4: Daten- & Lossy-Backward-Bundle ---
    reg["4a"] = make_base(
        exp_id="4a",
        group_id="exp4",
        name="Exp 4A: Model B (Self-JEST + Rho-1++ 40-90% Sparse Backward)",
        description="100% From Scratch: Self-Bootstrapped JEST + Sparse Backward (0-40% aus, 40-90% 55% Cutoff, 90-100% aus).",
        baseline_ref="2a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_lfm3_hybrid=True,
        use_head_gating=True,
        use_mla=True,
        partial_rope_frac=0.5,
        use_jest=True,
        sparse_bw=SparseBackwardConfig(enabled=True, start_frac=0.40, end_frac=0.90, cutoff_ratio=0.55, mode="self_ema"),
        max_minutes=90.0,
    )
    reg["4b"] = make_base(
        exp_id="4b",
        group_id="exp4",
        name="Exp 4B: Model A (Offline Top-16 Distill + Teacher Rho-1++)",
        description="Offline Top-16 Logit-Distillation + Teacher-gescortes Rho-1++ Sparse Backward (40-90%).",
        baseline_ref="4a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_lfm3_hybrid=True,
        use_head_gating=True,
        use_mla=True,
        partial_rope_frac=0.5,
        use_offline_distill=True,
        sparse_bw=SparseBackwardConfig(enabled=True, start_frac=0.40, end_frac=0.90, cutoff_ratio=0.55, mode="teacher_offline"),
        max_minutes=90.0,
    )

    # --- Exp 5: Run 2 Core (Looped Transformer & MoR) ---
    reg["5a"] = make_base(
        exp_id="5a",
        group_id="exp5",
        name="Exp 5A: Stabiler Huginn Loop-Kern (TBPTT k=2)",
        description="Huginn Sandwich + Input-Injection + Zero-Init Loop-Gate + Step-Norms + R in [1,4] + TBPTT k=2 + UltraMemV2 1x.",
        baseline_ref="2a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_ultramem=True,
        reasoning_mix_ratio=0.15,
        init_from_exp="2a",
        upcycle_to_loop=True,
        loop=LoopConfig(
            enabled=True,
            prelude_layers=2,
            core_layers=4,
            coda_layers=2,
            min_loops=1,
            max_loops=4,
            tbptt_k=2,
            use_mor=False,
        ),
        max_minutes=90.0,
    )
    reg["5b"] = make_base(
        exp_id="5b",
        group_id="exp5",
        name="Exp 5B: Max Loop-Sparsity (MoR 35% + TBPTT k=1)",
        description="5A + Mixture-of-Recursions (35% aktive Tokens in Loop 2-4) + selektives KV + TBPTT k=1 + LOMO-Hook.",
        baseline_ref="5a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_ultramem=True,
        reasoning_mix_ratio=0.15,
        init_from_exp="2a",
        upcycle_to_loop=True,
        loop=LoopConfig(
            enabled=True,
            prelude_layers=2,
            core_layers=4,
            coda_layers=2,
            min_loops=1,
            max_loops=4,
            tbptt_k=1,
            use_mor=True,
            mor_capacity=0.35,
            use_selective_kv=True,
        ),
        max_minutes=90.0,
    )

    # --- Exp 6: Run 2 Exp (LFM3-Conv-YOCO-Loop & CTM-Light) ---
    reg["6a"] = make_base(
        exp_id="6a",
        group_id="exp6",
        name="Exp 6A: Modul E4 (LFM3 Conv k=7 + YOCO-Loop + Head-Gating)",
        description="Prelude erzeugt 1x globalen YOCO-KV-Cache; Loop-Core nutzt Full-Seq Conv(k=7) + aktive MoR YOCO Cross-Attn.",
        baseline_ref="5a",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_head_gating=True,
        use_ultramem=True,
        reasoning_mix_ratio=0.15,
        init_from_exp="5a",
        upcycle_to_loop=False,
        loop=LoopConfig(
            enabled=True,
            prelude_layers=2,
            core_layers=4,
            coda_layers=2,
            min_loops=1,
            max_loops=4,
            tbptt_k=2,
            use_mor=True,
            mor_capacity=0.40,
            use_yoco_conv_core=True,
        ),
        max_minutes=75.0,
    )
    reg["6b"] = make_base(
        exp_id="6b",
        group_id="exp6",
        name="Exp 6B: CTM-Light (E1 Subsampled Sync + E2 Certainty-Loss + E3 Tick-EMA)",
        description="Testet Sakana CTM-Light (P=256 Sync, Certainty-Halting Loss, Tick-History-EMA) gegen Exp 5.",
        baseline_ref="5b",
        precision_mode="pure_bf16_sr",
        optimizer_type="muon_8bit",
        use_lomo_hook=True,
        use_ema=True,
        use_cola=True,
        use_ultramem=True,
        reasoning_mix_ratio=0.15,
        init_from_exp="5b",
        upcycle_to_loop=False,
        loop=LoopConfig(
            enabled=True,
            prelude_layers=2,
            core_layers=4,
            coda_layers=2,
            min_loops=1,
            max_loops=4,
            tbptt_k=2,
            use_mor=True,
            mor_capacity=0.35,
            use_ctm_sync=True,
            ctm_sync_pairs=256,
            use_ctm_certainty_loss=True,
            ctm_certainty_weight=0.2,
            use_ctm_tick_ema=True,
            ctm_ema_decay=0.7,
        ),
        max_minutes=75.0,
    )

    # Per-Experiment Overrides aus YAML anwenden
    exp_overrides = suite_cfg.get("experiment_overrides", {}) or {}
    for exp_id, ov in exp_overrides.items():
        if exp_id in reg and isinstance(ov, dict):
            _apply_dict_overrides(reg[exp_id], ov)

    # Re-apply scale preset falls override nichts anderes sagt (für tiny_smoke wichtig)
    if scale == "tiny_smoke":
        for cfg in reg.values():
            cfg.apply_scale_preset("tiny_smoke")

    return reg


def resolve_selection(
    tokens: list[str] | None,
    registry: OrderedDict[str, ExperimentConfig],
    exclude: list[str] | None = None,
) -> list[str]:
    """Löst Gruppen-Aliase ('0', 'exp5', 'all') und Einzel-IDs ('1a', '6b') in geordnete ID-Liste auf."""
    all_ids = list(registry.keys())
    if not tokens:
        selected = list(all_ids)
    else:
        chosen_set: set[str] = set()
        for raw in tokens:
            for part in str(raw).replace(",", " ").split():
                tok = part.strip().lower()
                if not tok:
                    continue
                if tok == "all":
                    chosen_set.update(all_ids)
                elif tok in GROUP_ALIASES:
                    chosen_set.update(GROUP_ALIASES[tok])
                elif tok in registry:
                    chosen_set.add(tok)
                else:
                    raise ValueError(
                        f"Unbekannte Experiment-ID oder Gruppe: {part!r}. "
                        f"Gültig: {', '.join(all_ids)} oder exp0..exp6 / all"
                    )
        selected = [eid for eid in all_ids if eid in chosen_set]

    if exclude:
        ex_set: set[str] = set()
        for raw in exclude:
            for part in str(raw).replace(",", " ").split():
                tok = part.strip().lower()
                if tok in GROUP_ALIASES:
                    ex_set.update(GROUP_ALIASES[tok])
                elif tok in registry:
                    ex_set.add(tok)
        selected = [eid for eid in selected if eid not in ex_set]

    return selected
