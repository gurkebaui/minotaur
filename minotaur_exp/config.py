"""Zentrale Konfiguration für alle Mini-Experimente und 1:1-Skalierung auf 3B / H100."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LoopConfig:
    """Konfiguration für Run 2 (Looped Transformer / Huginn / MoR) & Run 2 Exp (CTM-Light / YOCO-Conv)."""

    enabled: bool = False
    prelude_layers: int = 2
    core_layers: int = 4
    coda_layers: int = 2
    min_loops: int = 1
    max_loops: int = 4
    eval_loops: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tbptt_k: int = 2  # Loops vor den letzten k laufen unter torch.no_grad()
    use_input_injection: bool = True
    use_loop_gate: bool = True  # Zero-Init Per-Token-Gate
    use_step_embeddings: bool = True
    use_per_loop_rmsnorm: bool = True

    # Mixture-of-Recursions (MoR)
    use_mor: bool = False
    mor_capacity: float = 0.35
    mor_aux_loss_coef: float = 0.01
    use_selective_kv: bool = True

    # Modul E4 (Run 2 Exp): Prelude YOCO-KV + Core Conv(k=7) + YOCO Cross-Attention
    use_yoco_conv_core: bool = False

    # CTM-Light (Run 2 Exp: E1, E2, E3)
    use_ctm_sync: bool = False
    ctm_sync_pairs: int = 256
    use_ctm_certainty_loss: bool = False
    ctm_certainty_weight: float = 0.2
    use_ctm_tick_ema: bool = False
    ctm_ema_decay: float = 0.7


@dataclass
class SparseBackwardConfig:
    """Rho-1++ Token-Selective Sparse Backward (0-40% aus, 40-90% aktiv, 90-100% aus)."""

    enabled: bool = False
    start_frac: float = 0.40
    end_frac: float = 0.90
    cutoff_ratio: float = 0.55  # Untere 55% Tokens droppen -> Top 45% behalten
    mode: str = "self_ema"  # "self_ema" | "self_proxy" | "teacher_offline"


@dataclass
class ExperimentConfig:
    """Vollständige Feature-Flag-Konfiguration für ein einzelnes Experiment."""

    # Identität
    exp_id: str = "0b"
    group_id: str = "exp0"
    name: str = "Baseline B0"
    description: str = ""
    baseline_ref: str | None = None

    # Architektur (~150M muP-Proxy als Standard)
    vocab_size: int = 50304
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: int = 4
    mlp_dim: int = 3072
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    partial_rope_frac: float = 1.0  # 0.5 für LFM3
    rms_norm_eps: float = 1e-6
    use_mup: bool = True
    mup_base_width: int = 256

    # Systems-Unterbau
    use_flash_attn: bool = True
    use_flex_attn: bool = False
    use_fused_ops: bool = True
    use_compile: bool = True
    compile_mode: str = "default"
    use_cce: bool = True
    cce_chunk_size: int = 1024
    use_doc_masking: bool = True
    use_selective_ckpt: bool = True
    use_qk_norm: bool = True
    activation: str = "relu2"  # "relu2" | "swiglu"

    # Präzision & Optimizer
    precision_mode: str = "bf16_master"  # "bf16_master" | "pure_bf16_sr" | "ada_fp8"
    use_coat: bool = False
    optimizer_type: str = "adamw"  # "adamw" | "muon_8bit" | "apollo_mini"
    use_lomo_hook: bool = False
    use_wsd: bool = True
    wsd_warmup_frac: float = 0.05
    wsd_decay_frac: float = 0.10
    use_ema: bool = False
    ema_decay: float = 0.995
    lr: float = 3e-4
    muon_lr: float = 0.02
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Architektur- & FLOP-Killer (Exp 2)
    use_cola: bool = False
    cola_rank_ratio: float = 0.25
    use_lfm3_hybrid: bool = False
    conv_kernel_size: int = 7
    swa_window: int = 256
    use_head_gating: bool = False
    use_mla: bool = False
    mla_kv_lora_rank: int = 192
    use_mtp: bool = False
    mtp_k: int = 2
    mtp_weight: float = 0.3
    use_superbpe: bool = False
    superbpe_vocab_size: int = 65536

    # Memory & Self-Growth (Exp 3)
    use_ultramem: bool = False
    ultramem_keys_sqrt: int = 512
    ultramem_num_tables: int = 2
    ultramem_top_k: int = 16
    ultramem_val_dim: int = 512
    ultramem_layers: list[int] = field(default_factory=lambda: [1])
    use_self_growth: bool = False
    growth_stage1_width_div: int = 2
    growth_stage1_depth_div: int = 2
    growth_split_frac: float = 0.50

    # Daten & Distillation (Exp 4)
    use_jest: bool = False
    jest_superbatch_mult: int = 2
    use_offline_distill: bool = False
    distill_top_k: int = 16
    distill_alpha: float = 0.5
    distill_temp: float = 2.0
    teacher_model_name: str = "gpt2"
    reasoning_mix_ratio: float = 0.0

    # Verschachtelte Configs
    loop: LoopConfig = field(default_factory=LoopConfig)
    sparse_bw: SparseBackwardConfig = field(default_factory=SparseBackwardConfig)

    # Trainings-Budget & Ablauf
    batch_size: int = 8
    grad_accum_steps: int = 1
    seq_len: int = 1024
    max_minutes: float = 45.0
    max_steps: int = 200_000
    eval_every_steps: int = 200
    log_every_steps: int = 20
    seed: int = 42
    init_from_exp: str | None = None
    upcycle_to_loop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        d = dict(data)
        if "loop" in d and isinstance(d["loop"], dict):
            d["loop"] = LoopConfig(**d["loop"])
        if "sparse_bw" in d and isinstance(d["sparse_bw"], dict):
            d["sparse_bw"] = SparseBackwardConfig(**d["sparse_bw"])
        return cls(**d)

    def apply_scale_preset(self, preset: str) -> None:
        """Skaliert Architektur & UltraMemV2 1:1 zwischen Smoke-Test, 150M-Proxy und 3B-Target."""
        if preset == "proxy_150m":
            return
        if preset == "tiny_smoke":
            self.d_model = 128
            self.n_layers = 8
            self.n_heads = 4
            self.n_kv_heads = 2
            self.mlp_dim = 384
            self.max_seq_len = 128
            self.seq_len = 128
            self.batch_size = 4
            self.grad_accum_steps = 1
            self.cce_chunk_size = 128
            self.swa_window = 64
            self.mla_kv_lora_rank = 64
            self.ultramem_keys_sqrt = 32  # 1024 Keys/Table -> schneller Smoke-Test
            self.ultramem_top_k = 4
            self.ultramem_val_dim = 128
            self.ultramem_layers = [1]
            self.superbpe_vocab_size = 2048
            if self.loop.enabled:
                self.loop.prelude_layers = 2
                self.loop.core_layers = 4
                self.loop.coda_layers = 2
                self.loop.ctm_sync_pairs = 64
            return
        if preset == "target_3b":
            self.d_model = 1536
            self.n_layers = 24
            self.n_heads = 16
            self.n_kv_heads = 4
            self.mlp_dim = 6144
            self.max_seq_len = 2048
            self.seq_len = 2048
            self.swa_window = 512
            self.mla_kv_lora_rank = 384
            self.ultramem_keys_sqrt = 1024  # ~1M Keys/Table
            self.ultramem_num_tables = 2
            self.ultramem_val_dim = 1280
            if self.loop.enabled:
                self.loop.prelude_layers = 4
                self.loop.core_layers = 12
                self.loop.coda_layers = 4
            return
        raise ValueError(f"Unbekanntes model_scale Preset: {preset!r}")
