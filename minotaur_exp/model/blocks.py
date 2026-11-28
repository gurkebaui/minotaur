"""Zentrale Bausteine: CoLA, ReLU²/SwiGLU, Sandwich Prelude/Core/Coda, TBPTT, MTP & MinotaurModel."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from minotaur_exp.config import ExperimentConfig
from minotaur_exp.model.attention import CausalAttention, RMSNorm
from minotaur_exp.model.conv_mixer import DoubleGatedShortConv
from minotaur_exp.model.memory import UltraMemV2Layer
from minotaur_exp.model.router import CTMLightModule, MoRRouter


def _round_up_16(x: int) -> int:
    return max(16, ((x + 15) // 16) * 16)


class CoLALinear(nn.Module):
    """CoLA: Low-Rank Activation Bottleneck Linear (h = B(act(A(x))))."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = _round_up_16(rank)
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class MLPBlock(nn.Module):
    """FFN mit ReLU² oder SwiGLU und optionalem CoLA-Low-Rank-Bottleneck."""

    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        activation: str = "relu2",
        use_cola: bool = False,
        cola_rank_ratio: float = 0.25,
    ):
        super().__init__()
        self.activation = activation
        self.use_cola = use_cola

        if use_cola:
            rank = _round_up_16(int(d_model * cola_rank_ratio))
            self.w_in = CoLALinear(d_model, mlp_dim // 2, rank=rank)
            self.w_out = CoLALinear(mlp_dim // 2, d_model, rank=rank)
            self.w_gate = None
        elif activation == "swiglu":
            self.w_in = nn.Linear(d_model, mlp_dim, bias=False)
            self.w_gate = nn.Linear(d_model, mlp_dim, bias=False)
            self.w_out = nn.Linear(mlp_dim, d_model, bias=False)
        else:  # relu2
            self.w_in = nn.Linear(d_model, mlp_dim, bias=False)
            self.w_gate = None
            self.w_out = nn.Linear(mlp_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cola:
            h = self.w_in(x)
            h = F.relu(h).pow(2) if self.activation == "relu2" else F.silu(h)
            return self.w_out(h)
        if self.activation == "swiglu":
            return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))
        return self.w_out(F.relu(self.w_in(x)).pow(2))


class TransformerBlock(nn.Module):
    """Einheitlicher Block für Global-Attn, SWA, LFM3-Conv(k=7) oder YOCO-Cross-Attn (+ UltraMemV2)."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        mixer_type: str = "global_attn",
        has_ultramem: bool = False,
        zero_init_ultramem: bool = False,
        num_norm_variants: int = 1,
    ):
        super().__init__()
        self.mixer_type = mixer_type
        self.use_selective_ckpt = cfg.use_selective_ckpt
        self.num_norm_variants = max(1, num_norm_variants)

        self.ln1 = nn.ModuleList(
            [RMSNorm(cfg.d_model, cfg.rms_norm_eps, cfg.use_fused_ops) for _ in range(self.num_norm_variants)]
        )
        self.ln2 = nn.ModuleList(
            [RMSNorm(cfg.d_model, cfg.rms_norm_eps, cfg.use_fused_ops) for _ in range(self.num_norm_variants)]
        )

        lin_factory: Callable[[int, int], nn.Module] | None = None
        if cfg.use_cola:
            rank = _round_up_16(int(cfg.d_model * cfg.cola_rank_ratio))
            lin_factory = lambda i, o, r=rank: CoLALinear(i, o, rank=r)

        if mixer_type == "conv":
            self.mixer = DoubleGatedShortConv(cfg.d_model, cfg.conv_kernel_size, linear_factory=lin_factory)
        else:
            win = cfg.swa_window if mixer_type == "swa" else None
            is_yoco = mixer_type == "yoco_cross_attn"
            self.mixer = CausalAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                max_seq_len=cfg.max_seq_len,
                rope_base=cfg.rope_base,
                partial_rope_frac=cfg.partial_rope_frac,
                use_qk_norm=cfg.use_qk_norm,
                use_fused_ops=cfg.use_fused_ops,
                use_flash_attn=cfg.use_flash_attn,
                use_flex_attn=cfg.use_flex_attn,
                use_head_gating=cfg.use_head_gating,
                use_mla=cfg.use_mla,
                mla_kv_lora_rank=cfg.mla_kv_lora_rank,
                window_size=win,
                is_yoco_cross=is_yoco,
                linear_factory=lin_factory,
            )

        self.mlp = MLPBlock(
            d_model=cfg.d_model,
            mlp_dim=cfg.mlp_dim,
            activation=cfg.activation,
            use_cola=cfg.use_cola,
            cola_rank_ratio=cfg.cola_rank_ratio,
        )

        self.ultramem: UltraMemV2Layer | None = None
        if has_ultramem:
            self.ultramem = UltraMemV2Layer(
                d_model=cfg.d_model,
                keys_sqrt=cfg.ultramem_keys_sqrt,
                num_tables=cfg.ultramem_num_tables,
                top_k=cfg.ultramem_top_k,
                val_dim=cfg.ultramem_val_dim,
                zero_init_out=zero_init_ultramem,
            )

    def _run_mlp(self, h_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_norm)

    def forward(
        self,
        x: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        loop_idx: int = 0,
        external_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        q_positions: torch.Tensor | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        n_idx = min(loop_idx, self.num_norm_variants - 1)
        h_in = self.ln1[n_idx](x)

        kv_out = None
        if self.mixer_type == "conv":
            mix_out = self.mixer(h_in, doc_ids=doc_ids)
        else:
            mix_out, kv_out = self.mixer(
                h_in,
                doc_ids=doc_ids,
                external_kv=external_kv,
                q_positions=q_positions,
                return_kv=return_kv,
            )
        x = x + mix_out

        h_mlp_in = self.ln2[n_idx](x)
        if self.use_selective_ckpt and self.training and torch.is_grad_enabled():
            mlp_out = ckpt_fn(self._run_mlp, h_mlp_in, use_reentrant=False)
        else:
            mlp_out = self._run_mlp(h_mlp_in)

        mem_aux = None
        if self.ultramem is not None:
            mem_out, mem_aux = self.ultramem(h_mlp_in)
            x = x + mlp_out + mem_out
        else:
            x = x + mlp_out

        return x, kv_out, mem_aux


@dataclass
class ModelOutput:
    hidden_states: torch.Tensor
    mtp_hidden_states: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    ctm_tick_hiddens: list[torch.Tensor] | None = None
    mor_active_frac: float = 1.0
    loops_executed: int = 1


def _choose_mixer_type(layer_idx: int, cfg: ExperimentConfig) -> str:
    if not cfg.use_lfm3_hybrid:
        return "global_attn"
    # LFM3 Tri-Hybrid: 50% Conv(k=7), 25% SWA, 25% Global
    mod = layer_idx % 4
    if mod in (0, 2):
        return "conv"
    if mod == 1:
        return "swa"
    return "global_attn"


class MinotaurModel(nn.Module):
    """Einheitliches Modell für Exp 0-4 (Flat) und Exp 5-6 (Huginn Sandwich Looped + MoR + CTM-Light)."""

    def __init__(self, cfg: ExperimentConfig, zero_init_ultramem: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_looped = cfg.loop.enabled

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        mem_layers = set(cfg.ultramem_layers) if cfg.use_ultramem else set()

        if not self.is_looped:
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(
                        cfg,
                        mixer_type=_choose_mixer_type(i, cfg),
                        has_ultramem=(i in mem_layers),
                        zero_init_ultramem=zero_init_ultramem,
                        num_norm_variants=1,
                    )
                    for i in range(cfg.n_layers)
                ]
            )
        else:
            lc = cfg.loop
            # Prelude (mit 1x UltraMemV2 außerhalb des Loops und YOCO-Global-Attn am Ende falls E4 aktiv)
            prelude_list = []
            for i in range(lc.prelude_layers):
                if lc.use_yoco_conv_core:
                    m_type = "global_attn" if i == lc.prelude_layers - 1 else ("conv" if i % 2 == 0 else "swa")
                else:
                    m_type = _choose_mixer_type(i, cfg)
                prelude_list.append(
                    TransformerBlock(
                        cfg,
                        mixer_type=m_type,
                        has_ultramem=(i == 0 and cfg.use_ultramem),
                        zero_init_ultramem=zero_init_ultramem,
                        num_norm_variants=1,
                    )
                )
            self.prelude = nn.ModuleList(prelude_list)

            # Looped Core
            norm_vars = lc.max_loops if lc.use_per_loop_rmsnorm else 1
            core_list = []
            for i in range(lc.core_layers):
                if lc.use_yoco_conv_core:
                    # Modul E4: abwechselnd Full-Seq Conv(k=7) und Active-Token YOCO Cross-Attention
                    m_type = "conv" if i % 2 == 0 else "yoco_cross_attn"
                else:
                    m_type = _choose_mixer_type(i, cfg)
                core_list.append(
                    TransformerBlock(
                        cfg,
                        mixer_type=m_type,
                        has_ultramem=False,
                        num_norm_variants=norm_vars,
                    )
                )
            self.core = nn.ModuleList(core_list)

            # Coda
            self.coda = nn.ModuleList(
                [
                    TransformerBlock(
                        cfg,
                        mixer_type="global_attn" if i == lc.coda_layers - 1 else _choose_mixer_type(i, cfg),
                        has_ultramem=False,
                        num_norm_variants=1,
                    )
                    for i in range(lc.coda_layers)
                ]
            )

            # Loop-Conditioning & Stabilisierung
            if lc.use_step_embeddings:
                self.step_emb = nn.Embedding(max(8, lc.max_loops + 2), cfg.d_model)
            if lc.use_loop_gate:
                self.loop_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
                nn.init.zeros_(self.loop_gate.weight)
                nn.init.zeros_(self.loop_gate.bias)
            if lc.use_input_injection:
                self.input_inj_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

            self.mor_router = MoRRouter(cfg.d_model, capacity=lc.mor_capacity) if lc.use_mor else None
            if lc.use_ctm_sync or lc.use_ctm_certainty_loss or lc.use_ctm_tick_ema:
                self.ctm = CTMLightModule(
                    cfg.d_model,
                    use_sync=lc.use_ctm_sync,
                    sync_pairs=lc.ctm_sync_pairs,
                    use_certainty_loss=lc.use_ctm_certainty_loss,
                    use_tick_ema=lc.use_ctm_tick_ema,
                    ema_decay=lc.ctm_ema_decay,
                )
            else:
                self.ctm = None

        self.ln_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps, cfg.use_fused_ops)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight Tying

        # Optional: Multi-Token Prediction (k=2)
        if cfg.use_mtp and cfg.mtp_k >= 2:
            self.mtp_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
            self.mtp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, cfg.use_fused_ops)
        else:
            self.mtp_proj = None

        self.init_weights_mup()

    def init_weights_mup(self) -> None:
        """muP-kompatible Initialisierung für 1:1-Transfer auf 3B (inkl. Residual- & Conv1D-Skalierung)."""
        width_mult = (self.cfg.mup_base_width / self.cfg.d_model) if self.cfg.use_mup else 1.0
        std_hidden = 0.02 * math.sqrt(width_mult)
        total_layers = (
            (self.cfg.loop.prelude_layers + self.cfg.loop.core_layers + self.cfg.loop.coda_layers)
            if self.is_looped
            else self.cfg.n_layers
        )
        res_scale = 1.0 / math.sqrt(2.0 * max(1, total_layers))

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            elif isinstance(mod, nn.Conv1d):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Linear):
                if mod is self.lm_head:
                    continue
                if "loop_gate" in name or "gate" in name or "sync_proj" in name:
                    continue
                if "ultramem" in name and "out_proj" in name:
                    continue
                std = std_hidden * res_scale if ("out_proj" in name or "w_out" in name) else std_hidden
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None and "head_gate" not in name:
                    nn.init.zeros_(mod.bias)

    def _run_core_step(
        self,
        h: torch.Tensor,
        x_in: torch.Tensor,
        loop_idx: int,
        doc_ids: torch.Tensor | None,
        yoco_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        lc = self.cfg.loop
        h_prev = h
        if loop_idx > 0 and lc.use_input_injection:
            h = 0.5 * (h + self.input_inj_proj(x_in))
        if lc.use_step_embeddings:
            s_idx = min(loop_idx, self.step_emb.num_embeddings - 1)
            h = h + self.step_emb.weight[s_idx].to(h.dtype).view(1, 1, -1)

        aux_list: list[torch.Tensor] = []
        active_frac = 1.0

        # MoR-Routing ab Loop 1 (Loop 0 verarbeitet alle Tokens für saubere KV/Basis)
        use_mor_here = self.mor_router is not None and loop_idx > 0
        if use_mor_here:
            active_idx, active_w, mor_aux, active_frac = self.mor_router(h)
            aux_list.append(mor_aux * lc.mor_aux_loss_coef)
        else:
            active_idx, active_w = None, None

        for blk in self.core:
            if use_mor_here and blk.mixer_type == "conv":
                # Modul E4: Conv(k=7) läuft auf voller Sequenz -> keine Nachbarschaftslücken!
                h, _, m_aux = blk(h, doc_ids=doc_ids, loop_idx=loop_idx)
            elif use_mor_here and blk.mixer_type == "yoco_cross_attn":
                # Modul E4: YOCO Cross-Attention nur auf aktiven MoR-Tokens gegen statischen Prelude-KV
                h_act = self.mor_router.gather(h, active_idx)
                h_act_new, _, m_aux = blk(
                    h_act,
                    doc_ids=doc_ids,
                    loop_idx=loop_idx,
                    external_kv=yoco_kv,
                    q_positions=active_idx,
                )
                h = self.mor_router.scatter_update(h, h_act_new, active_idx, active_w)
            elif use_mor_here:
                # Standard MoR: selektive Token-Subsequenz
                h_act = self.mor_router.gather(h, active_idx)
                doc_act = torch.gather(doc_ids, 1, active_idx) if doc_ids is not None else None
                h_act_new, _, m_aux = blk(h_act, doc_ids=doc_act, loop_idx=loop_idx)
                h = self.mor_router.scatter_update(h, h_act_new, active_idx, active_w)
            else:
                h, _, m_aux = blk(h, doc_ids=doc_ids, loop_idx=loop_idx, external_kv=yoco_kv)

            if m_aux is not None:
                aux_list.append(m_aux * 0.01)

        if lc.use_loop_gate and loop_idx > 0:
            gate = torch.sigmoid(self.loop_gate(h_prev))
            h = h_prev + gate * (h - h_prev)

        aux = torch.stack(aux_list).sum() if aux_list else None
        return h, aux, active_frac

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        num_loops: int | None = None,
        collect_tick_hiddens: bool = False,
    ) -> ModelOutput:
        h = self.tok_emb(input_ids)
        aux_losses: list[torch.Tensor] = []

        if not self.is_looped:
            for blk in self.layers:
                h, _, m_aux = blk(h, doc_ids=doc_ids)
                if m_aux is not None:
                    aux_losses.append(m_aux * 0.01)
            h_final = self.ln_f(h)
            mtp_h = self._compute_mtp_hidden(h_final, input_ids)
            aux = torch.stack(aux_losses).sum() if aux_losses else None
            return ModelOutput(hidden_states=h_final, mtp_hidden_states=mtp_h, aux_loss=aux)

        # --- Huginn Sandwich Looped Forward ---
        lc = self.cfg.loop
        yoco_kv = None
        for i, blk in enumerate(self.prelude):
            want_kv = lc.use_yoco_conv_core and (i == len(self.prelude) - 1)
            h, kv_out, m_aux = blk(h, doc_ids=doc_ids, return_kv=want_kv)
            if want_kv and kv_out is not None:
                yoco_kv = kv_out
            if m_aux is not None:
                aux_losses.append(m_aux * 0.01)

        if num_loops is not None:
            R = num_loops
        elif self.training:
            R = random.randint(lc.min_loops, lc.max_loops)
        else:
            R = lc.max_loops

        k_tbptt = min(R, max(1, lc.tbptt_k)) if self.training else R
        no_grad_loops = max(0, R - k_tbptt)

        x_in = h.clone()
        ctm_state = self.ctm.init_state(h) if self.ctm is not None else None
        tick_hiddens: list[torch.Tensor] = []
        active_fracs: list[float] = []

        for loop_idx in range(R):
            in_no_grad = self.training and (loop_idx < no_grad_loops)
            if in_no_grad:
                with torch.no_grad():
                    h, step_aux, a_frac = self._run_core_step(h, x_in, loop_idx, doc_ids, yoco_kv)
                    if self.ctm is not None and ctm_state is not None:
                        h = self.ctm.step_tick(h, ctm_state)
                h = h.detach()
                if ctm_state is not None and ctm_state["sync"] is not None:
                    ctm_state["sync"] = ctm_state["sync"].detach()
                if ctm_state is not None and ctm_state["ema"] is not None:
                    ctm_state["ema"] = ctm_state["ema"].detach()
            else:
                h, step_aux, a_frac = self._run_core_step(h, x_in, loop_idx, doc_ids, yoco_kv)
                if self.ctm is not None and ctm_state is not None:
                    h = self.ctm.step_tick(h, ctm_state)
                if step_aux is not None:
                    aux_losses.append(step_aux)
                if collect_tick_hiddens or (self.training and lc.use_ctm_certainty_loss):
                    tick_hiddens.append(self.ln_f(h))
            active_fracs.append(a_frac)

        if self.ctm is not None and ctm_state is not None:
            h = self.ctm.apply_readout(h, ctm_state)

        for blk in self.coda:
            h, _, m_aux = blk(h, doc_ids=doc_ids)
            if m_aux is not None:
                aux_losses.append(m_aux * 0.01)

        h_final = self.ln_f(h)
        mtp_h = self._compute_mtp_hidden(h_final, input_ids)
        aux = torch.stack(aux_losses).sum() if aux_losses else None
        return ModelOutput(
            hidden_states=h_final,
            mtp_hidden_states=mtp_h,
            aux_loss=aux,
            ctm_tick_hiddens=tick_hiddens if tick_hiddens else None,
            mor_active_frac=sum(active_fracs) / max(1, len(active_fracs)),
            loops_executed=R,
        )

    def _compute_mtp_hidden(self, h_final: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.mtp_proj is None or input_ids.shape[1] < 2:
            return None
        # Sagt Token t+2 aus [h_t, tok_emb(x_{t+1})] voraus
        next_emb = self.tok_emb(input_ids[:, 1:])
        cat = torch.cat([h_final[:, :-1], next_emb], dim=-1)
        return self.mtp_norm(self.mtp_proj(cat))
