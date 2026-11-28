"""Self-Growth (HyperCloning + GStack + Zero-Init UltraMemV2) & Loop-Upcycling."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from minotaur_exp.config import ExperimentConfig
from minotaur_exp.model.blocks import CoLALinear, MinotaurModel


def _tile_weight_2d(src: torch.Tensor, dst_shape: tuple[int, int]) -> torch.Tensor:
    """Funktions-erhaltendes HyperCloning-Tiling für 2D-Matrizen (out, in)."""
    dst_out, dst_in = dst_shape
    src_out, src_in = src.shape
    rep_out = math.ceil(dst_out / src_out)
    rep_in = math.ceil(dst_in / src_in)
    tiled = src.repeat(rep_out, rep_in)[:dst_out, :dst_in]
    # In-Dimension skaliert mit src_in / dst_in, damit Aktivierungs-Skala erhalten bleibt
    scale = float(src_in) / float(dst_in)
    return tiled * scale


def _tile_vec_1d(src: torch.Tensor, dst_len: int) -> torch.Tensor:
    rep = math.ceil(dst_len / src.numel())
    return src.repeat(rep)[:dst_len]


@torch.no_grad()
def hyperclone_expand_width(small_model: MinotaurModel, large_model: MinotaurModel) -> None:
    """Kopiert Gewichte funktions-erhaltend von schmalerem Modell in breiteres Modell."""
    s_sd = small_model.state_dict()
    l_sd = large_model.state_dict()
    for k, l_t in l_sd.items():
        if k not in s_sd:
            continue
        s_t = s_sd[k]
        if s_t.shape == l_t.shape:
            l_t.copy_(s_t)
        elif s_t.ndim == 2 and l_t.ndim == 2:
            if "tok_emb" in k or "lm_head" in k:
                # Vocab gleich, Embedding-Dim wächst -> replizieren ohne Downscale auf Output
                rep_in = math.ceil(l_t.shape[1] / s_t.shape[1])
                l_t.copy_(s_t.repeat(1, rep_in)[:, : l_t.shape[1]].to(l_t.dtype))
            else:
                l_t.copy_(_tile_weight_2d(s_t.float(), l_t.shape).to(l_t.dtype))
        elif s_t.ndim == 1 and l_t.ndim == 1:
            l_t.copy_(_tile_vec_1d(s_t, l_t.numel()).to(l_t.dtype))
        elif s_t.ndim == 3 and l_t.ndim == 3 and "dw_conv" in k:
            # Depthwise Conv1D (C, 1, K)
            rep_c = math.ceil(l_t.shape[0] / s_t.shape[0])
            l_t.copy_(s_t.repeat(rep_c, 1, 1)[: l_t.shape[0]].to(l_t.dtype))


@torch.no_grad()
def gstack_expand_depth(src_layers: nn.ModuleList, dst_layers: nn.ModuleList) -> None:
    """GStack-Prinzip: Stapelt Layer periodisch von src_layers auf dst_layers."""
    n_src = len(src_layers)
    n_dst = len(dst_layers)
    for idx_dst in range(n_dst):
        idx_src = (idx_dst * n_src) // n_dst
        s_sd = src_layers[idx_src].state_dict()
        d_sd = dst_layers[idx_dst].state_dict()
        for k, d_t in d_sd.items():
            if k not in s_sd:
                continue
            s_t = s_sd[k]
            if s_t.shape == d_t.shape:
                d_t.copy_(s_t)
            elif s_t.ndim == 2 and d_t.ndim == 2:
                d_t.copy_(_tile_weight_2d(s_t.float(), d_t.shape).to(d_t.dtype))
            elif s_t.ndim == 1 and d_t.ndim == 1:
                d_t.copy_(_tile_vec_1d(s_t, d_t.numel()).to(d_t.dtype))
            elif s_t.ndim == 3 and d_t.ndim == 3:
                rep_c = math.ceil(d_t.shape[0] / s_t.shape[0])
                d_t.copy_(s_t.repeat(rep_c, 1, 1)[: d_t.shape[0]].to(d_t.dtype))


def build_stage1_config_for_growth(target_cfg: ExperimentConfig) -> ExperimentConfig:
    """Erzeugt die schmalere & flachere Stage-1-Config für Exp 3B Self-Growth."""
    s1 = copy.deepcopy(target_cfg)
    s1.use_ultramem = False
    s1.d_model = max(64, target_cfg.d_model // target_cfg.growth_stage1_width_div)
    s1.n_heads = max(2, target_cfg.n_heads // target_cfg.growth_stage1_width_div)
    s1.n_kv_heads = max(1, min(s1.n_heads, target_cfg.n_kv_heads // target_cfg.growth_stage1_width_div))
    s1.mlp_dim = max(128, target_cfg.mlp_dim // target_cfg.growth_stage1_width_div)
    s1.n_layers = max(2, target_cfg.n_layers // target_cfg.growth_stage1_depth_div)
    s1.mla_kv_lora_rank = max(32, target_cfg.mla_kv_lora_rank // target_cfg.growth_stage1_width_div)
    return s1


@torch.no_grad()
def grow_model_self_growth(stage1_model: MinotaurModel, target_cfg: ExperimentConfig) -> MinotaurModel:
    """Exp 3B: HyperCloning (Breite) + GStack (Tiefe) + Zero-Init UltraMemV2 Sparse Upcycling."""
    stage2 = MinotaurModel(target_cfg, zero_init_ultramem=True).to(
        next(stage1_model.parameters()).device
    )
    # 1. Breite auf Embedding / Norms / Heads übertragen
    hyperclone_expand_width(stage1_model, stage2)
    # 2. Tiefe via GStack auf alle Target-Layer übertragen
    if not stage1_model.is_looped and not stage2.is_looped:
        gstack_expand_depth(stage1_model.layers, stage2.layers)
    # 3. UltraMemV2 Output bleibt Zero-Init -> Funktion am Growth-Punkt exakt erhalten!
    return stage2


@torch.no_grad()
def upcycle_flat_to_looped(flat_model: MinotaurModel, looped_cfg: ExperimentConfig) -> MinotaurModel:
    """Run 2 Pfad 1: Übernimmt äußere Layer als Prelude/Coda und mittlere Layer als Loop-Core."""
    looped = MinotaurModel(looped_cfg, zero_init_ultramem=True).to(
        next(flat_model.parameters()).device
    )
    # Embeddings & Final Norm übernehmen
    looped.tok_emb.weight.copy_(flat_model.tok_emb.weight.to(looped.tok_emb.weight.dtype))
    looped.ln_f.weight.copy_(flat_model.ln_f.weight.to(looped.ln_f.weight.dtype))

    if flat_model.is_looped:
        # Direktes State-Dict-Laden falls bereits Looped (z.B. Exp 6 auf Exp 5)
        l_sd = looped.state_dict()
        f_sd = flat_model.state_dict()
        for k, v in f_sd.items():
            if k in l_sd and l_sd[k].shape == v.shape:
                l_sd[k].copy_(v.to(l_sd[k].dtype))
        return looped

    src = flat_model.layers
    n_src = len(src)
    n_pre = len(looped.prelude)
    n_core = len(looped.core)
    n_coda = len(looped.coda)

    def _copy_block_partial(s_blk: nn.Module, d_blk: nn.Module) -> None:
        s_sd = s_blk.state_dict()
        d_sd = d_blk.state_dict()
        for k, d_t in d_sd.items():
            if k in s_sd and s_sd[k].shape == d_t.shape:
                d_t.copy_(s_sd[k].to(d_t.dtype))
            elif k.startswith("ln1.") or k.startswith("ln2."):
                # Per-Loop-RMSNorms aus Norm 0 initialisieren
                base_k = k.split(".")[0] + ".0.weight"
                if base_k in s_sd and s_sd[base_k].shape == d_t.shape:
                    d_t.copy_(s_sd[base_k].to(d_t.dtype))

    for i in range(n_pre):
        _copy_block_partial(src[min(i, n_src - 1)], looped.prelude[i])
    mid_start = min(n_pre, max(0, n_src - n_core - n_coda))
    for i in range(n_core):
        _copy_block_partial(src[min(mid_start + i, n_src - 1)], looped.core[i])
    coda_start = max(0, n_src - n_coda)
    for i in range(n_coda):
        _copy_block_partial(src[min(coda_start + i, n_src - 1)], looped.coda[i])

    return looped
