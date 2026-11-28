"""Einheitlicher Experiment-Runner für alle 15 Sub-Runs (Exp 0 bis Exp 6)."""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minotaur_exp.config import ExperimentConfig
from minotaur_exp.data.distill import OfflineDistillCache
from minotaur_exp.data.pipeline import (
    JESTBatchSelector,
    PackedSequenceDataset,
    _iter_dataset_texts,
    prepare_corpus_cache,
)
from minotaur_exp.data.tokenizer import get_tokenizer
from minotaur_exp.eval.profiler import StepTimer, measure_vram_breakdown
from minotaur_exp.eval.reasoning_probes import evaluate_reasoning_probes, generate_training_samples
from minotaur_exp.model.blocks import MinotaurModel
from minotaur_exp.model.growth import (
    build_stage1_config_for_growth,
    grow_model_self_growth,
    upcycle_flat_to_looped,
)
from minotaur_exp.model.losses import cut_cross_entropy_loss, standard_cross_entropy_loss
from minotaur_exp.model.router import CTMLightModule
from minotaur_exp.optim.muon_8bit import build_optimizer
from minotaur_exp.optim.precision import (
    COATSavedTensorContext,
    convert_model_pure_bf16,
    replace_linear_with_ada_fp8,
)
from minotaur_exp.optim.schedule import ModelEMA, Rho1SparseBackwardSchedule, WSDScheduler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_batch(
    dataset: PackedSequenceDataset,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    xs, ys, docs = [], [], []
    for idx in indices:
        x, y, d = dataset[idx]
        xs.append(x)
        ys.append(y)
        docs.append(d)
    return (
        torch.stack(xs).to(device, non_blocking=True),
        torch.stack(ys).to(device, non_blocking=True),
        torch.stack(docs).to(device, non_blocking=True),
        indices,
    )


@torch.no_grad()
def _evaluate_validation(
    model: nn.Module,
    val_ds: PackedSequenceDataset,
    bytes_per_token: float,
    cfg: ExperimentConfig,
    device: torch.device,
    max_batches: int = 8,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_eval = min(max_batches, max(1, len(val_ds) // max(1, cfg.batch_size)))
    rng = random.Random(777)

    for _ in range(n_eval):
        x, y, d, _ = _sample_batch(val_ds, cfg.batch_size, rng, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(x, doc_ids=d if cfg.use_doc_masking else None)
            if cfg.use_cce:
                _, per_tok = cut_cross_entropy_loss(
                    out.hidden_states, model.lm_head.weight, y, chunk_size=cfg.cce_chunk_size
                )
            else:
                _, per_tok = standard_cross_entropy_loss(out.hidden_states, model.lm_head.weight, y)
        total_nll += float(per_tok.sum().item())
        total_tokens += int(per_tok.numel())

    val_loss = total_nll / max(1, total_tokens)
    val_bpb = val_loss / (math.log(2.0) * max(1e-6, bytes_per_token))
    if was_training:
        model.train()
    return val_loss, val_bpb


def _prepare_model_and_optim(
    cfg: ExperimentConfig,
    device: torch.device,
    init_ckpt_path: str | None,
    logger,
) -> tuple[MinotaurModel, nn.Module, object, object]:
    """Erstellt Modell (inkl. optionalem Checkpoint-Laden/Upcycling), Präzision, Compile & Optimizer."""
    if init_ckpt_path and Path(init_ckpt_path).exists():
        ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        src_cfg = ExperimentConfig.from_dict(ckpt["config"])
        if cfg.upcycle_to_loop and not src_cfg.loop.enabled and cfg.loop.enabled:
            logger(f"[{cfg.exp_id}] Loop-Upcycling von '{init_ckpt_path}'...")
            flat_m = MinotaurModel(src_cfg)
            flat_m.load_state_dict(ckpt["model_state"], strict=False)
            raw_model = upcycle_flat_to_looped(flat_m, cfg).to(device)
            del flat_m
        else:
            raw_model = MinotaurModel(cfg).to(device)
            sd = raw_model.state_dict()
            loaded = {k: v for k, v in ckpt["model_state"].items() if k in sd and sd[k].shape == v.shape}
            raw_model.load_state_dict(loaded, strict=False)
            logger(f"[{cfg.exp_id}] Checkpoint geladen ({len(loaded)}/{len(sd)} Tensoren) von '{init_ckpt_path}'.")
    else:
        if cfg.init_from_exp:
            logger(
                f"[{cfg.exp_id}] Hinweis: Kein Checkpoint für '{cfg.init_from_exp}' gefunden -> starte sauber ab Init."
            )
        raw_model = MinotaurModel(cfg).to(device)

    if cfg.precision_mode in ("pure_bf16_sr", "ada_fp8"):
        convert_model_pure_bf16(raw_model)
    else:
        raw_model.to(dtype=torch.float32)

    if cfg.precision_mode == "ada_fp8":
        n_rep = replace_linear_with_ada_fp8(raw_model)
        logger(f"[{cfg.exp_id}] Ada-FP8 aktiviert auf {n_rep} Linear-Layern.")

    exec_model: nn.Module = raw_model
    if cfg.use_compile and device.type == "cuda":
        try:
            # dynamic=False vermeidet Symbolic-Shape-Crash mit MoR + Selective Checkpointing
            exec_model = torch.compile(raw_model, mode=cfg.compile_mode, fullgraph=False, dynamic=False)
        except Exception as exc:
            logger(f"[{cfg.exp_id}] Warnung: torch.compile Fallback auf Eager ({exc}).")
            exec_model = raw_model

    opt, lomo = build_optimizer(raw_model, cfg)
    return raw_model, exec_model, opt, lomo


def _run_sweep_0(
    base_cfg: ExperimentConfig,
    suite_cfg: dict[str, Any],
    train_ds: PackedSequenceDataset,
    val_ds: PackedSequenceDataset,
    bytes_per_token: float,
    device: torch.device,
    exp_log_dir: Path,
    timestamp: str,
    logger,
) -> dict[str, Any]:
    """Exp 0_sweep: Isoliert exakte Deltas durch Deaktivieren von Fused-Ops, CCE und Compile."""
    variants = [
        ("no_fused_ops", {"use_fused_ops": False}),
        ("no_cce", {"use_cce": False}),
        ("no_compile", {"use_compile": False}),
    ]
    sub_minutes = max(0.05, base_cfg.max_minutes / len(variants))
    sub_steps = max(4, min(base_cfg.max_steps, 60))
    sweep_results: dict[str, dict] = {}

    for tag, patch in variants:
        sub_cfg = copy.deepcopy(base_cfg)
        sub_cfg.exp_id = f"0_sweep_{tag}"
        sub_cfg.max_minutes = sub_minutes
        sub_cfg.max_steps = sub_steps
        for k, v in patch.items():
            setattr(sub_cfg, k, v)
        logger(f"[0_sweep] Sub-Run '{tag}' ({sub_steps} max steps, {sub_minutes:.2f} min)...")
        r = _execute_training_run(
            sub_cfg, suite_cfg, train_ds, val_ds, bytes_per_token, None, None, device, exp_log_dir, timestamp, None, logger
        )
        sweep_results[tag] = {
            "tokens_per_sec": r["tokens_per_sec"],
            "peak_allocated_mb": r["vram"]["peak_allocated_mb"],
            "val_bpb": r["val_bpb"],
        }

    avg_tps = sum(v["tokens_per_sec"] for v in sweep_results.values()) / len(sweep_results)
    max_vram = max(v["peak_allocated_mb"] for v in sweep_results.values())
    best_bpb = min(v["val_bpb"] for v in sweep_results.values())
    return {
        "exp_id": "0_sweep",
        "group_id": base_cfg.group_id,
        "name": base_cfg.name,
        "baseline_ref": base_cfg.baseline_ref,
        "status": "OK",
        "val_loss": 0.0,
        "val_bpb": best_bpb,
        "tokens_per_sec": avg_tps,
        "bytes_per_sec": avg_tps * bytes_per_token,
        "ms_per_step": 0.0,
        "forward_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "vram": {
            "weights_mb": 0.0,
            "gradients_mb": 0.0,
            "optimizer_states_mb": 0.0,
            "peak_allocated_mb": max_vram,
            "peak_activations_mb": 0.0,
        },
        "nan_inf_count": 0,
        "loss_spikes": 0,
        "max_grad_norm": 0.0,
        "steps_done": sub_steps * len(variants),
        "elapsed_sec": 0.0,
        "sweep_deltas": sweep_results,
    }


def _execute_training_run(
    cfg: ExperimentConfig,
    suite_cfg: dict[str, Any],
    train_ds: PackedSequenceDataset,
    val_ds: PackedSequenceDataset,
    bytes_per_token: float,
    distill_cache: OfflineDistillCache | None,
    tokenizer,
    device: torch.device,
    exp_log_dir: Path,
    timestamp: str,
    init_ckpt_path: str | None,
    logger,
) -> dict[str, Any]:
    _set_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    is_growth = bool(cfg.use_self_growth)
    active_cfg = build_stage1_config_for_growth(cfg) if is_growth else cfg
    if is_growth:
        logger(
            f"[{cfg.exp_id}] Exp 3B Self-Growth Stage 1: d_model={active_cfg.d_model}, "
            f"n_layers={active_cfg.n_layers} (Wachstum bei {cfg.growth_split_frac*100:.0f}%)"
        )

    raw_model, exec_model, optimizer, lomo = _prepare_model_and_optim(
        active_cfg, device, init_ckpt_path, logger
    )
    ema = ModelEMA(raw_model, decay=cfg.ema_decay, offload_to_cpu=True) if cfg.use_ema else None
    wsd = WSDScheduler(
        total_steps=cfg.max_steps,
        warmup_frac=cfg.wsd_warmup_frac,
        decay_frac=cfg.wsd_decay_frac,
        enabled=cfg.use_wsd,
    )
    rho1 = Rho1SparseBackwardSchedule(cfg.sparse_bw)
    jest = JESTBatchSelector() if cfg.use_jest else None
    timer = StepTimer(device)
    rng = random.Random(cfg.seed + 101)

    metrics_file = exp_log_dir / f"metrics_{timestamp}.jsonl"
    budget_sec = cfg.max_minutes * 60.0
    t_run_start = time.perf_counter()

    step = 0
    nan_inf_count = 0
    loss_spikes = 0
    max_grad_norm = 0.0
    prev_loss: float | None = None
    grew_already = False
    growth_metrics: dict[str, float] = {}

    fwd_times, bwd_times, opt_times, step_times = [], [], [], []
    dense_tps_list, sparse_tps_list = [], []
    last_grad_bytes = 0

    exec_model.train()
    while step < cfg.max_steps and (time.perf_counter() - t_run_start) < budget_sec:
        progress = max(step / max(1, cfg.max_steps), (time.perf_counter() - t_run_start) / max(1e-3, budget_sec))

        # Exp 3B Self-Growth Transition
        if is_growth and (not grew_already) and progress >= cfg.growth_split_frac:
            grew_already = True
            _, pre_bpb = _evaluate_validation(raw_model, val_ds, bytes_per_token, active_cfg, device, max_batches=4)
            if lomo is not None:
                lomo.remove()
            del optimizer, exec_model
            raw_model = grow_model_self_growth(raw_model, cfg)
            convert_model_pure_bf16(raw_model)
            exec_model = raw_model
            if cfg.use_compile and device.type == "cuda":
                try:
                    exec_model = torch.compile(raw_model, mode=cfg.compile_mode, fullgraph=False)
                except Exception:
                    exec_model = raw_model
            optimizer, lomo = build_optimizer(raw_model, cfg)
            ema = ModelEMA(raw_model, decay=cfg.ema_decay, offload_to_cpu=True) if cfg.use_ema else None
            _, post_bpb = _evaluate_validation(raw_model, val_ds, bytes_per_token, cfg, device, max_batches=4)
            growth_metrics = {"pre_growth_bpb": pre_bpb, "post_growth_bpb": post_bpb}
            logger(f"[{cfg.exp_id}] Self-Growth ausgeführt! BPB: {pre_bpb:.4f} -> {post_bpb:.4f}")
            active_cfg = cfg

        lr_scale = wsd.get_lr_scale_by_progress(progress)
        sparse_active, keep_ratio = rho1.status(progress)
        if lomo is not None:
            lomo.begin_step(lr_scale)
        else:
            optimizer.zero_grad(set_to_none=True)

        t_step_0 = timer.start()
        step_fwd_ms = 0.0
        step_bwd_ms = 0.0
        step_loss_val = 0.0
        tokens_this_step = 0
        last_mor_frac = 1.0

        for _accum in range(active_cfg.grad_accum_steps):
            bs = active_cfg.batch_size * (active_cfg.jest_superbatch_mult if jest is not None else 1)
            x, y, d, batch_indices = _sample_batch(train_ds, bs, rng, device)
            if jest is not None:
                x, y, d = jest.select_batch(
                    raw_model, x, y, d, active_cfg.batch_size, use_doc_masking=active_cfg.use_doc_masking
                )

            d_ids, d_lps, ref_nll = (None, None, None)
            if distill_cache is not None:
                d_ids, d_lps, ref_nll = distill_cache.get_batch(batch_indices, device)

            # 1. Forward
            t_f0 = timer.start()
            with COATSavedTensorContext(enabled=active_cfg.use_coat):
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
                ):
                    out = exec_model(x, doc_ids=d if active_cfg.use_doc_masking else None)
                    token_mask = None
                    if sparse_active:
                        # Vorab Per-Token-Loss ohne Grad für Rho-1++ Token-Maske bestimmen
                        with torch.no_grad():
                            _, s_per_tok = cut_cross_entropy_loss(
                                out.hidden_states.detach(),
                                raw_model.lm_head.weight.detach(),
                                y,
                                chunk_size=active_cfg.cce_chunk_size,
                            )
                        token_mask = rho1.compute_token_keep_mask(s_per_tok, ref_nll, keep_ratio)

                    if active_cfg.use_cce:
                        loss, _ = cut_cross_entropy_loss(
                            out.hidden_states,
                            raw_model.lm_head.weight,
                            y,
                            chunk_size=active_cfg.cce_chunk_size,
                            token_mask=token_mask,
                            distill_ids=d_ids,
                            distill_logprobs=d_lps,
                            distill_alpha=active_cfg.distill_alpha,
                            distill_temp=active_cfg.distill_temp,
                        )
                    else:
                        loss, _ = standard_cross_entropy_loss(out.hidden_states, raw_model.lm_head.weight, y)

                    if out.mtp_hidden_states is not None and y.shape[1] >= 2:
                        mtp_loss, _ = cut_cross_entropy_loss(
                            out.mtp_hidden_states,
                            raw_model.lm_head.weight,
                            y[:, 1:],
                            chunk_size=active_cfg.cce_chunk_size,
                        )
                        loss = loss + active_cfg.mtp_weight * mtp_loss

                    if out.aux_loss is not None:
                        loss = loss + out.aux_loss

                    if (
                        active_cfg.loop.enabled
                        and active_cfg.loop.use_ctm_certainty_loss
                        and out.ctm_tick_hiddens is not None
                    ):
                        ctm_l, _ = CTMLightModule.compute_certainty_halting_loss(
                            out.ctm_tick_hiddens, raw_model.lm_head.weight, y
                        )
                        loss = loss + active_cfg.loop.ctm_certainty_weight * ctm_l

                step_fwd_ms += timer.elapsed_ms(t_f0)

                # 2. Backward (innerhalb COAT-Context!)
                t_b0 = timer.start()
                scaled_loss = loss / float(active_cfg.grad_accum_steps)
                scaled_loss.backward()
                step_bwd_ms += timer.elapsed_ms(t_b0)

            step_loss_val += float(loss.detach().item()) / float(active_cfg.grad_accum_steps)
            tokens_this_step += int(x.numel())
            last_mor_frac = out.mor_active_frac

        # Gradient-Bytes vor Optimizer-Step/Zero-Grad erfassen
        last_grad_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in raw_model.parameters()
            if p.grad is not None and p.grad.is_cuda
        )

        # 3. Optimizer Step
        t_o0 = timer.start()
        if lomo is not None:
            grad_norm = lomo.end_step()
        else:
            if active_cfg.grad_clip > 0.0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), active_cfg.grad_clip))
            else:
                grad_norm = 0.0
            optimizer.step(lr_scale=lr_scale)
        step_opt_ms = timer.elapsed_ms(t_o0)
        step_total_ms = timer.elapsed_ms(t_step_0)

        if ema is not None:
            ema.update(raw_model)

        # Stabilitätstracking
        if not math.isfinite(step_loss_val):
            nan_inf_count += 1
        else:
            if prev_loss is not None and step_loss_val > prev_loss * 1.8 and step > 10:
                loss_spikes += 1
            prev_loss = step_loss_val
        max_grad_norm = max(max_grad_norm, float(grad_norm) if math.isfinite(grad_norm) else 0.0)

        fwd_times.append(step_fwd_ms)
        bwd_times.append(step_bwd_ms)
        opt_times.append(step_opt_ms)
        step_times.append(step_total_ms)

        tps_cur = tokens_this_step / max(1e-6, step_total_ms / 1000.0)
        if sparse_active:
            sparse_tps_list.append(tps_cur)
        else:
            dense_tps_list.append(tps_cur)

        if step % active_cfg.log_every_steps == 0 or step == cfg.max_steps - 1:
            rec = {
                "step": step,
                "loss": round(step_loss_val, 4),
                "lr_scale": round(lr_scale, 4),
                "sparse_bw": sparse_active,
                "mor_frac": round(last_mor_frac, 3),
                "tps": round(tps_cur, 1),
                "ms_step": round(step_total_ms, 2),
                "grad_norm": round(grad_norm, 3),
            }
            with metrics_file.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec) + "\n")

        step += 1

    # Finale Validation (ggf. mit EMA)
    ema_backup = ema.apply_to(raw_model) if ema is not None else None
    val_batches = 4 if suite_cfg.get("model_scale") == "tiny_smoke" else 32
    val_loss, val_bpb = _evaluate_validation(
        raw_model, val_ds, bytes_per_token, active_cfg, device, max_batches=val_batches
    )

    reasoning_probes = {}
    if (active_cfg.loop.enabled or cfg.group_id in ("exp5", "exp6")) and tokenizer is not None:
        reasoning_probes = evaluate_reasoning_probes(
            raw_model, tokenizer, device, eval_loops=active_cfg.loop.eval_loops
        )

    vram = measure_vram_breakdown(raw_model, optimizer, grad_bytes_captured=last_grad_bytes)

    ckpt_path = None
    if suite_cfg.get("logging", {}).get("save_checkpoints", True):
        ckpt_path = str(exp_log_dir / "checkpoint.pt")
        torch.save(
            {"config": active_cfg.to_dict(), "model_state": raw_model.state_dict()},
            ckpt_path,
        )

    if ema_backup is not None and ema is not None:
        ema.restore(raw_model, ema_backup)
    if lomo is not None:
        lomo.remove()

    # Erste 2 JIT/Compile-Warmup-Schritte für saubere Steady-State-Speeds ausklammern
    s_slice = slice(2, None) if len(step_times) > 4 else slice(None)
    st_ss = step_times[s_slice]
    fw_ss = fwd_times[s_slice]
    bw_ss = bwd_times[s_slice]
    op_ss = opt_times[s_slice]

    avg_step_ms = sum(st_ss) / max(1, len(st_ss))
    avg_fwd_ms = sum(fw_ss) / max(1, len(fw_ss))
    avg_bwd_ms = sum(bw_ss) / max(1, len(bw_ss))
    avg_opt_ms = sum(op_ss) / max(1, len(op_ss))
    toks_per_step = active_cfg.batch_size * active_cfg.grad_accum_steps * active_cfg.seq_len
    avg_tps = toks_per_step / max(1e-6, avg_step_ms / 1000.0)

    dense_tps = sum(dense_tps_list) / max(1, len(dense_tps_list))
    sparse_tps = sum(sparse_tps_list) / max(1, len(sparse_tps_list)) if sparse_tps_list else dense_tps
    sparse_speedup = sparse_tps / max(1e-6, dense_tps)

    res = {
        "exp_id": cfg.exp_id,
        "group_id": cfg.group_id,
        "name": cfg.name,
        "baseline_ref": cfg.baseline_ref,
        "status": "OK",
        "val_loss": round(val_loss, 4),
        "val_bpb": round(val_bpb, 4),
        "tokens_per_sec": round(avg_tps, 1),
        "bytes_per_sec": round(avg_tps * bytes_per_token, 1),
        "ms_per_step": round(avg_step_ms, 2),
        "forward_ms": round(avg_fwd_ms, 2),
        "backward_ms": round(avg_bwd_ms, 2),
        "optimizer_ms": round(avg_opt_ms, 2),
        "vram": {
            "weights_mb": vram.weights_mb,
            "gradients_mb": vram.gradients_mb,
            "optimizer_states_mb": vram.optimizer_states_mb,
            "peak_allocated_mb": vram.peak_allocated_mb,
            "peak_activations_mb": vram.peak_activations_mb,
        },
        "nan_inf_count": nan_inf_count,
        "loss_spikes": loss_spikes,
        "max_grad_norm": round(max_grad_norm, 3),
        "steps_done": step,
        "elapsed_sec": round(time.perf_counter() - t_run_start, 2),
        "sparse_phase_speedup": round(sparse_speedup, 2),
        "reasoning_probes": reasoning_probes,
        "growth_metrics": growth_metrics,
        "checkpoint_path": ckpt_path,
    }

    del raw_model, exec_model, optimizer, ema
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return res


def run_single_experiment(
    exp_cfg: ExperimentConfig,
    suite_cfg: dict[str, Any],
    log_dir: Path,
    run_timestamp: str,
    shared_state: dict[str, Any],
    logger,
) -> dict[str, Any]:
    """Öffentlicher Einstiegspunkt für ein einzelnes Experiment (inkl. Daten- & Distill-Setup)."""
    exp_log_dir = log_dir / exp_cfg.exp_id
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        suite_cfg.get("hardware", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    data_cfg = suite_cfg.get("data", {})
    preset = str(data_cfg.get("dataset_preset", "fineweb_edu"))
    cache_dir = Path(data_cfg.get("cache_dir", "minotaur_exp/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_toks = int(data_cfg.get("train_tokens", 300_000_000))
    val_toks = int(data_cfg.get("val_tokens", 10_000_000))
    if suite_cfg.get("model_scale") == "tiny_smoke":
        train_toks = min(train_toks, 16384)
        val_toks = min(val_toks, 4096)

    tok_key = ("superbpe", exp_cfg.superbpe_vocab_size) if exp_cfg.use_superbpe else ("gpt2", 50257)
    if tok_key not in shared_state["tokenizers"]:
        def _texts_for_bpe():
            it, _ = _iter_dataset_texts(preset, "train", max_docs=2000, logger=logger)
            return it

        shared_state["tokenizers"][tok_key] = get_tokenizer(
            use_superbpe=exp_cfg.use_superbpe,
            superbpe_vocab_size=exp_cfg.superbpe_vocab_size,
            cache_dir=cache_dir,
            texts_iter_fn=_texts_for_bpe,
            logger=logger,
        )
    tokenizer = shared_state["tokenizers"][tok_key]
    if exp_cfg.use_superbpe:
        exp_cfg.vocab_size = max(exp_cfg.vocab_size, ((tokenizer.vocab_size + 63) // 64) * 64)

    corpus_key = (preset, tok_key)
    if corpus_key not in shared_state["corpora"]:
        shared_state["corpora"][corpus_key] = prepare_corpus_cache(
            preset=preset,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            train_tokens_target=train_toks,
            val_tokens_target=val_toks,
            logger=logger,
        )
    meta = shared_state["corpora"][corpus_key]

    reasoning_samples = None
    if exp_cfg.reasoning_mix_ratio > 0.0:
        reasoning_samples = generate_training_samples(tokenizer, n_samples=150)

    train_ds = PackedSequenceDataset(
        meta["train_bin"],
        seq_len=exp_cfg.seq_len,
        eos_id=meta["eos_id"],
        reasoning_samples=reasoning_samples,
        reasoning_mix_ratio=exp_cfg.reasoning_mix_ratio,
    )
    val_ds = PackedSequenceDataset(
        meta["val_bin"],
        seq_len=exp_cfg.seq_len,
        eos_id=meta["eos_id"],
    )

    if exp_cfg.exp_id == "0_sweep":
        return _run_sweep_0(
            exp_cfg, suite_cfg, train_ds, val_ds, meta["bytes_per_token"], device, exp_log_dir, run_timestamp, logger
        )

    distill_cache = None
    if exp_cfg.use_offline_distill:
        max_chunks = 64 if suite_cfg.get("model_scale") == "tiny_smoke" else 1024
        distill_cache = OfflineDistillCache(
            teacher_name=exp_cfg.teacher_model_name,
            dataset=train_ds,
            cache_dir=cache_dir,
            top_k=exp_cfg.distill_top_k,
            max_chunks=max_chunks,
            device=str(device),
            logger=logger,
        )

    ckpt_overrides = suite_cfg.get("checkpoint_overrides", {}) or {}
    init_ckpt = ckpt_overrides.get(exp_cfg.exp_id)
    if not init_ckpt and exp_cfg.init_from_exp:
        init_ckpt = shared_state["checkpoints"].get(exp_cfg.init_from_exp)

    res = _execute_training_run(
        exp_cfg,
        suite_cfg,
        train_ds,
        val_ds,
        meta["bytes_per_token"],
        distill_cache,
        tokenizer,
        device,
        exp_log_dir,
        run_timestamp,
        init_ckpt,
        logger,
    )
    if res.get("checkpoint_path"):
        shared_state["checkpoints"][exp_cfg.exp_id] = res["checkpoint_path"]
    return res
