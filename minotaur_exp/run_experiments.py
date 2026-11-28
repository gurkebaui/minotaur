"""Übergeordnetes Master-Script zum Starten ausgewählter Experimente mit Datum/Uhrzeit-Logging."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Erlaubt Ausführung sowohl via `python minotaur_exp/run_experiments.py` als auch `-m minotaur_exp.run_experiments`
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from minotaur_exp.eval.profiler import evaluate_go_no_go
from minotaur_exp.suite_config import (
    build_experiment_registry,
    load_suite_config,
    resolve_selection,
)
from minotaur_exp.train_eval import run_single_experiment


def _create_timestamped_log_dir(logs_root: Path, run_tag: str = "") -> tuple[Path, str]:
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{now_str}_{run_tag.strip()}" if run_tag.strip() else now_str
    candidate = logs_root / folder_name
    idx = 1
    while candidate.exists():
        candidate = logs_root / f"{folder_name}_{idx:02d}"
        idx += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, now_str


class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def __call__(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _write_summaries(
    log_dir: Path,
    timestamp: str,
    selected_ids: list[str],
    results: dict[str, dict[str, Any]],
) -> None:
    csv_path = log_dir / f"summary_{timestamp}.csv"
    md_path = log_dir / f"summary_{timestamp}.md"

    fieldnames = [
        "exp_id",
        "name",
        "status",
        "verdict",
        "val_loss",
        "val_bpb",
        "tokens_per_sec",
        "bytes_per_sec",
        "ms_per_step",
        "weights_mb",
        "gradients_mb",
        "optimizer_states_mb",
        "peak_allocated_mb",
        "peak_activations_mb",
        "nan_inf_count",
        "loss_spikes",
        "steps_done",
        "elapsed_sec",
        "verdict_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for eid in selected_ids:
            if eid not in results:
                continue
            r = results[eid]
            gng = evaluate_go_no_go(eid, r, results)
            v = r.get("vram", {})
            writer.writerow(
                {
                    "exp_id": eid,
                    "name": r.get("name", ""),
                    "status": r.get("status", "ERR"),
                    "verdict": gng["verdict"],
                    "val_loss": r.get("val_loss", ""),
                    "val_bpb": r.get("val_bpb", ""),
                    "tokens_per_sec": r.get("tokens_per_sec", ""),
                    "bytes_per_sec": r.get("bytes_per_sec", ""),
                    "ms_per_step": r.get("ms_per_step", ""),
                    "weights_mb": v.get("weights_mb", ""),
                    "gradients_mb": v.get("gradients_mb", ""),
                    "optimizer_states_mb": v.get("optimizer_states_mb", ""),
                    "peak_allocated_mb": v.get("peak_allocated_mb", ""),
                    "peak_activations_mb": v.get("peak_activations_mb", ""),
                    "nan_inf_count": r.get("nan_inf_count", ""),
                    "loss_spikes": r.get("loss_spikes", ""),
                    "steps_done": r.get("steps_done", ""),
                    "elapsed_sec": r.get("elapsed_sec", ""),
                    "verdict_reason": gng["reason"],
                }
            )

    lines = [
        f"# Minotaur Experiment Report ({timestamp})",
        "",
        "| ID | Status | Verdict | Val BPB | Tok/s | Bytes/s | Weights MB | Grads MB | Opt MB | Peak MB | Hinweis |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for eid in selected_ids:
        if eid not in results:
            continue
        r = results[eid]
        gng = evaluate_go_no_go(eid, r, results)
        v = r.get("vram", {})
        lines.append(
            f"| `{eid}` | {r.get('status')} | **{gng['verdict']}** | "
            f"{r.get('val_bpb', '-')} | {r.get('tokens_per_sec', '-')} | {r.get('bytes_per_sec', '-')} | "
            f"{v.get('weights_mb', '-')} | {v.get('gradients_mb', '-')} | {v.get('optimizer_states_mb', '-')} | "
            f"{v.get('peak_allocated_mb', '-')} | {gng['reason']} |"
        )
    lines.append("")

    # Zusatz-Details für Reasoning-Probes (Exp 5/6) & Self-Growth (3B) & Sweep (0_sweep)
    for eid in selected_ids:
        r = results.get(eid)
        if not r:
            continue
        if r.get("reasoning_probes"):
            lines.append(f"## Reasoning-Probes `{eid}`")
            for loop_k, metrics in r["reasoning_probes"].items():
                lines.append(
                    f"- **{loop_k}**: Overall={metrics['overall_acc']*100:.1f}%, "
                    f"Arith={metrics['arithmetic_acc']*100:.1f}%, "
                    f"Parity={metrics['parity_acc']*100:.1f}%, "
                    f"Brackets={metrics['brackets_acc']*100:.1f}%"
                )
            lines.append("")
        if r.get("growth_metrics"):
            gm = r["growth_metrics"]
            lines.append(
                f"## Self-Growth `{eid}`: Pre-Growth BPB={gm.get('pre_growth_bpb', '-'):.4f}, "
                f"Post-Growth BPB={gm.get('post_growth_bpb', '-'):.4f}\n"
            )
        if r.get("sweep_deltas"):
            lines.append(f"## Exp 0 Sweep Deltas (`{eid}`)")
            for tag, sd in r["sweep_deltas"].items():
                lines.append(
                    f"- `{tag}`: {sd['tokens_per_sec']:.1f} tok/s, Peak VRAM={sd['peak_allocated_mb']:.1f} MB, BPB={sd['val_bpb']:.4f}"
                )
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def _print_registry(registry) -> None:
    print("\nVerfügbare Experimente (auswahlbar via --exp oder suite_config.yaml):")
    print("-" * 92)
    for eid, cfg in registry.items():
        print(f"  {eid:<8} | {cfg.group_id:<5} | {cfg.max_minutes:>5.1f} min | {cfg.name}")
    print("-" * 92)
    print("Gruppen-Aliase: exp0 (0), exp1 (1), ..., exp6 (6) oder 'all'\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minotaur Experiment Orchestrator")
    parser.add_argument("--config", type=str, default=None, help="Pfad zur suite_config.yaml")
    parser.add_argument("--exp", nargs="+", default=None, help="Auswahl: z.B. --exp 0 1a 5b oder all")
    parser.add_argument("--exclude", nargs="+", default=None, help="Auszuschließende IDs/Gruppen")
    parser.add_argument("--interactive", action="store_true", help="Interaktive Experiment-Auswahl im Terminal")
    parser.add_argument("--list", action="store_true", help="Alle verfügbaren Experimente anzeigen und beenden")
    parser.add_argument("--smoke", action="store_true", help="Schneller End-to-End Smoke-Test (tiny_smoke + synthetic)")
    parser.add_argument("--dataset-preset", choices=["fineweb_edu", "tinystories", "synthetic"], default=None)
    parser.add_argument("--time-scale", type=float, default=None, help="Zeitfaktor für max_minutes")
    parser.add_argument("--tag", type=str, default=None, help="Optionaler Tag im Log-Ordnernamen")
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", default=None)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    args = parser.parse_args()

    suite_cfg = load_suite_config(args.config)
    if args.dataset_preset is not None:
        suite_cfg["data"]["dataset_preset"] = args.dataset_preset
    if args.time_scale is not None:
        suite_cfg["time_scale"] = args.time_scale
    if args.tag is not None:
        suite_cfg["logging"]["run_tag"] = args.tag
    if args.continue_on_error is not None:
        suite_cfg["logging"]["continue_on_error"] = args.continue_on_error

    if args.smoke:
        suite_cfg["model_scale"] = "tiny_smoke"
        if args.dataset_preset is None:
            suite_cfg["data"]["dataset_preset"] = "synthetic"
        suite_cfg["logging"]["log_every_steps"] = 5
        suite_cfg["logging"]["eval_every_steps"] = 12

    registry = build_experiment_registry(suite_cfg)
    if args.smoke:
        for cfg in registry.values():
            cfg.max_steps = 12
            cfg.max_minutes = 1.0
            cfg.eval_every_steps = 12
            cfg.log_every_steps = 5

    if args.list:
        _print_registry(registry)
        return 0

    sel_tokens = args.exp if args.exp is not None else suite_cfg.get("selected_experiments", ["all"])
    if args.interactive:
        _print_registry(registry)
        raw = input("Welche Experimente starten? (z.B. '0a 1a 5' oder 'all', Enter=Config): ").strip()
        if raw:
            sel_tokens = raw.split()

    selected_ids = resolve_selection(sel_tokens, registry, exclude=args.exclude)
    if not selected_ids:
        print("Keine Experimente ausgewählt. Beende.")
        return 1

    logs_root = Path(suite_cfg["logging"]["logs_root"])
    log_dir, timestamp = _create_timestamped_log_dir(logs_root, suite_cfg["logging"].get("run_tag", ""))
    logger = TeeLogger(log_dir / f"run_{timestamp}.log")

    if suite_cfg.get("hardware", {}).get("allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_rev = "unknown"

    snapshot = {
        "timestamp": timestamp,
        "git_rev": git_rev,
        "selected_experiments": selected_ids,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "suite_config": suite_cfg,
    }
    (log_dir / f"suite_config_snapshot_{timestamp}.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )

    logger(f"Minotaur Suite gestartet | Log-Ordner: {log_dir}")
    logger(f"Ausgewählte Experimente ({len(selected_ids)}): {', '.join(selected_ids)}")
    logger(f"Scale={suite_cfg['model_scale']}, Data={suite_cfg['data']['dataset_preset']}, GPU={snapshot['gpu']}")

    shared_state: dict[str, Any] = {"tokenizers": {}, "corpora": {}, "checkpoints": {}}
    results: dict[str, dict[str, Any]] = {}
    continue_on_err = bool(suite_cfg["logging"].get("continue_on_error", True))

    for idx, eid in enumerate(selected_ids, start=1):
        cfg = registry[eid]
        logger("=" * 72)
        logger(f"[{idx}/{len(selected_ids)}] Starte {cfg.exp_id}: {cfg.name} (Budget: {cfg.max_minutes:.1f} min)")
        try:
            res = run_single_experiment(cfg, suite_cfg, log_dir, timestamp, shared_state, logger)
            results[eid] = res
            logger(
                f"[{eid}] Fertig: BPB={res['val_bpb']:.4f}, Tok/s={res['tokens_per_sec']:.1f}, "
                f"Peak VRAM={res['vram']['peak_allocated_mb']:.1f} MB (Grads={res['vram']['gradients_mb']:.1f} MB)"
            )
        except Exception as exc:
            logger(f"[{eid}] FEHLER: {type(exc).__name__}: {exc}")
            results[eid] = {"exp_id": eid, "name": cfg.name, "status": "FAIL", "error": str(exc), "vram": {}}
            if not continue_on_err:
                _write_summaries(log_dir, timestamp, selected_ids, results)
                raise
        _write_summaries(log_dir, timestamp, selected_ids, results)

    logger("=" * 72)
    logger(f"Suite abgeschlossen! Zusammenfassung unter: {log_dir / f'summary_{timestamp}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
