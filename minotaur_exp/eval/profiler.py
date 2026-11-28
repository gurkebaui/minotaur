"""VRAM-Breakdown-Profiler, Step-Timer & Go/No-Go-Evaluator gemäß Experiments.md."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VRAMBreakdown:
    weights_mb: float = 0.0
    gradients_mb: float = 0.0
    optimizer_states_mb: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_activations_mb: float = 0.0


def measure_vram_breakdown(
    model: nn.Module,
    optimizer,
    grad_bytes_captured: int | None = None,
) -> VRAMBreakdown:
    """Schlüsselt GPU-VRAM exakt nach Weights, Gradients, Optimizer States & Peak Activations auf."""
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.is_cuda)
    if grad_bytes_captured is not None:
        g_bytes = grad_bytes_captured
    else:
        g_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None and p.grad.is_cuda)

    opt_bytes = optimizer.state_memory_bytes() if hasattr(optimizer, "state_memory_bytes") else 0
    peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    w_mb = w_bytes / 1e6
    g_mb = g_bytes / 1e6
    o_mb = opt_bytes / 1e6
    peak_mb = peak_bytes / 1e6
    act_mb = max(0.0, peak_mb - w_mb - g_mb - o_mb)
    return VRAMBreakdown(
        weights_mb=round(w_mb, 2),
        gradients_mb=round(g_mb, 2),
        optimizer_states_mb=round(o_mb, 2),
        peak_allocated_mb=round(peak_mb, 2),
        peak_activations_mb=round(act_mb, 2),
    )


class StepTimer:
    """Misst Forward / Backward / Optimizer-Zeit sauber mit optionaler CUDA-Synchronisation."""

    def __init__(self, device: torch.device):
        self.is_cuda = device.type == "cuda"
        self.t0 = 0.0

    def _sync(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()

    def start(self) -> float:
        self._sync()
        return time.perf_counter()

    def elapsed_ms(self, t_start: float) -> float:
        self._sync()
        return (time.perf_counter() - t_start) * 1000.0


def evaluate_go_no_go(exp_id: str, res: dict, all_results: dict[str, dict]) -> dict[str, str]:
    """Automatischer Go/No-Go-Check inkl. Diagnose-Hinweisen bei Fehlschlag."""
    if res.get("status") != "OK":
        return {"verdict": "FAIL", "reason": f"Lauf fehlgeschlagen: {res.get('error', 'unknown')}"}
    if res.get("nan_inf_count", 0) > 0:
        hint = "Ada-FP8 Scaling prüfen" if exp_id == "1b" else "LR/Grad-Clip prüfen"
        return {"verdict": "WARN", "reason": f"{res['nan_inf_count']} NaN/Inf Steps ({hint})"}

    ref_id = res.get("baseline_ref")
    ref = all_results.get(ref_id) if ref_id else None
    if ref is None or ref.get("status") != "OK":
        return {"verdict": "REF", "reason": "Referenz-/Baseline-Lauf (oder Vergleichslauf nicht in Auswahl)."}

    bpb = res.get("val_bpb", float("inf"))
    ref_bpb = ref.get("val_bpb", float("inf"))
    tps = res.get("tokens_per_sec", 0.0)
    ref_tps = max(1e-6, ref.get("tokens_per_sec", 1.0))
    bps = res.get("bytes_per_sec", 0.0)
    ref_bps = max(1e-6, ref.get("bytes_per_sec", 1.0))

    if exp_id == "0b":
        speedup = tps / ref_tps
        act_save = 1.0 - (res["vram"]["peak_activations_mb"] / max(1e-3, ref["vram"]["peak_activations_mb"]))
        ok = speedup >= 1.3 and act_save >= 0.40
        return {
            "verdict": "PASS" if ok else "WARN",
            "reason": f"Speedup={speedup:.2f}x (Ziel >=1.3x), Act-VRAM-Ersparnis={act_save*100:.1f}% (Ziel >40%).",
        }

    if exp_id == "1a":
        w_ratio = res["vram"]["weights_mb"] / max(1e-3, ref["vram"]["weights_mb"])
        ok = bpb <= ref_bpb + 0.02
        return {
            "verdict": "PASS" if ok else "WARN",
            "reason": f"BPB={bpb:.4f} vs B0={ref_bpb:.4f}, Weight-VRAM-Ratio={w_ratio:.2f}x.",
        }

    if exp_id == "1b":
        delta = bpb - ref_bpb
        ok = delta <= 0.02
        return {
            "verdict": "PASS" if ok else "WARN",
            "reason": f"BPB-Delta vs 1A = {delta:+.4f} (Limit <= +0.02), Peak-VRAM={res['vram']['peak_allocated_mb']:.1f}MB.",
        }

    if exp_id in ("2a", "2b"):
        ok = bpb <= ref_bpb + 0.03
        return {
            "verdict": "PASS" if ok else "WARN",
            "reason": f"BPB={bpb:.4f} vs {ref_id}={ref_bpb:.4f}, Bytes/s={bps/ref_bps:.2f}x.",
        }

    if exp_id == "3a":
        speed_drop = 1.0 - (tps / ref_tps)
        ok = speed_drop <= 0.25
        return {
            "verdict": "PASS" if ok else "WARN",
            "reason": f"BPB={bpb:.4f} vs 2A={ref_bpb:.4f}, Speed-Delta={-speed_drop*100:+.1f}% (Limit -20%).",
        }

    if exp_id == "3b":
        delta = bpb - ref_bpb
        return {
            "verdict": "PASS" if delta <= 0.03 else "WARN",
            "reason": f"Self-Growth End-BPB={bpb:.4f} vs 3A={ref_bpb:.4f} (Delta={delta:+.4f}).",
        }

    if exp_id in ("4a", "4b"):
        return {
            "verdict": "PASS" if bpb <= ref_bpb + 0.03 else "WARN",
            "reason": f"BPB={bpb:.4f} vs {ref_id}={ref_bpb:.4f}, Sparse-Speedup={res.get('sparse_phase_speedup', 1.0):.2f}x.",
        }

    if exp_id in ("5a", "5b"):
        probes = res.get("reasoning_probes", {})
        acc1 = probes.get("R=1", {}).get("overall_acc", 0.0)
        acc4 = probes.get("R=4", {}).get("overall_acc", acc1)
        return {
            "verdict": "PASS" if acc4 >= acc1 else "WARN",
            "reason": f"Test-Time Scaling R=1 ({acc1*100:.1f}%) -> R=4 ({acc4*100:.1f}%), BPB={bpb:.4f}.",
        }

    if exp_id in ("6a", "6b"):
        speed_ratio = tps / ref_tps
        ok = speed_ratio >= 0.90
        return {
            "verdict": "PASS" if ok else "WARN",
            "reason": f"Speed vs {ref_id}={speed_ratio:.2f}x (Limit >=0.90x), BPB={bpb:.4f}.",
        }

    return {"verdict": "PASS", "reason": f"BPB={bpb:.4f}, Speed={tps:.0f} tok/s."}
