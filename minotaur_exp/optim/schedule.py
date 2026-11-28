"""Scheduler für LR (WSD), Model-EMA (CPU-offloadbar) und Rho-1++ Sparse Backward."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class WSDScheduler:
    """Warmup-Stable-Decay Learning-Rate-Schedule."""

    def __init__(
        self,
        total_steps: int,
        warmup_frac: float = 0.05,
        decay_frac: float = 0.10,
        min_lr_ratio: float = 0.05,
        enabled: bool = True,
    ):
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(round(self.total_steps * warmup_frac)))
        self.decay_steps = max(1, int(round(self.total_steps * decay_frac)))
        self.decay_start = max(self.warmup_steps, self.total_steps - self.decay_steps)
        self.min_lr_ratio = min_lr_ratio
        self.enabled = enabled

    def get_lr_scale(self, step: int) -> float:
        return self.get_lr_scale_by_progress(min(1.0, step / max(1, self.total_steps)))

    def get_lr_scale_by_progress(self, progress: float) -> float:
        p = max(0.0, min(1.0, float(progress)))
        w = max(1e-4, self.warmup_steps / self.total_steps)
        d_start = max(w, self.decay_start / self.total_steps)
        if not self.enabled:
            if p < w:
                return max(0.05, p / w)
            c_prog = (p - w) / max(1e-4, 1.0 - w)
            return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * c_prog))

        if p < w:
            return max(0.05, p / w)
        if p < d_start:
            return 1.0
        d_prog = (p - d_start) / max(1e-4, 1.0 - d_start)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * d_prog))


class ModelEMA:
    """CPU-offloaded EMA-Shadow-Weights (0 MB GPU-VRAM-Overhead auf 4060 Ti!)."""

    def __init__(self, model: nn.Module, decay: float = 0.995, offload_to_cpu: bool = True):
        self.decay = decay
        self.offload_to_cpu = offload_to_cpu
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                t = p.detach().clone()
                if offload_to_cpu:
                    t = t.cpu()
                self.shadow[name] = t

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            src = p.detach().cpu() if self.offload_to_cpu else p.detach()
            self.shadow[name].mul_(d).add_(src, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])


class Rho1SparseBackwardSchedule:
    """Exakter Zeitplan für Token-Selective Sparse Backward:
    - 0% .. start_frac (40%): Aus (100% Tokens)
    - start_frac .. end_frac (40%..90%): Aktiv (cutoff_ratio=55% -> Top 45% Tokens)
    - end_frac .. 100% (90%..100% WSD-Cooldown): Aus (100% Tokens)
    """

    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.enabled)
        self.start_frac = float(cfg.start_frac)
        self.end_frac = float(cfg.end_frac)
        self.cutoff_ratio = float(cfg.cutoff_ratio)
        self.mode = str(cfg.mode)
        self.ema_token_loss: float | None = None

    def status(self, progress: float) -> tuple[bool, float]:
        """Gibt (is_active, keep_ratio) für Fortschritt in [0, 1] zurück."""
        if not self.enabled:
            return False, 1.0
        if self.start_frac <= progress < self.end_frac:
            return True, max(0.1, min(1.0, 1.0 - self.cutoff_ratio))
        return False, 1.0

    @torch.no_grad()
    def compute_token_keep_mask(
        self,
        student_losses: torch.Tensor,
        ref_losses: torch.Tensor | None,
        keep_ratio: float,
    ) -> torch.Tensor:
        """Berechnet Bool-Maske (N,) der Top-`keep_ratio` Tokens nach Excess-Loss."""
        flat_s = student_losses.detach().reshape(-1).float()
        n = flat_s.numel()
        if keep_ratio >= 0.999 or n == 0:
            return torch.ones(n, dtype=torch.bool, device=flat_s.device)

        if ref_losses is not None:
            scores = flat_s - ref_losses.detach().reshape(-1).float()
        else:
            # Self-EMA Modus: Tokens mit hohem Excess über aktuellem laufenden Schnitt behalten
            cur_mean = float(flat_s.mean().item())
            if self.ema_token_loss is None:
                self.ema_token_loss = cur_mean
            else:
                self.ema_token_loss = 0.95 * self.ema_token_loss + 0.05 * cur_mean
            scores = flat_s - self.ema_token_loss

        k = max(1, int(math.ceil(n * keep_ratio)))
        _, top_idx = torch.topk(scores, k, sorted=False)
        mask = torch.zeros(n, dtype=torch.bool, device=flat_s.device)
        mask[top_idx] = True
        return mask
