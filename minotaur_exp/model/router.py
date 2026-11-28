"""MoR-Token-Router (Run 2) & CTM-Light Module E1, E2, E3 (Run 2 Exp)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoRRouter(nn.Module):
    """Mixture-of-Recursions Router: wählt Top-C komplexeste Tokens für tiefere Loops."""

    def __init__(self, d_model: int, capacity: float = 0.35):
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Gibt (active_idx [B, K], active_weights [B, K, 1], aux_loss, active_frac) zurück."""
        B, T, _ = h.shape
        scores = torch.sigmoid(self.gate(h).squeeze(-1))  # (B, T)
        k = max(1, min(T, int(math.ceil(T * self.capacity))))
        top_vals, top_idx = torch.topk(scores, k, dim=-1, sorted=True)
        # Sortiere nach Sequenzposition, damit Kausalität/RoPE-Ordnung sauber bleibt
        top_idx, order = torch.sort(top_idx, dim=-1)
        top_vals = torch.gather(top_vals, 1, order)

        # Balance-Loss: mittlerer Router-Score soll nahe Ziel-Kapazität bleiben
        aux_loss = (scores.mean() - self.capacity).pow(2)
        return top_idx, top_vals.unsqueeze(-1), aux_loss, float(k / max(1, T))

    @staticmethod
    def gather(h: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
        idx_exp = active_idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        return torch.gather(h, 1, idx_exp)

    @staticmethod
    def scatter_update(
        h_base: torch.Tensor,
        h_active_new: torch.Tensor,
        active_idx: torch.Tensor,
        active_weights: torch.Tensor,
    ) -> torch.Tensor:
        idx_exp = active_idx.unsqueeze(-1).expand(-1, -1, h_base.shape[-1])
        old_active = torch.gather(h_base, 1, idx_exp)
        blended = old_active + active_weights.to(h_base.dtype) * (h_active_new - old_active)
        return h_base.scatter(1, idx_exp, blended)


class CTMLightModule(nn.Module):
    """Sakana AI Continuous Thought Machine (Light-Variante für 4060 Ti 16 GB):
    - E1: Subsampled Neural Synchronization (O(P), P=256 Paare)
    - E2: Certainty-Based Adaptive Halting Loss (dual-point min_loss + max_certainty)
    - E3: Lightweight Tick-History EMA (per-channel EMA statt Per-Neuron-LSTM)
    """

    def __init__(
        self,
        d_model: int,
        use_sync: bool = False,
        sync_pairs: int = 256,
        use_certainty_loss: bool = False,
        use_tick_ema: bool = False,
        ema_decay: float = 0.7,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_sync = use_sync
        self.sync_pairs = sync_pairs
        self.use_certainty_loss = use_certainty_loss
        self.use_tick_ema = use_tick_ema
        self.ema_decay = ema_decay

        if use_sync:
            rng = torch.Generator().manual_seed(1337)
            idx_a = torch.randint(0, d_model, (sync_pairs,), generator=rng)
            idx_b = torch.randint(0, d_model, (sync_pairs,), generator=rng)
            self.register_buffer("sync_idx_a", idx_a)
            self.register_buffer("sync_idx_b", idx_b)
            self.sync_decay = nn.Parameter(torch.full((sync_pairs,), 0.85))
            self.sync_proj = nn.Linear(sync_pairs, d_model, bias=False)
            nn.init.zeros_(self.sync_proj.weight)  # Zero-Init für sicheres Upcycling

        if use_tick_ema:
            self.ema_mix = nn.Parameter(torch.tensor([0.85, 0.15]))

    def init_state(self, h: torch.Tensor) -> dict[str, torch.Tensor | None]:
        B, T, _ = h.shape
        sync_acc = (
            torch.zeros(B, T, self.sync_pairs, device=h.device, dtype=torch.float32)
            if self.use_sync
            else None
        )
        ema_state = h.detach().clone() if self.use_tick_ema else None
        return {"sync": sync_acc, "ema": ema_state}

    def step_tick(self, h: torch.Tensor, state: dict[str, torch.Tensor | None]) -> torch.Tensor:
        if self.use_sync and state["sync"] is not None:
            a = h.index_select(-1, self.sync_idx_a).float()
            b = h.index_select(-1, self.sync_idx_b).float()
            decay = torch.sigmoid(self.sync_decay).view(1, 1, -1)
            state["sync"] = decay * state["sync"] + (1.0 - decay) * (a * b)

        if self.use_tick_ema and state["ema"] is not None:
            state["ema"] = self.ema_decay * state["ema"] + (1.0 - self.ema_decay) * h
            w = F.softmax(self.ema_mix, dim=0).to(h.dtype)
            h = w[0] * h + w[1] * state["ema"].to(h.dtype)

        return h

    def apply_readout(self, h: torch.Tensor, state: dict[str, torch.Tensor | None]) -> torch.Tensor:
        if self.use_sync and state["sync"] is not None:
            normed = F.normalize(state["sync"], dim=-1).to(h.dtype)
            h = h + self.sync_proj(normed)
        return h

    @staticmethod
    def compute_certainty_halting_loss(
        tick_hiddens: list[torch.Tensor],
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_tokens: int = 512,
    ) -> tuple[torch.Tensor, float]:
        """Berechnet E2 CTM-Certainty-Loss auf einem Token-Subsample (spart VRAM)."""
        if not tick_hiddens:
            zero = torch.zeros((), device=targets.device)
            return zero, 1.0

        y_flat = targets.reshape(-1)
        n = y_flat.numel()
        step_stride = max(1, n // chunk_tokens)
        sub_y = y_flat[::step_stride]
        v_size = lm_head_weight.shape[0]
        log_v = math.log(max(2, v_size))

        losses_per_tick = []
        certs_per_tick = []
        for h_t in tick_hiddens:
            h_sub = h_t.reshape(-1, h_t.shape[-1])[::step_stride]
            logits = F.linear(h_sub, lm_head_weight.to(h_sub.dtype)).float()
            ce = F.cross_entropy(logits, sub_y, reduction="none")  # (M,)
            log_p = F.log_softmax(logits, dim=-1)
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1)
            certainty = (1.0 - entropy / log_v).clamp(0.0, 1.0)
            losses_per_tick.append(ce)
            certs_per_tick.append(certainty)

        loss_mat = torch.stack(losses_per_tick, dim=0)  # (R, M)
        cert_mat = torch.stack(certs_per_tick, dim=0)  # (R, M)

        t_min_loss = loss_mat.argmin(dim=0)
        t_max_cert = cert_mat.argmax(dim=0)

        idx = torch.arange(sub_y.numel(), device=sub_y.device)
        l_best = loss_mat[t_min_loss, idx].mean()
        l_cert = loss_mat[t_max_cert, idx].mean()
        ctm_loss = 0.5 * (l_best + l_cert)
        avg_exit_tick = float(t_max_cert.float().mean().item()) + 1.0
        return ctm_loss, avg_exit_tick
