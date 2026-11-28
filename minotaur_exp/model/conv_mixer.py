"""LFM3 Double-Gated Short Convolution (k=7) mit EoS-Document-Reset & O(1) Ringbuffer-Decode."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleGatedShortConv(nn.Module):
    """Liquid-AI LFM2/LFM3 Gated Short Conv (k=7):
    u = B_gate * x_v -> causal depthwise conv1d(k=7) -> C_gate * conv_out -> out_proj.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        linear_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        mk_lin = linear_factory or (lambda i, o: nn.Linear(i, o, bias=False))

        self.in_proj = mk_lin(d_model, 3 * d_model)
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=True,
        )
        self.out_proj = mk_lin(d_model, d_model)

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        b_gate, c_gate, x_v = self.in_proj(x).chunk(3, dim=-1)
        u = (torch.sigmoid(b_gate) * x_v).transpose(1, 2)  # (B, D, T)

        # Kausales Left-Padding (k-1)
        u_pad = F.pad(u, (self.kernel_size - 1, 0))
        if doc_ids is not None:
            # Verhindert Konvolution über Dokumentgrenzen hinweg (EoS-Aligned)
            # Nullt Inputs vor einem neuen Dokument innerhalb des Fensters
            is_boundary = torch.zeros_like(doc_ids, dtype=torch.bool)
            is_boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
            # Schnelle Maskierung: an Boundary-Startpositionen kausale Historie dämpfen
            if is_boundary.any():
                # Setze u an Boundary-Vorgängern nur lokal über Unfold oder einfache Boundary-Reset-Maske
                # Exakt: für jedes Fenster prüfe gleiche doc_id
                u_win = u_pad.unfold(dimension=-1, size=self.kernel_size, step=1)  # (B, D, T, k)
                doc_pad = F.pad(doc_ids, (self.kernel_size - 1, 0), value=-1)
                doc_win = doc_pad.unfold(dimension=-1, size=self.kernel_size, step=1)  # (B, T, k)
                valid = (doc_win == doc_ids.unsqueeze(-1)).unsqueeze(1).to(u_win.dtype)
                w = self.dw_conv.weight.view(1, D, 1, self.kernel_size).to(u_win.dtype)
                y = (u_win * valid * w).sum(dim=-1) + self.dw_conv.bias.view(1, D, 1).to(u_win.dtype)
            else:
                y = self.dw_conv(u_pad)
        else:
            y = self.dw_conv(u_pad)

        y = y.transpose(1, 2) * torch.sigmoid(c_gate)
        return self.out_proj(y)

    @torch.no_grad()
    def step_ringbuffer(
        self, x_t: torch.Tensor, buffer: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """O(1)-Decode-Schritt mit 7-Token-Ringbuffer für Inferenz."""
        B, _, D = x_t.shape
        if buffer is None:
            buffer = torch.zeros(B, D, self.kernel_size, device=x_t.device, dtype=x_t.dtype)
        b_gate, c_gate, x_v = self.in_proj(x_t).chunk(3, dim=-1)
        u_t = (torch.sigmoid(b_gate) * x_v).squeeze(1)  # (B, D)
        new_buf = torch.cat([buffer[:, :, 1:], u_t.unsqueeze(-1)], dim=-1)
        w = self.dw_conv.weight.view(1, D, self.kernel_size).to(new_buf.dtype)
        y_t = (new_buf * w).sum(dim=-1) + self.dw_conv.bias.view(1, D).to(new_buf.dtype)
        y_t = (y_t * torch.sigmoid(c_gate.squeeze(1))).unsqueeze(1)
        return self.out_proj(y_t), new_buf
