"""UltraMemV2 / PKM-Lookup-Layer: Entkoppelt Parameter (Fakten) von FLOPs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UltraMemV2Layer(nn.Module):
    """Product-Key Memory (PKM) Fakten-Speicher:
    - keys_sqrt = M -> M^2 virtuelle Slots pro Tabelle (z.B. 512^2 = 262.144)
    - Sparse Lookup lädt nur Top-K Vektoren pro Token aus GDDR6
    - Zero-Init Output-Projektion für verlustfreies Sparse Upcycling (Exp 3B)
    """

    def __init__(
        self,
        d_model: int,
        keys_sqrt: int = 512,
        num_tables: int = 2,
        top_k: int = 16,
        val_dim: int = 768,
        key_dim: int = 128,
        zero_init_out: bool = False,
    ):
        super().__init__()
        assert key_dim % 2 == 0
        self.d_model = d_model
        self.keys_sqrt = keys_sqrt
        self.num_keys = keys_sqrt * keys_sqrt
        self.num_tables = num_tables
        self.top_k = top_k
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.half_k = key_dim // 2

        self.q_proj = nn.Linear(d_model, num_tables * key_dim, bias=False)
        self.keys1 = nn.Parameter(torch.randn(num_tables, keys_sqrt, self.half_k) * (1.0 / math.sqrt(self.half_k)))
        self.keys2 = nn.Parameter(torch.randn(num_tables, keys_sqrt, self.half_k) * (1.0 / math.sqrt(self.half_k)))

        # Eine große, geteilte Value-Tabelle pro Table -> (num_tables * num_keys, val_dim)
        self.values = nn.Parameter(torch.randn(num_tables * self.num_keys, val_dim) * 0.02)
        self.out_proj = nn.Linear(num_tables * val_dim, d_model, bias=False)
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        N = B * T
        q = self.q_proj(x).view(N, self.num_tables, self.key_dim)
        q1, q2 = q.split(self.half_k, dim=-1)  # (N, H, half_k)

        sub_k = min(self.keys_sqrt, max(self.top_k, int(math.ceil(math.sqrt(self.top_k))) * 2))
        table_outs = []
        aux_losses = []

        for h in range(self.num_tables):
            s1 = q1[:, h] @ self.keys1[h].t()  # (N, M)
            s2 = q2[:, h] @ self.keys2[h].t()  # (N, M)

            v1, i1 = torch.topk(s1, sub_k, dim=-1)  # (N, sub_k)
            v2, i2 = torch.topk(s2, sub_k, dim=-1)  # (N, sub_k)

            # Kartesisches Produkt der Top-Subkeys
            comb_scores = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).reshape(N, sub_k * sub_k)
            comb_idx = (i1.unsqueeze(-1) * self.keys_sqrt + i2.unsqueeze(-2)).reshape(N, sub_k * sub_k)

            top_scores, top_pos = torch.topk(comb_scores, self.top_k, dim=-1)
            top_idx = torch.gather(comb_idx, 1, top_pos) + h * self.num_keys

            weights = F.softmax(top_scores.float(), dim=-1).to(x.dtype)
            vals = F.embedding(top_idx, self.values)  # (N, top_k, val_dim)
            out_h = torch.einsum("nk,nkd->nd", weights, vals)
            table_outs.append(out_h)

            # Kleiner Load-Balance-Term auf Sub-Keys gegen Memory-Collapse
            p1 = F.softmax(s1.float(), dim=-1).mean(dim=0)
            p2 = F.softmax(s2.float(), dim=-1).mean(dim=0)
            aux_losses.append((p1.pow(2).sum() + p2.pow(2).sum()) * self.keys_sqrt)

        merged = torch.cat(table_outs, dim=-1).view(B, T, self.num_tables * self.val_dim)
        aux = torch.stack(aux_losses).mean()
        return self.out_proj(merged), aux
