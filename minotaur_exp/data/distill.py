"""Offline Top-K Teacher Logit Cache für Exp 4B / Modell A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from minotaur_exp.data.pipeline import PackedSequenceDataset


class OfflineDistillCache:
    """Berechnet 1x vorab Top-K Teacher-Logits & NLL auf SSD (für Exp 4B & Teacher Rho-1++)."""

    def __init__(
        self,
        teacher_name: str,
        dataset: PackedSequenceDataset,
        cache_dir: Path,
        top_k: int = 16,
        max_chunks: int = 512,
        device: str = "cuda",
        logger=None,
    ):
        self.top_k = top_k
        safe_name = teacher_name.replace("/", "_")
        out_dir = cache_dir / f"distill_{safe_name}_k{top_k}_n{max_chunks}_s{dataset.seq_len}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ids_path = out_dir / "topk_ids.npy"
        lps_path = out_dir / "topk_logprobs.npy"
        nll_path = out_dir / "teacher_nll.npy"

        n_use = min(len(dataset), max_chunks)
        if not (ids_path.exists() and lps_path.exists() and nll_path.exists()):
            if logger:
                logger(f"[Distill] Erstelle Offline Top-{top_k} Cache von '{teacher_name}' ({n_use} Chunks)...")
            teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.bfloat16).to(device)
            teacher.eval()
            t_vocab = int(teacher.config.vocab_size)

            all_ids = np.empty((n_use, dataset.seq_len, top_k), dtype=np.int32)
            all_lps = np.empty((n_use, dataset.seq_len, top_k), dtype=np.float16)
            all_nll = np.empty((n_use, dataset.seq_len), dtype=np.float16)

            bs = 4
            with torch.no_grad():
                for start in range(0, n_use, bs):
                    end = min(n_use, start + bs)
                    xs, ys = [], []
                    for i in range(start, end):
                        x_i, y_i, _ = dataset[i]
                        xs.append(x_i)
                        ys.append(y_i)
                    x_b = torch.stack(xs).to(device).clamp(max=t_vocab - 1)
                    y_b = torch.stack(ys).to(device).clamp(max=t_vocab - 1)
                    logits = teacher(x_b).logits.float()
                    lp = F.log_softmax(logits, dim=-1)
                    topk_lp, topk_id = torch.topk(lp, top_k, dim=-1)
                    nll = F.nll_loss(lp.view(-1, lp.size(-1)), y_b.view(-1), reduction="none").view_as(y_b)

                    all_ids[start:end] = topk_id.cpu().numpy().astype(np.int32)
                    all_lps[start:end] = topk_lp.half().cpu().numpy()
                    all_nll[start:end] = nll.half().cpu().numpy()

            del teacher
            torch.cuda.empty_cache()

            np.save(ids_path, all_ids)
            np.save(lps_path, all_lps)
            np.save(nll_path, all_nll)

        self.topk_ids = np.load(ids_path, mmap_mode="r")
        self.topk_logprobs = np.load(lps_path, mmap_mode="r")
        self.teacher_nll = np.load(nll_path, mmap_mode="r")
        self.n_cached = self.topk_ids.shape[0]

    def get_batch(self, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = [i % self.n_cached for i in indices]
        ids = torch.from_numpy(np.asarray(self.topk_ids[idx])).to(device)
        lps = torch.from_numpy(np.asarray(self.topk_logprobs[idx])).to(device)
        nll = torch.from_numpy(np.asarray(self.teacher_nll[idx])).to(device)
        return ids, lps, nll
