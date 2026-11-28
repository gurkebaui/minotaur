"""Daten-Pipeline: Korpus-Caching (FineWeb-Edu / TinyStories / Synthetic), EoS-Packing & JEST-Sampler."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset

from minotaur_exp.data.tokenizer import TokenizerWrapper


def _synthetic_text_stream(n_docs: int, seed: int = 42) -> Iterator[str]:
    """Deterministischer lokaler Text+Code+Arithmetik-Korpus für Smoke-Tests."""
    rng = random.Random(seed)
    templates = [
        "def compute_sum(n):\n    total = 0\n    for i in range(n):\n        total += i * 2\n    return total\n",
        "The RTX 4060 Ti uses Ada Lovelace architecture with 32 MB L2 cache and 288 GB/s GDDR6 bandwidth. ",
        "In looped transformers, input injection prevents representation drift across recurrent reasoning steps. ",
        "CALC: {a} + {b} - {c} = {res}. Step by step verification confirms the result is {res}.\n",
        "PARITY: 1 0 1 1 0 -> 1. Tracking state transitions modulo two yields an exact deterministic answer.\n",
    ]
    for i in range(n_docs):
        a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 30)
        res = a + b - c
        t1 = templates[i % len(templates)].format(a=a, b=b, c=c, res=res)
        t2 = templates[(i * 3 + 1) % len(templates)].format(a=a + 1, b=b + 2, c=c, res=res + 3)
        yield t1 + " " + t2


def _iter_dataset_texts(preset: str, split: str, max_docs: int, logger=None) -> tuple[Iterator[str], str]:
    if preset == "synthetic":
        seed = 42 if split == "train" else 9999
        return _synthetic_text_stream(max_docs, seed=seed), "synthetic"

    if preset == "fineweb_edu":
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            # Val-Split aus festen ersten Docs, Train danach
            def _gen():
                skip = 0 if split == "validation" else 2000
                yielded = 0
                for idx, item in enumerate(ds):
                    if idx < skip:
                        continue
                    yield item["text"]
                    yielded += 1
                    if yielded >= max_docs:
                        break
            return _gen(), "fineweb_edu"
        except Exception as exc:
            if logger:
                logger(f"[Data] Warnung: fineweb_edu Stream fehlgeschlagen ({exc}); Fallback auf TinyStories.")
            preset = "tinystories"

    # TinyStories (bereits lokal im HF-Cache vorhanden)
    hf_split = "validation" if split == "validation" else "train"
    try:
        ds = load_dataset("roneneldan/TinyStories", split=hf_split, streaming=True)
        def _gen_ts():
            yielded = 0
            for item in ds:
                yield item["text"]
                yielded += 1
                if yielded >= max_docs:
                    break
        return _gen_ts(), "tinystories"
    except Exception as exc:
        if logger:
            logger(f"[Data] Warnung: TinyStories fehlgeschlagen ({exc}); Fallback auf synthetic.")
        seed = 42 if split == "train" else 9999
        return _synthetic_text_stream(max_docs, seed=seed), "synthetic"


def prepare_corpus_cache(
    preset: str,
    tokenizer: TokenizerWrapper,
    cache_dir: Path,
    train_tokens_target: int,
    val_tokens_target: int,
    logger=None,
) -> dict:
    """Erstellt oder lädt deterministischen uint32-Binär-Cache inkl. exakter UTF-8 Byte-Zählung für BPB."""
    tag = f"{preset}_{tokenizer.name}_{tokenizer.vocab_size}_{train_tokens_target}_{val_tokens_target}"
    out_dir = cache_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"

    if meta_path.exists() and train_bin.exists() and val_bin.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    if logger:
        logger(f"[Data] Baue Korpus-Cache '{tag}'...")

    def _encode_split(split_name: str, target_toks: int, bin_path: Path) -> tuple[int, int, str]:
        max_docs = max(400, target_toks // 80)
        text_iter, actual_src = _iter_dataset_texts(preset, split_name, max_docs=max_docs, logger=logger)
        mmap_arr = np.memmap(bin_path, dtype=np.uint32, mode="w+", shape=(target_toks,))
        written = 0
        total_bytes = 0
        batch_texts: list[str] = []
        next_log = 10_000_000

        def _flush(buf_texts: list[str]) -> None:
            nonlocal written, total_bytes, next_log
            if not buf_texts or written >= target_toks:
                return
            for t in buf_texts:
                total_bytes += len(t.encode("utf-8"))
            encoded = tokenizer.encode_batch(buf_texts)
            flat: list[int] = []
            for ids in encoded:
                flat.extend(ids)
                flat.append(tokenizer.eos_id)
                if written + len(flat) >= target_toks:
                    break
            take = min(len(flat), target_toks - written)
            if take > 0:
                mmap_arr[written : written + take] = np.asarray(flat[:take], dtype=np.uint32)
                written += take
            if logger and written >= next_log:
                logger(f"[Data] {split_name}: {written:,}/{target_toks:,} Tokens gecacht...")
                next_log += 10_000_000

        for txt in text_iter:
            batch_texts.append(txt)
            if len(batch_texts) >= 256:
                _flush(batch_texts)
                batch_texts.clear()
                if written >= target_toks:
                    break
        if batch_texts and written < target_toks:
            _flush(batch_texts)

        mmap_arr.flush()
        del mmap_arr
        if written < target_toks:
            # Kürze Datei exakt auf geschriebene Token-Anzahl
            trimmed = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(written,))
            tmp_path = bin_path.with_suffix(".tmp")
            np.asarray(trimmed).tofile(tmp_path)
            del trimmed
            tmp_path.replace(bin_path)
        return int(written), int(total_bytes), actual_src

    n_train, train_bytes, src_train = _encode_split("train", train_tokens_target, train_bin)
    n_val, val_bytes, _ = _encode_split("validation", val_tokens_target, val_bin)

    meta = {
        "source": src_train,
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "train_tokens": n_train,
        "val_tokens": n_val,
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
        "bytes_per_token": float(val_bytes) / max(1, n_val),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if logger:
        logger(
            f"[Data] Cache fertig: train={n_train:,} toks, val={n_val:,} toks, "
            f"bytes/token={meta['bytes_per_token']:.3f}"
        )
    return meta


class PackedSequenceDataset(Dataset):
    """Memmap-basiertes Sequence-Packing-Dataset mit EoS-Document-IDs."""

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        eos_id: int,
        reasoning_samples: list[list[int]] | None = None,
        reasoning_mix_ratio: float = 0.0,
    ):
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.chunk_size = seq_len + 1
        self.num_chunks = max(1, (len(self.tokens) - 1) // self.chunk_size)
        self.reasoning_samples = reasoning_samples or []
        self.reasoning_mix_ratio = reasoning_mix_ratio

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.reasoning_samples
            and self.reasoning_mix_ratio > 0.0
            and ((idx % 100) / 100.0) < self.reasoning_mix_ratio
        ):
            seq = self.reasoning_samples[idx % len(self.reasoning_samples)]
            # Wiederhole/packe Reasoning-Beispiele auf chunk_size
            buf: list[int] = []
            pos = idx
            while len(buf) < self.chunk_size:
                s = self.reasoning_samples[pos % len(self.reasoning_samples)]
                buf.extend(s)
                buf.append(self.eos_id)
                pos += 1
            chunk = torch.tensor(buf[: self.chunk_size], dtype=torch.long)
        else:
            start = (idx % self.num_chunks) * self.chunk_size
            arr = self.tokens[start : start + self.chunk_size].astype(np.int64)
            if arr.size < self.chunk_size:
                pad = np.full(self.chunk_size - arr.size, self.eos_id, dtype=np.int64)
                arr = np.concatenate([arr, pad])
            chunk = torch.from_numpy(arr)

        x = chunk[:-1]
        y = chunk[1:]
        # Dokument-ID erhöht sich nach jedem EoS-Token -> sauberes Document Masking
        is_eos = (x == self.eos_id).to(torch.int32)
        doc_ids = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int32), is_eos[:-1]]), dim=0)
        return x, y, doc_ids


class JESTBatchSelector:
    """Self-Bootstrapped JEST: Wählt aus Super-Batch die lernfähigsten Sub-Batches."""

    def __init__(self, ema_baseline: float | None = None):
        self.ema_loss = ema_baseline

    @torch.no_grad()
    def select_batch(
        self,
        model: torch.nn.Module,
        super_x: torch.Tensor,
        super_y: torch.Tensor,
        super_doc: torch.Tensor,
        target_batch_size: int,
        use_doc_masking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if super_x.shape[0] <= target_batch_size:
            return super_x, super_y, super_doc

        was_training = model.training
        model.eval()
        out = model(super_x, doc_ids=super_doc if use_doc_masking else None)
        h = out.hidden_states
        # Schneller Subsample-Score (jedes 4. Token) für minimalen Overhead
        h_sub = h[:, ::4, :]
        y_sub = super_y[:, ::4]
        logits = F.linear(h_sub, model.lm_head.weight.to(h_sub.dtype)).float()
        per_seq = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_sub.reshape(-1), reduction="none")
        seq_losses = per_seq.view(super_x.shape[0], -1).mean(dim=1)

        if self.ema_loss is None:
            self.ema_loss = float(seq_losses.mean().item())
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * float(seq_losses.mean().item())

        # Bevorzuge Sequenzen nahe dem optimalen Lern-Fenster (über EMA, aber keine reinen Noise-Ausreißer)
        learnability = (seq_losses - self.ema_loss).clamp(min=-2.0, max=2.0)
        _, top_idx = torch.topk(learnability, target_batch_size, sorted=False)
        if was_training:
            model.train()
        return super_x[top_idx], super_y[top_idx], super_doc[top_idx]
