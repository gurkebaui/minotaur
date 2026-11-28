"""Tokenizer-Wrapper für Standard GPT-2 und 2-Stage SuperBPE (Cross-Whitespace Superwords)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer


class TokenizerWrapper:
    """Einheitliche Schnittstelle für HF-AutoTokenizer und SuperBPE-Tokenizer."""

    def __init__(self, backend, eos_id: int, vocab_size: int, name: str, is_hf: bool):
        self._backend = backend
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.name = name
        self._is_hf = is_hf

    def encode(self, text: str) -> list[int]:
        if self._is_hf:
            return self._backend.encode(text, add_special_tokens=False)
        return self._backend.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        if self._is_hf:
            return self._backend(texts, add_special_tokens=False)["input_ids"]
        return [enc.ids for enc in self._backend.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids)


def train_or_load_superbpe(
    texts_iter_fn,
    vocab_size: int,
    save_dir: Path,
    logger=None,
) -> TokenizerWrapper:
    """Trainiert einen 2-Stufen-SuperBPE-Tokenizer (Phase 1: Whitespace-PreSplit, Phase 2: Superwords)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    tok_path = save_dir / "tokenizer.json"
    meta_path = save_dir / "meta.json"

    if tok_path.exists() and meta_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return TokenizerWrapper(tok, eos_id=int(meta["eos_id"]), vocab_size=tok.get_vocab_size(), name="superbpe", is_hf=False)

    if logger:
        logger(f"[SuperBPE] Trainiere 2-Stage SuperBPE (vocab_size={vocab_size})...")

    # Phase 1: Standard ByteLevel Pre-Tokenizer (lernt reguläre Wort-Merges bis ~75% Vocab)
    phase1_vocab = max(512, int(vocab_size * 0.75))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer1 = trainers.BpeTrainer(
        vocab_size=phase1_vocab,
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    texts = list(texts_iter_fn())
    tok.train_from_iterator(texts, trainer=trainer1)

    # Phase 2: Pre-Tokenizer ohne Whitespace-Split -> lernt Cross-Whitespace Superwords!
    # Wir trainieren auf den Texten weiter, indem wir ByteLevel ohne regex-split verwenden
    tok2 = Tokenizer(models.BPE())
    tok2.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok2.decoder = decoders.ByteLevel()
    trainer2 = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    tok2.train_from_iterator(texts, trainer=trainer2)

    eos_id = tok2.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 0
    tok2.save(str(tok_path))
    meta_path.write_text(json.dumps({"eos_id": eos_id, "vocab_size": tok2.get_vocab_size()}), encoding="utf-8")

    return TokenizerWrapper(tok2, eos_id=eos_id, vocab_size=tok2.get_vocab_size(), name="superbpe", is_hf=False)


def get_tokenizer(
    use_superbpe: bool,
    superbpe_vocab_size: int,
    cache_dir: Path,
    texts_iter_fn=None,
    logger=None,
) -> TokenizerWrapper:
    if use_superbpe:
        assert texts_iter_fn is not None, "texts_iter_fn benötigt für SuperBPE-Training."
        return train_or_load_superbpe(
            texts_iter_fn=texts_iter_fn,
            vocab_size=superbpe_vocab_size,
            save_dir=cache_dir / f"superbpe_{superbpe_vocab_size}",
            logger=logger,
        )
    hf_tok = AutoTokenizer.from_pretrained("gpt2")
    eos_id = int(hf_tok.eos_token_id if hf_tok.eos_token_id is not None else 50256)
    return TokenizerWrapper(hf_tok, eos_id=eos_id, vocab_size=len(hf_tok), name="gpt2", is_hf=True)
