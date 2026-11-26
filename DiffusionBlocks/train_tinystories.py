"""
Trainings- und Generierungsskript für den AR-DiffusionBlocks-Transformer auf TinyStories.

Ausführen:
    python3 ar_diffusion_blocks/train_tinystories.py
    python3 ar_diffusion_blocks/train_tinystories.py --steps 8000
    python3 ar_diffusion_blocks/train_tinystories.py --generate-only --prompt "Once upon a time, a little cat "
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import itertools
import math
import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from datasets import load_dataset
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import torch

from config import ARConfig
from DiffusionBlocks.Dif_ar import ARDiffusionBlocksModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "tinystories_data")
DEFAULT_CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "tinystories_dblocks.pt")


def prepare_tinystories_data(
    data_dir: str,
    num_train_stories: int = 50_000,
    num_val_stories: int = 1_000,
    vocab_size: int = 4096,
) -> tuple[Tokenizer, np.ndarray, np.ndarray]:
    """
    Lädt einen Teil von TinyStories per Streaming, trainiert einen kompakten BPE-Tokenizer (4096)
    und speichert die tokenisierten Streams als uint16-NumPy-Dateien im Cache.
    """
    os.makedirs(data_dir, exist_ok=True)
    tok_path = os.path.join(data_dir, "tokenizer.json")
    train_path = os.path.join(data_dir, "train_tokens.npy")
    val_path = os.path.join(data_dir, "val_tokens.npy")

    if os.path.exists(tok_path) and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Lade gecachte Daten aus {data_dir} ...")
        tokenizer = Tokenizer.from_file(tok_path)
        train_tokens = np.load(train_path, mmap_mode="r")
        val_tokens = np.load(val_path, mmap_mode="r")
        return tokenizer, train_tokens, val_tokens

    print(f"Streame {num_train_stories:,} Train- & {num_val_stories:,} Val-Stories von TinyStories ...")
    t0 = time.time()
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    train_texts = [ex["text"] for ex in itertools.islice(train_ds, num_train_stories)]
    val_texts = [ex["text"] for ex in itertools.islice(val_ds, num_val_stories)]

    print("Trainiere ByteLevel-BPE Tokenizer ...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>", "<unk>"],
        show_progress=False,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.save(tok_path)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    def encode_corpus(texts: list[str]) -> np.ndarray:
        chunks = []
        for enc in tokenizer.encode_batch(texts):
            chunks.append(np.array([bos_id] + enc.ids + [eos_id], dtype=np.uint16))
        return np.concatenate(chunks)

    train_tokens = encode_corpus(train_texts)
    val_tokens = encode_corpus(val_texts)
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    print(
        f"Daten bereit in {time.time() - t0:.1f}s | "
        f"Train-Tokens: {len(train_tokens):,} | Val-Tokens: {len(val_tokens):,}"
    )
    return tokenizer, train_tokens, val_tokens


def sample_batch(
    tokens_np: np.ndarray, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Zieht zufällige Sequenz-Fenster der Länge `seq_len + 1` aus dem Token-Stream."""
    max_start = len(tokens_np) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    batch = np.stack([tokens_np[s : s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(batch).to(device)


@torch.no_grad()
def evaluate_per_block(
    model: ARDiffusionBlocksModel,
    val_tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    batches_per_block: int = 8,
) -> dict[int, float]:
    """Berechnet den mittleren Validierungs-CE-Loss getrennt für jeden der B Blöcke."""
    was_training = model.training
    model.eval()
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    results = {}
    for b in range(model.config.num_blocks):
        losses = []
        for _ in range(batches_per_block):
            batch = sample_batch(val_tokens, batch_size, seq_len, device)
            with amp_ctx:
                _, info = model.training_loss(batch, block_idx=b)
            losses.append(info["ce_loss"])
        results[b] = sum(losses) / len(losses)

    if was_training:
        model.train()
    return results


def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    """Linearer Warmup gefolgt von Cosine-Decay bis min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (max_lr - min_lr) * cosine


def generate_story(
    model: ARDiffusionBlocksModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    num_steps: int = 4,
    temperature: float = 0.0,
) -> str:
    """Generiert eine Geschichte ab `prompt` und stoppt sauber bei <eos>."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    prompt_ids = [bos_id] + tokenizer.encode(prompt).ids
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        out_ids = model.generate(
            inp, max_new_tokens=max_new_tokens, num_steps=num_steps, temperature=temperature
        )[0].tolist()

    # Alles nach <bos> bis zum ersten <eos> dekodieren
    gen_ids = out_ids[1:]
    if eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids)


def train(args: argparse.Namespace, device: torch.device) -> None:
    """Trainiert das AR-DiffusionBlocks-Modell auf TinyStories und speichert den Checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)

    tokenizer, train_tokens, val_tokens = prepare_tinystories_data(
        args.data_dir,
        num_train_stories=args.num_stories,
        vocab_size=args.vocab_size,
    )

    cfg = ARConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=args.seq_len,
        dim=384,
        num_heads=6,
        num_layers=8,
        num_blocks=4,
        dropout=0.0,
        gamma=0.1,
    )
    model = ARDiffusionBlocksModel(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Modell: {num_params / 1e6:.2f}M Parameter | "
        f"{cfg.num_layers} Layer in {cfg.num_blocks} Blöcken | Device: {device}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    warmup_steps = max(10, int(0.05 * args.steps))
    min_lr = args.lr * 0.1
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_every = max(50, args.steps // 10)
    val_every = max(100, args.steps // 4)
    block_ce = {b: [] for b in range(cfg.num_blocks)}
    t_start = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        lr = get_lr(step - 1, args.steps, warmup_steps, args.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = sample_batch(train_tokens, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            loss, info = model.training_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        block_ce[info["block_idx"]].append(info["ce_loss"])

        if step % log_every == 0 or step == args.steps:
            elapsed = time.time() - t_start
            sps = step / elapsed
            train_str = " | ".join(
                f"B{b}:{sum(v)/len(v):.2f}" if v else f"B{b}:-"
                for b, v in block_ce.items()
            )
            print(f"Step {step:5d}/{args.steps} ({sps:4.1f} it/s, lr={lr:.1e}) | Train CE -> {train_str}")
            block_ce = {b: [] for b in range(cfg.num_blocks)}

        if step % val_every == 0 or step == args.steps:
            val_res = evaluate_per_block(model, val_tokens, args.batch_size, args.seq_len, device)
            val_str = " | ".join(f"B{b}:{v:.2f}" for b, v in val_res.items())
            print(f"  [Val CE] {val_str}")

        if step == args.steps // 2:
            preview = generate_story(
                model, tokenizer, "Once upon a time, there was a little ", device, max_new_tokens=60
            )
            print(f"  [Zwischen-Sample] {preview}\n")

    total_time = time.time() - t_start
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"\nTraining fertig in {total_time:.1f}s | Peak-VRAM: {peak_vram:.0f} MiB")
    else:
        print(f"\nTraining fertig in {total_time:.1f}s")

    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, args.ckpt_path)
    print(f"Checkpoint gespeichert: {args.ckpt_path}")

    prompts = [
        ("Once upon a time, there was a little ", 0.0),
        ("One day, Lily found a ", 0.0),
        ("Tim and his dog went to the park. ", 0.7),
    ]
    print("\n=== Generierte TinyStories ===")
    for prompt, temp in prompts:
        story = generate_story(
            model, tokenizer, prompt, device, max_new_tokens=120, num_steps=4, temperature=temp
        )
        print(f"\n[Prompt (temp={temp})]: {prompt!r}")
        print(story)


def generate_from_checkpoint(args: argparse.Namespace, device: torch.device) -> None:
    """Lädt einen gespeicherten Checkpoint und generiert Text zu `args.prompt`."""
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    cfg = ARConfig(**ckpt["config"])
    model = ARDiffusionBlocksModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    story = generate_story(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=140,
        num_steps=4,
        temperature=args.temperature,
    )
    print(f"\n[Prompt (temp={args.temperature})]: {args.prompt!r}")
    print(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyStories Training mit AR DiffusionBlocks")
    parser.add_argument("--steps", type=int, default=8000, help="Anzahl Trainingsschritte")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch-Größe")
    parser.add_argument("--seq-len", type=int, default=192, help="Kontextlänge in Tokens")
    parser.add_argument("--lr", type=float, default=6e-4, help="Maximale Lernrate")
    parser.add_argument("--num-stories", type=int, default=50_000, help="Anzahl Trainings-Stories")
    parser.add_argument("--vocab-size", type=int, default=4096, help="BPE-Vokabulargröße")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt-path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--generate-only", action="store_true", help="Nur Checkpoint laden & generieren")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little ")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.generate_only:
        generate_from_checkpoint(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
