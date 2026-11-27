"""
Vergleicht 4 Modelle über 2k Steps auf TinyStories (data.py & eval_data.py):
  - baseline:   Transformer aus Transformer/model.py
  - dif_legacy: ARDiffusionBlocksModel mit dichtem SwiGLU aus DiffusionBlocks/Dif_trans-Leg.py
  - dif_moe:    ARDiffusionBlocksModel mit LatentMoE aus DiffusionBlocks/Dif_trans.py
  - dif_looped: LoopedARDiffusionModel aus DiffusionBlocks/Looped_Dif.py

Ausführen:
    python3 compare_models.py
    python3 compare_models.py --steps 2000 --models baseline dif_moe dif_looped
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import importlib.util
import math
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ARConfig, ModelConfig
from data import CustomDataset
from eval_data import EvalDataset
from DiffusionBlocks.Dif_ar import ARDiffusionBlocksModel
from DiffusionBlocks.Looped_Dif import LoopedARConfig, LoopedARDiffusionModel
from Transformer.model import Transformer


PROMPTS = [
    "Once upon a time, there was a little ",
    "One day, Lily found a ",
    "Hi, how are you?",
]


def _load_legacy_diffusion_block():
    path = Path(__file__).resolve().parent / "DiffusionBlocks" / "Dif_trans-Leg.py"
    spec = importlib.util.spec_from_file_location("Dif_trans_leg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DiffusionBlock


def build_model(name: str, m_cfg: ModelConfig, vocab_size: int, device: torch.device) -> tuple[nn.Module, dict]:
    common_ar = dict(
        vocab_size=vocab_size,
        max_seq_len=m_cfg.seq_len,
        dim=m_cfg.d_model,
        num_heads=m_cfg.n_heads,
        num_layers=m_cfg.n_layers,
        num_blocks=3,
        rms_norm_eps=m_cfg.rms_norm_eps,
        rope_base=m_cfg.rope_base,
        tokenizer_name=m_cfg.tokenizer_name,
    )
    if name == "baseline":
        model = Transformer(m_cfg).to(device)
        return model, asdict(m_cfg)
    if name == "dif_legacy":
        LegacyBlock = _load_legacy_diffusion_block()
        cfg = ARConfig(**common_ar)
        model = ARDiffusionBlocksModel(cfg).to(device)
        layers_per_block = cfg.num_layers // cfg.num_blocks
        model.blocks = nn.ModuleList(
            [
                LegacyBlock(layers_per_block, cfg.dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout, cfg.rms_norm_eps)
                for _ in range(cfg.num_blocks)
            ]
        ).to(device)
        model._init_weights()
        return model, asdict(cfg)
    if name == "dif_moe":
        cfg = ARConfig(**common_ar)
        return ARDiffusionBlocksModel(cfg).to(device), asdict(cfg)
    if name == "dif_looped":
        cfg = LoopedARConfig(
            vocab_size=vocab_size,
            max_seq_len=m_cfg.seq_len,
            dim=m_cfg.d_model,
            num_heads=m_cfg.n_heads,
            prelude_layers=1,
            core_layers=4,
            coda_layers=1,
            rms_norm_eps=m_cfg.rms_norm_eps,
            rope_base=m_cfg.rope_base,
            inference_steps=8,
        )
        return LoopedARDiffusionModel(cfg).to(device), asdict(cfg)
    raise ValueError(f"Unbekanntes Modell: {name}")


def _batch_to_tokens(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y[:, -1:]], dim=1)


def train_model(
    name: str,
    model: nn.Module,
    cfg_dict: dict,
    train_loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[float, str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    log_every = max(1, steps // 4)
    it = iter(train_loader)
    is_multiblock = hasattr(model, "config") and hasattr(model.config, "num_blocks")
    block_ce = {b: [] for b in range(model.config.num_blocks)} if is_multiblock else None
    ce_buf = []

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            if name == "baseline":
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                ce_buf.append(float(loss.item()))
            else:
                loss, info = model.training_loss(_batch_to_tokens(x, y))
                if block_ce is not None:
                    block_ce[info["block_idx"]].append(info["ce_loss"])
                else:
                    ce_buf.append(info["ce_loss"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == steps:
            if block_ce is not None:
                summary = " | ".join(
                    f"B{b}:{sum(v)/len(v):.3f}" if v else f"B{b}:-"
                    for b, v in block_ce.items()
                )
                print(f"  [{name}] Step {step:4d}/{steps} | Ø Train CE -> {summary}")
                block_ce = {b: [] for b in range(model.config.num_blocks)}
            else:
                print(f"  [{name}] Step {step:4d}/{steps} | Ø Train CE: {sum(ce_buf)/len(ce_buf):.3f}")
                ce_buf.clear()

    elapsed = time.time() - t0
    os.makedirs("checkpoints", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    ckpt_path = os.path.join("checkpoints", f"{name}_{ts}.pth")
    torch.save({"model_state": model.state_dict(), "config": cfg_dict, "steps": steps}, ckpt_path)
    return elapsed, ckpt_path


@torch.no_grad()
def eval_ppl(
    name: str,
    model: nn.Module,
    eval_loader: DataLoader,
    eval_batches: int,
    device: torch.device,
) -> tuple[float, float, str]:
    torch.manual_seed(0)
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    ces, extra_ces = [], []

    for i, (x, y) in enumerate(eval_loader):
        if i >= eval_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            if name == "baseline":
                logits = model(x)
                ces.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()))
            elif name in ("dif_legacy", "dif_moe"):
                toks = _batch_to_tokens(x, y)
                block_ces = [model.training_loss(toks, block_idx=b)[1]["ce_loss"] for b in range(model.config.num_blocks)]
                ces.append(block_ces[0])
                extra_ces.append(sum(block_ces) / len(block_ces))
            else:
                toks = _batch_to_tokens(x, y)
                _, info = model.training_loss(toks)
                ces.append(info["ce_loss"])
                x_ids, y_ids = toks[:, :-1], toks[:, 1:]
                sigma_hi = torch.full((x_ids.size(0),), model.config.sigma_max, device=device)
                z_hi = model.get_normalized_embeddings(y_ids) + sigma_hi[:, None, None] * torch.randn(
                    x_ids.size(0), x_ids.size(1), model.config.dim, device=device
                )
                logits_hi = model.denoise_train(x_ids, z_hi, sigma_hi)
                extra_ces.append(float(F.cross_entropy(logits_hi.reshape(-1, logits_hi.size(-1)), y_ids.reshape(-1)).item()))

    val_ce = sum(ces) / max(1, len(ces))
    ppl = math.exp(min(val_ce, 20.0))
    if name in ("dif_legacy", "dif_moe"):
        extra = f"Mean CE={sum(extra_ces)/max(1, len(extra_ces)):.3f}"
    elif extra_ces:
        extra = f"σ_max CE={sum(extra_ces)/len(extra_ces):.3f}"
    else:
        extra = "-"
    return val_ce, ppl, extra


@torch.no_grad()
def generate_samples(
    name: str,
    model: nn.Module,
    tokenizer,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> None:
    model.eval()
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    print(f"\n--- Samples: {name} ---")
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if name != "baseline":
            ids = [bos_id] + ids
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        if name == "baseline":
            out = inp.clone()
            for _ in range(max_new_tokens):
                logits = model(out[:, -max_seq_len:])
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                out = torch.cat([out, next_tok], dim=1)
            gen_ids = out[0, len(ids) :].tolist()
        else:
            out = model.generate(inp, max_new_tokens=max_new_tokens, temperature=0.0)
            gen_ids = out[0, len(ids) :].tolist()

        print(f"  Prompt: {prompt!r}\n  Output: {tokenizer.decode(gen_ids, skip_special_tokens=True)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modell-Vergleich auf TinyStories (2k Steps)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "dif_legacy", "dif_moe", "dif_looped"],
        choices=["baseline", "dif_legacy", "dif_moe", "dif_looped"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = ModelConfig(batch_size=args.batch_size, max_samples=args.max_samples)
    eval_cfg = ModelConfig(batch_size=args.batch_size, max_samples=args.eval_samples)

    train_ds = CustomDataset(train_cfg)
    eval_ds = EvalDataset(eval_cfg)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=(len(train_ds) >= args.batch_size)
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    vocab_size = len(train_ds.tokenizer)
    train_cfg.vocab_size = vocab_size

    results = []
    for name in args.models:
        torch.manual_seed(42)
        model, cfg_dict = build_model(name, train_cfg, vocab_size, device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        num_blocks = model.config.num_blocks if (hasattr(model, "config") and hasattr(model.config, "num_blocks")) else 1
        model_steps = args.steps * num_blocks
        step_note = f"{model_steps} Steps ({args.steps}/Block)" if num_blocks > 1 else f"{model_steps} Steps"
        print(f"\n=== Training {name} ({params:.2f}M Params, {step_note}, Device={device}) ===")

        elapsed, ckpt_path = train_model(name, model, cfg_dict, train_loader, model_steps, args.lr, device)
        val_ce, val_ppl, extra = eval_ppl(name, model, eval_loader, args.eval_batches, device)
        generate_samples(name, model, train_ds.tokenizer, train_cfg.seq_len, args.max_new_tokens, device)
        results.append((name, params, elapsed, val_ce, val_ppl, extra, ckpt_path))

    print("\n=== Vergleich (Zusammenfassung) ===")
    print(f"{'Modell':<12} | {'Params':>7} | {'Zeit':>6} | {'Val CE':>7} | {'Val PPL':>9} | {'Info':<14} | Checkpoint")
    print("-" * 95)
    for name, params, elapsed, val_ce, val_ppl, extra, ckpt_path in results:
        print(f"{name:<12} | {params:6.2f}M | {elapsed:5.1f}s | {val_ce:7.3f} | {val_ppl:9.2f} | {extra:<14} | {ckpt_path}")


if __name__ == "__main__":
    main()
