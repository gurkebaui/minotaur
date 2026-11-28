"""Synthetische Reasoning-Probes (Exp 5 & 6): Multi-Step Arithmetic, Parity, Nested Brackets."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from minotaur_exp.data.tokenizer import TokenizerWrapper


def _make_probe_pair(kind: str, rng: random.Random) -> tuple[str, str]:
    if kind == "arithmetic":
        vals = [rng.randint(1, 20) for _ in range(4)]
        ops = [rng.choice(["+", "-"]) for _ in range(3)]
        total = vals[0]
        expr = str(vals[0])
        for v, op in zip(vals[1:], ops):
            expr += f" {op} {v}"
            total = total + v if op == "+" else total - v
        return f"CALC: {expr} =", f" {total}"

    if kind == "parity":
        bits = [rng.randint(0, 1) for _ in range(6)]
        par = sum(bits) % 2
        seq = " ".join(str(b) for b in bits)
        return f"PARITY: {seq} ->", f" {par}"

    # nested brackets
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    depth = rng.randint(2, 4)
    stack = [rng.choice(pairs) for _ in range(depth)]
    opens = " ".join(p[0] for p in stack)
    closes = " ".join(p[1] for p in reversed(stack[:-1]))
    ans = f" {stack[0][1]}"
    prompt = f"BRACKETS: {opens} {closes} ->".replace("  ", " ")
    return prompt, ans


def generate_training_samples(
    tokenizer: TokenizerWrapper,
    n_samples: int = 200,
    seed: int = 1234,
) -> list[list[int]]:
    """Erzeugt Training-Samples für optionales Reasoning-Mix in Exp 5 & 6."""
    rng = random.Random(seed)
    kinds = ["arithmetic", "parity", "brackets"]
    out: list[list[int]] = []
    for i in range(n_samples):
        prompt, ans = _make_probe_pair(kinds[i % len(kinds)], rng)
        out.append(tokenizer.encode(prompt + ans))
    return out


@torch.no_grad()
def evaluate_reasoning_probes(
    model: torch.nn.Module,
    tokenizer: TokenizerWrapper,
    device: torch.device,
    eval_loops: list[int] | None = None,
    n_per_task: int = 24,
    seed: int = 98765,
) -> dict[str, dict[str, float]]:
    """Evaluiert Arithmetic, Parity & Brackets bei verschiedenen Loop-Tiefen R in {1, 2, 4, 8}."""
    was_training = model.training
    model.eval()
    rng = random.Random(seed)
    kinds = ["arithmetic", "parity", "brackets"]

    probes: list[tuple[str, list[int], int]] = []
    for kind in kinds:
        for _ in range(n_per_task):
            prompt, ans = _make_probe_pair(kind, rng)
            p_ids = tokenizer.encode(prompt)
            a_ids = tokenizer.encode(ans)
            if not p_ids or not a_ids:
                continue
            probes.append((kind, p_ids, a_ids[0]))

    loops_to_test = eval_loops if getattr(model, "is_looped", False) and eval_loops else [1]
    results: dict[str, dict[str, float]] = {}

    for r in loops_to_test:
        correct_by_kind = {k: 0 for k in kinds}
        total_by_kind = {k: 0 for k in kinds}
        nll_sum = 0.0
        n_total = 0

        for kind, p_ids, target_id in probes:
            x = torch.tensor([p_ids], dtype=torch.long, device=device)
            if getattr(model, "is_looped", False):
                out = model(x, num_loops=r)
            else:
                out = model(x)
            h_last = out.hidden_states[:, -1, :]
            logits = F.linear(h_last, model.lm_head.weight.to(h_last.dtype)).float()
            pred = int(logits.argmax(dim=-1).item())
            nll = float(F.cross_entropy(logits, torch.tensor([target_id], device=device)).item())

            total_by_kind[kind] += 1
            if pred == target_id:
                correct_by_kind[kind] += 1
            nll_sum += nll
            n_total += 1

        total_acc = sum(correct_by_kind.values()) / max(1, n_total)
        results[f"R={r}"] = {
            "overall_acc": total_acc,
            "avg_nll": nll_sum / max(1, n_total),
            "arithmetic_acc": correct_by_kind["arithmetic"] / max(1, total_by_kind["arithmetic"]),
            "parity_acc": correct_by_kind["parity"] / max(1, total_by_kind["parity"]),
            "brackets_acc": correct_by_kind["brackets"] / max(1, total_by_kind["brackets"]),
        }

    if was_training:
        model.train()
    return results
