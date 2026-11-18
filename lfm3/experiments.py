"""Runnable experiments for the LFM3 lab.

Every function takes a :class:`~lfm3.loader.ModelBundle` and prints a compact report.
They are written so they work both with the real 1.7B checkpoint and with the tiny
random model (``lfm3.config.TinyLFM3``), which makes them usable as smoke tests.

Run them from the CLI::

    python lfm3_lab.py demo --all                 # real checkpoint
    python lfm3_lab.py demo --all --tiny          # seconds, no download

or from Python::

    from lfm3.loader import load_base
    from lfm3.experiments import demo_hidden_states

    bundle = load_base()
    demo_hidden_states(bundle)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from .config import LoadOptions, TinyLFM3
from .modules import (
    BottleneckAdapter,
    BottleneckConfig,
    LoRAConfig,
    NgramConfig,
    inject_lora,
    mark_only_lora_trainable,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Once upon a time"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text_config(bundle: Any) -> Any:
    """The text sub-config (omni nests it, the base model is the config itself)."""
    return getattr(bundle.config, "text_config", bundle.config)


def _vocab_size(bundle: Any) -> int:
    return _text_config(bundle).vocab_size


def _hidden_size(bundle: Any) -> int:
    return _text_config(bundle).hidden_size


def _is_tiny(bundle: Any) -> bool:
    """True when the tokenizer cannot be used (random tiny model)."""
    if bundle.tokenizer is None:
        return True
    vocab = _vocab_size(bundle)
    return len(bundle.tokenizer) > vocab


def sample_input_ids(bundle: Any, prompt: str | None = None, seq_len: int = 16) -> torch.Tensor:
    """Tokenize ``prompt`` if possible, otherwise draw random ids inside the model vocab."""
    vocab = _vocab_size(bundle)
    if prompt and not _is_tiny(bundle):
        ids = bundle.tokenizer(prompt, return_tensors="pt").input_ids
    else:
        ids = torch.randint(0, vocab, (1, seq_len))
    return ids.to(bundle.device)


def _top_token(logits: torch.Tensor, tokenizer: Any) -> str:
    token_id = int(logits.argmax(-1))
    if tokenizer is not None:
        try:
            return tokenizer.decode([token_id])
        except Exception:
            pass
    return f"id={token_id}"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def _kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Mean KL(a || b) over positions, in nats."""
    log_p = F.log_softmax(logits_a.float(), dim=-1)
    log_q = F.log_softmax(logits_b.float(), dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum(-1).mean())


# ---------------------------------------------------------------------------
# 1. overview
# ---------------------------------------------------------------------------


def demo_overview(bundle: Any, prompt: str = DEFAULT_PROMPT) -> None:
    """Print the architecture report, then run one forward pass."""
    from .inspect import describe

    print(describe(bundle))

    input_ids = sample_input_ids(bundle, prompt)
    with torch.no_grad():
        out = bundle(input_ids)
    print()
    print(f"input tokens   : {input_ids.shape[-1]}")
    print(f"logits         : {tuple(out.logits.shape)}  (batch, seq, vocab)")
    print(f"next token     : {_top_token(out.logits[0, -1], bundle.tokenizer)!r}")


# ---------------------------------------------------------------------------
# 2. hidden states through the stack
# ---------------------------------------------------------------------------


def demo_hidden_states(bundle: Any, prompt: str = DEFAULT_PROMPT) -> None:
    """Per-block activation statistics and layer-to-layer similarity."""
    input_ids = sample_input_ids(bundle, prompt)
    acc = bundle.access

    with torch.no_grad():
        stages = acc.forward_stages(input_ids)

    names = [n for n in stages if n.startswith("layers.")]
    print("hidden states through the stack")
    print(f"  {'stage':12s}{'mean':>10s}{'std':>10s}{'norm':>10s}{'cos(prev)':>11s}")
    prev = None
    for name in names:
        h = stages[name][0].float()
        cos = _cosine(prev, h) if prev is not None else float("nan")
        print(f"  {name:12s}{h.mean():10.4f}{h.std():10.4f}{h.norm():10.2f}{cos:11.4f}")
        prev = h
    final = stages["final"][0].float()
    cos = _cosine(prev, final)
    print(f"  {'final':12s}{final.mean():10.4f}{final.std():10.4f}{final.norm():10.2f}{cos:11.4f}")
    print(f"  embeddings    : {tuple(stages['embeddings'].shape)}")

    # where does the probability mass sit?
    probs = F.softmax(stages["logits"][0, -1].float(), dim=-1)
    top = probs.topk(5)
    print("  top-5 next-token probabilities:", [f"{p:.3f}" for p in top.values.tolist()])


# ---------------------------------------------------------------------------
# 3. attach a small model in front of the output embedding
# ---------------------------------------------------------------------------


def demo_adapter(bundle: Any, steps: int = 30, bottleneck: int = 256) -> None:
    """Attach a bottleneck adapter, then distil the original model into it.

    The adapter starts as the identity (zero-init output projection), so the model is
    unchanged at step 0.  Training it to reproduce the *original* logits shows how much
    a small module in front of the LM head can recover -- and gives a template for
    attaching your own probe.
    """
    acc = bundle.access
    input_ids = sample_input_ids(bundle, DEFAULT_PROMPT)

    with torch.no_grad():
        reference_logits = acc.forward_logits(input_ids)

    adapter = BottleneckAdapter(
        _hidden_size(bundle), BottleneckConfig(bottleneck=bottleneck, layers=2, residual=True)
    ).to(bundle.device, bundle.dtype)

    # a fresh adapter is the identity -> logits must match exactly
    acc.attach_after_final_norm(adapter)
    with torch.no_grad():
        before = acc.forward_logits(input_ids)
    delta = (before.float() - reference_logits.float()).abs().max()
    print(f"attached adapter: {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M params")
    print(f"max |logit delta| at init: {float(delta):.3e} (0 = perfect identity)")

    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    for step in range(steps):
        logits = acc.forward_logits(input_ids)
        loss = F.kl_div(
            F.log_softmax(logits.float(), -1),
            F.log_softmax(reference_logits.float(), -1),
            log_target=True,
            reduction="batchmean",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"  step {step:3d}  KL(loss) {float(loss.detach()):.6f}")

    with torch.no_grad():
        after = acc.forward_logits(input_ids)
    print(f"cosine(before, after) : {_cosine(before, after):.5f}")
    print(f"max |logit delta|     : {float((after.float()-reference_logits.float()).abs().max()):.3e}")

    # restore the pristine head so later demos are unaffected
    if hasattr(acc.model, "lm_head") and hasattr(acc.model.lm_head, "head"):
        acc.model.lm_head = acc.model.lm_head.head


# ---------------------------------------------------------------------------
# 4. n-gram layer
# ---------------------------------------------------------------------------


def demo_ngram(bundle: Any, n: int = 3, steps: int = 30) -> None:
    """Attach a hashed n-gram bias and train it on a repeated sentence."""
    from .modules import NgramLayer

    acc = bundle.access
    hidden_size = _hidden_size(bundle)
    vocab = _vocab_size(bundle)
    cfg = NgramConfig(n=n, num_buckets=4096, rank=32, mode="logit_bias", vocab_size=vocab, hidden_size=hidden_size)

    if _is_tiny(bundle):
        base_ids = torch.randint(0, vocab, (1, 24), device=bundle.device)
        ngram = NgramLayer(cfg).to(bundle.device, bundle.dtype)
    else:
        text = (DEFAULT_PROMPT + " ") * 6
        base_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
        ngram = NgramLayer(cfg).to(bundle.device, bundle.dtype)

    with torch.no_grad():
        reference = acc.forward_logits(base_ids)

    # at init the output projection is zero -> logits unchanged
    with torch.no_grad():
        fresh = acc.forward_with_ngram(base_ids, ngram)
    print(f"n-gram layer: n={n}, buckets={cfg.num_buckets}, rank={cfg.rank}, params="
          f"{sum(p.numel() for p in ngram.parameters())/1e6:.2f}M")
    print(f"max |logit delta| at init: {float((fresh.float()-reference.float()).abs().max()):.3e}")

    # train it to predict the *residual* the base model leaves: a toy task where the
    # n-gram bias should memorise frequent continuations
    targets = base_ids.roll(-1, dims=1)
    optimizer = torch.optim.Adam(ngram.parameters(), lr=5e-3)
    for step in range(steps):
        logits = acc.forward_with_ngram(base_ids, ngram)
        loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {float(loss.detach()):.4f}")

    with torch.no_grad():
        trained = acc.forward_with_ngram(base_ids, ngram)
    print(f"top-1 agreement base vs +ngram: {float((trained.argmax(-1)==reference.argmax(-1)).float().mean()):.3f}")
    print(f"max |logit delta| after training: {float((trained.float()-reference.float()).abs().max()):.3e}")


# ---------------------------------------------------------------------------
# 5. block ablations
# ---------------------------------------------------------------------------


def demo_block_ablation(bundle: Any, prompt: str = DEFAULT_PROMPT) -> None:
    """Bypass every block once and measure how much the next-token distribution moves."""
    acc = bundle.access
    input_ids = sample_input_ids(bundle, prompt)
    with torch.no_grad():
        reference = acc.forward_logits(input_ids)
    ref_dist = F.log_softmax(reference.float(), -1)

    print("block ablation (identity bypass), KL(base || bypassed) in nats")
    print(f"  {'block':>5s}  {'type':18s}{'KL':>10s}{'top-1 changed':>15s}")
    for i in range(acc.num_blocks):
        original = acc.blocks[i]
        acc.bypass_block(i, mode="identity")
        with torch.no_grad():
            out = acc.forward_logits(input_ids)
        kl = _kl_divergence(reference, out)
        changed = float((out.argmax(-1) != reference.argmax(-1)).float().mean())
        print(f"  {i:5d}  {acc.layer_type(i):18s}{kl:10.4f}{changed:15.3f}")
        acc.blocks[i] = original


# ---------------------------------------------------------------------------
# 6. embeddings
# ---------------------------------------------------------------------------


def demo_embeddings(bundle: Any) -> None:
    """Inspect the input embedding matrix and try two surgical edits."""
    acc = bundle.access
    report = acc.embedding_report()
    print("embedding report")
    for key, value in report.items():
        print(f"  {key:16s}: {value}")

    weight = acc.embed_tokens.weight
    with torch.no_grad():
        norms = weight.float().norm(dim=-1)
        top = norms.topk(5)
        print("  longest rows (token id, norm):", [(int(i), round(float(v), 2)) for v, i in zip(top.values, top.indices)])

    # edit 1: scale the whole matrix, edit 2: zero one row (restored afterwards)
    acc.scale_embeddings(1.05)
    with torch.no_grad():
        scaled = acc.embed_tokens.weight.float().norm(dim=-1).mean()
    print(f"  mean row norm after scale(1.05): {float(scaled):.4f}")
    acc.scale_embeddings(1 / 1.05)

    with torch.no_grad():
        saved = acc.embed_tokens.weight[0].clone()
        acc.zero_row(0)
        print(f"  norm of row 0 after zero_row: {float(acc.embed_tokens.weight[0].float().norm()):.4f}")
        acc.embed_tokens.weight[0].copy_(saved)


# ---------------------------------------------------------------------------
# 7. LoRA
# ---------------------------------------------------------------------------


def demo_lora(bundle: Any, steps: int = 20) -> None:
    """Inject LoRA into the attention projections and take a few training steps."""
    vocab = _vocab_size(bundle)
    acc = bundle.access

    report = inject_lora(bundle.model, LoRAConfig(rank=8, alpha=16.0, targets=("q_proj", "k_proj", "v_proj", "o_proj")))
    mark_only_lora_trainable(bundle.model)
    print("LoRA injected")
    print(f"  wrapped layers : {report['wrapped']}")
    print(f"  budget         : {acc.trainable_summary()}")

    if _is_tiny(bundle):
        input_ids = torch.randint(0, vocab, (2, 24), device=bundle.device)
    else:
        input_ids = bundle.tokenizer((DEFAULT_PROMPT + " ") * 4, return_tensors="pt").input_ids.to(bundle.device)
    labels = input_ids

    optimizer = torch.optim.AdamW([p for p in bundle.model.parameters() if p.requires_grad], lr=1e-3)
    bundle.model.train()
    for step in range(steps):
        logits = bundle.model(input_ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {float(loss.detach()):.4f}")


# ---------------------------------------------------------------------------
# 8. fine-tuning smoke test
# ---------------------------------------------------------------------------


def demo_finetune(bundle: Any, steps: int = 30) -> None:
    """A few optimizer steps of language modelling on a tiny corpus.

    On the real checkpoint this only LoRA-trains the attention (cheap); on the tiny
    model it trains everything.  The point is to show the plumbing, not convergence.
    """
    vocab = _vocab_size(bundle)
    if _is_tiny(bundle):
        input_ids = torch.randint(0, vocab, (1, 96), device=bundle.device)
    else:
        text = " ".join([DEFAULT_PROMPT + " the little dragon lived in a cave."] * 40)
        input_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)

    input_ids = input_ids.to(bundle.device)[:, :96]
    if input_ids.shape[1] < 8:
        input_ids = torch.randint(0, vocab, (1, 32), device=bundle.device)

    if not _is_tiny(bundle):
        inject_lora(bundle.model, LoRAConfig(rank=8, alpha=16.0))
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    bundle.model.train()
    print("fine-tuning smoke test")
    for step in range(steps):
        logits = bundle.model(input_ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), input_ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {float(loss.detach()):.4f}")


# ---------------------------------------------------------------------------
# 9. reference implementation parity
# ---------------------------------------------------------------------------


def demo_reference_parity(bundle: Any, prompt: str = DEFAULT_PROMPT, seq_len: int = 16) -> None:
    """Compare the pure-PyTorch blocks against the Hugging Face model.

    Two checks:

    1. per layer -- the same input is pushed through the HF block and through
       :class:`lfm3.pytorch_blocks.ReferenceDecoderLayer`; both run in the model dtype,
       so the difference is pure implementation difference.
    2. end to end -- the whole reference backbone vs the HF backbone.

    The end-to-end delta is larger than the per-layer one because bf16 rounding
    accumulates over 24 blocks; cast the model to fp32 (``bundle.to(dtype=torch.float32)``)
    if you need a bit-exact comparison.
    """
    from .pytorch_blocks import ReferenceDecoderLayer, ReferenceBackbone, _rope_tables

    acc = bundle.access
    cfg = acc.config          # text config (also correct for omni)
    input_ids = sample_input_ids(bundle, prompt, seq_len=seq_len)
    seq = input_ids.shape[1]

    print("per-layer parity (same input, model dtype)")
    print(f"  {'block':>5s}  {'type':18s}{'max |delta|':>14s}{'cosine':>10s}")
    with torch.no_grad():
        hidden = acc.embed(input_ids)
        position_ids = torch.arange(seq, device=hidden.device)[None]
        for i in range(acc.num_blocks):
            hf_layer = acc.blocks[i]
            layer_type = acc.layer_type(i)
            ref_layer = ReferenceDecoderLayer(cfg, i).to(hidden.device, hidden.dtype)
            ref_layer.load_state_dict(hf_layer.state_dict())

            if layer_type == "conv":
                hf_out = hf_layer(hidden)
                ref_out = ref_layer(hidden)
            else:
                rope = cfg.rope_parameters[layer_type]
                cos, sin = _rope_tables(
                    cfg.head_dim, rope["rope_theta"], rope.get("partial_rotary_factor", 1.0), position_ids
                )
                cos, sin = cos.to(hidden.dtype), sin.to(hidden.dtype)
                hf_out = hf_layer(
                    hidden,
                    position_embeddings=(cos, sin),
                    attention_mask=_causal_mask(
                        seq,
                        cfg.sliding_window if layer_type == "sliding_attention" else None,
                        hidden.device,
                        hidden.dtype,
                    ),
                )
                ref_out = ref_layer(hidden, cos, sin)

            delta = (hf_out.float() - ref_out.float()).abs()
            print(f"  {i:5d}  {layer_type:18s}{float(delta.max()):14.3e}{_cosine(hf_out, ref_out):10.6f}")
            hidden = hf_out

    with torch.no_grad():
        reference_hf = acc.forward_hidden(input_ids=input_ids).last_hidden_state
        reference_torch = ReferenceBackbone.from_hf(acc.backbone).to(bundle.device, bundle.dtype)
        ours = reference_torch.forward_hidden(input_ids=input_ids)

    diff = (reference_hf.float() - ours.float()).abs()
    print()
    print(f"end-to-end ({acc.num_blocks} blocks; larger deltas are rounding that accumulates)")
    print(f"  parameters        : hf={acc.num_parameters()/1e6:.2f}M  reference={reference_torch.parameter_count()/1e6:.2f}M")
    print(f"  max |delta|       : {float(diff.max()):.3e}")
    print(f"  mean |delta|      : {float(diff.mean()):.3e}")
    print(f"  cosine            : {_cosine(reference_hf.float(), ours.float()):.6f}")


def _causal_mask(seq: int, sliding: int | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q = torch.arange(seq, device=device)[:, None]
    k = torch.arange(seq, device=device)[None, :]
    allowed = k <= q
    if sliding is not None:
        allowed = allowed & (k > q - sliding)
    mask = torch.zeros(seq, seq, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


# ---------------------------------------------------------------------------
# 10. generation
# ---------------------------------------------------------------------------


def demo_generate(bundle: Any, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 32) -> None:
    """Sample a continuation (base models read better with sampling than greedily)."""
    if _is_tiny(bundle):
        vocab = _vocab_size(bundle)
        ids = torch.randint(0, vocab, (1, 8), device=bundle.device)
        print(f"tiny model: random prompt ids {ids.tolist()}")
    else:
        ids = bundle.tokenizer(prompt, return_tensors="pt").input_ids.to(bundle.device)
        print(f"prompt: {prompt!r}")

    out = bundle.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        use_cache=True,
    )
    if _is_tiny(bundle):
        print("generated ids:", out[0, ids.shape[1]:].tolist())
    else:
        print("continuation:", repr(bundle.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

DEMOS = {
    "overview": demo_overview,
    "hidden": demo_hidden_states,
    "adapter": demo_adapter,
    "ngram": demo_ngram,
    "ablation": demo_block_ablation,
    "embeddings": demo_embeddings,
    "lora": demo_lora,
    "finetune": demo_finetune,
    "parity": demo_reference_parity,
    "generate": demo_generate,
}


def run_all(bundle: Any, skip: tuple[str, ...] = ("finetune",)) -> None:
    """Run every demo in a fixed order.

    The order matters: ``parity`` and ``generate`` need the pristine model, ``lora``
    rewrites parameter names, so they run last.
    """
    order = ["overview", "hidden", "adapter", "ngram", "ablation", "embeddings", "parity", "generate", "lora", "finetune"]
    for name in order:
        if name in skip:
            continue
        print()
        print("=" * 78)
        print(f"### {name}")
        print("=" * 78)
        DEMOS[name](bundle)


def build_tiny_bundle() -> Any:
    """A tiny random bundle for tests (no checkpoint download)."""
    from .loader import load_base

    return load_base(TinyLFM3, LoadOptions(device="cpu", dtype="float32", attn_implementation="eager", random_init=True))


__all__ = ["DEMOS", "run_all", "build_tiny_bundle", "sample_input_ids"]