"""Structural access and surgery for a loaded LFM3 model.

The Hugging Face ``Lfm3ForCausalLM`` already exposes its submodules, but it does so
through long attribute chains that differ between the base and the omni checkpoint, and
it does not offer the small helpers a research loop needs (hooks, parameter tables,
swapping the LM head, bypassing a block, freezing layers).

:class:`ModelAccess` is that missing layer.  It is a *view*, not a wrapper: every
helper reads and mutates the very modules of ``bundle.model``, so anything you change
here is what the model uses on the next forward call.

Typical session::

    bundle = load_base()
    acc = bundle.access

    acc.parameter_table()                 # where do the 1.73B parameters live?
    with acc.capture() as acts:           # record activations
        bundle(input_ids)
    acts.last("layers.5")                 # hidden state after block 5

    acc.bypass_block(7)                   # ablate block 7
    acc.attach_after_final_norm(adapter)  # small model in front of the LM head
    acc.replace_output_head(my_head)      # custom output embedding
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# activation capture
# ---------------------------------------------------------------------------


class Activations(dict):
    """Mapping ``name -> list[Tensor]`` collected by :meth:`ModelAccess.capture`.

    One list entry per forward call, so generation loops accumulate instead of
    overwriting.  ``last(name)`` and ``stack(name)`` are the usual accessors.
    """

    def last(self, name: str) -> torch.Tensor:
        values = self[name]
        if not values:
            raise KeyError(f"no activation recorded for {name!r}")
        return values[-1]

    def stack(self, name: str) -> torch.Tensor:
        return torch.stack(self[name])

    def names(self) -> list[str]:
        return list(self.keys())

    def summary(self) -> str:
        rows = [f"  {name:28s} {tuple(values[-1].shape)}" for name, values in self.items() if values]
        return "activations:\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# parameter table
# ---------------------------------------------------------------------------


@dataclass
class ParamRow:
    name: str
    shape: tuple[int, ...]
    numel: int
    trainable: bool
    group: str


# ---------------------------------------------------------------------------
# main view
# ---------------------------------------------------------------------------


class ModelAccess:
    """Experiment-friendly view on a :class:`lfm3.loader.ModelBundle`."""

    def __init__(self, bundle: Any):
        self.bundle = bundle
        self.model = bundle.model
        self.backbone = self._resolve_backbone()
        self._handles: list[Any] = []

    # ---------------------------------------------------------------- basics
    def _resolve_backbone(self) -> nn.Module:
        """Return the text backbone: ``model.model`` (base) or ``model.model.language_model``."""
        inner = getattr(self.model, "model", None)
        if inner is None:
            return self.model
        language_model = getattr(inner, "language_model", None)
        if language_model is not None:
            return language_model
        return inner

    @property
    def embed_tokens(self) -> nn.Module:
        return self.backbone.embed_tokens

    @property
    def blocks(self) -> nn.ModuleList:
        return self.backbone.layers

    @property
    def final_norm(self) -> nn.Module:
        """The RMSNorm applied to the last hidden state (upstream calls it ``embedding_norm``)."""
        return self.backbone.embedding_norm

    @property
    def lm_head(self) -> nn.Module:
        head = getattr(self.model, "lm_head", None)
        if head is not None:
            return head
        return getattr(self.model, "output_embeddings", None)

    @property
    def rotary(self) -> nn.Module | None:
        return getattr(self.backbone, "rotary_emb", None)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def layer_type(self, index: int) -> str:
        """Operator type of block ``index``: conv / sliding_attention / full_attention."""
        return self.config.layer_types[index]

    @property
    def config(self) -> Any:
        return getattr(self.backbone, "config", self.model.config)

    @property
    def layer_types(self) -> list[str]:
        return list(self.config.layer_types)

    # ----------------------------------------------------------- parameters
    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield from self.model.named_parameters()

    def parameter_table(self, trainable_only: bool = False) -> list[ParamRow]:
        """One row per parameter tensor, with the module group it belongs to."""
        rows: list[ParamRow] = []
        for name, param in self.model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            group = ".".join(name.split(".")[:2]) if name.startswith("model.") else name.split(".")[0]
            rows.append(
                ParamRow(
                    name=name,
                    shape=tuple(param.shape),
                    numel=param.numel(),
                    trainable=param.requires_grad,
                    group=group,
                )
            )
        return rows

    def parameter_groups(self) -> dict[str, dict[str, int]]:
        """Aggregate ``{group: {"total": n, "trainable": n, "tensors": k}}``."""
        out: dict[str, dict[str, int]] = {}
        for row in self.parameter_table():
            group = out.setdefault(row.group, {"total": 0, "trainable": 0, "tensors": 0})
            group["total"] += row.numel
            group["trainable"] += row.numel if row.trainable else 0
            group["tensors"] += 1
        return out

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad or not trainable_only)

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    def freeze_before(self, block_index: int) -> None:
        """Freeze embeddings and blocks ``[0, block_index)`` (LoRA-style partial fine-tuning)."""
        self.embed_tokens.requires_grad_(False)
        for block in self.blocks[:block_index]:
            block.requires_grad_(False)

    def unfreeze_after(self, block_index: int) -> None:
        for block in self.blocks[block_index:]:
            block.requires_grad_(True)

    def trainable_summary(self) -> str:
        total = self.num_parameters()
        trainable = self.num_parameters(trainable_only=True)
        pct = 100.0 * trainable / total if total else 0.0
        return f"trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({pct:.2f}%)"

    # -------------------------------------------------------------- forward
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> input embeddings (before any block)."""
        return self.embed_tokens(input_ids)

    def forward_hidden(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the text backbone and return its output.

        With ``output_hidden_states=False`` this is a ``BaseModelOutputWithPast`` whose
        ``last_hidden_state`` is the vector *after* the final RMSNorm -- i.e. the input
        of the LM head.  With ``True`` the output additionally carries
        ``hidden_states``, a tuple ``(embeddings, block_0, ..., block_N)`` where the last
        entry is *before* the final norm.
        """
        return self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

    def forward_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Full forward including the LM head; returns logits."""
        return self.model(*args, **kwargs).logits

    def logits_from_hidden(self, hidden: torch.Tensor, head: nn.Module | None = None) -> torch.Tensor:
        """Apply the LM head (or ``head``) to a hidden state."""
        return (head or self.lm_head)(hidden)

    def forward_with_ngram(
        self,
        input_ids: torch.Tensor,
        ngram: Any,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward with a hashed n-gram bias from :class:`lfm3.modules.NgramLayer`.

        ``ngram.cfg.mode`` decides where the bias enters:

        * ``"logit_bias"`` -> ``head(h) + ngram(ids)``
        * ``"hidden_add"`` -> ``head(h + ngram(ids))``

        The n-gram layer needs the *full* token sequence, so this is the path to use for
        training and teacher forcing.  During cached generation only the newest token is
        passed to the model; use the ``hidden_add`` mode there or decode without a cache.
        """
        out = self.forward_hidden(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = out.last_hidden_state
        bias = ngram(input_ids, hidden).to(hidden.dtype)
        if getattr(ngram.cfg, "mode", "logit_bias") == "logit_bias":
            return self.lm_head(hidden) + bias
        return self.lm_head(hidden + bias)

    def forward_stages(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Hidden state after every stage, keyed by a readable name.

        Returns ``embeddings``, ``layers.0 ... layers.N-1`` (each block output, *before*
        the final norm), ``final`` (after the final RMSNorm -- the input of the LM head)
        and ``logits``.

        Implemented with forward hooks rather than ``output_hidden_states``: in
        ``transformers`` >= 5.13 the last entry of ``hidden_states`` is the *post-norm*
        state, so it is not a reliable list of block outputs.  All returned tensors are
        detached.
        """
        names = ["embed_tokens", *[f"layers.{i}" for i in range(self.num_blocks)], "embedding_norm"]
        with self.capture(names) as acts:
            out = self.forward_hidden(input_ids=input_ids, **kwargs)
        stages = {name: acts.last(name) for name in acts}
        stages["embeddings"] = stages.pop("embed_tokens")
        stages["final"] = out.last_hidden_state
        stages["logits"] = self.lm_head(out.last_hidden_state)
        return stages

    def forward_with_head(self, hidden: torch.Tensor, head: nn.Module) -> torch.Tensor:
        """Apply an arbitrary module to a hidden state -- the hook point for adapters."""
        return head(hidden)

    # --------------------------------------------------------------- hooks
    def hook(self, module: nn.Module | str, fn: Callable, *, name: str | None = None) -> Any:
        """Register a forward hook on a module (or a dotted path such as ``layers.3.self_attn``)."""
        target = self.resolve(module)
        label = name or getattr(module, "__name__", None) or self.path_of(target)
        return target.register_forward_hook(lambda mod, inp, out: fn(label, inp, out))

    def resolve(self, module: nn.Module | str) -> nn.Module:
        """Resolve a module or a dotted path (relative to the model or the backbone)."""
        if isinstance(module, str):
            for root in (self.model, self.backbone):
                current: Any = root
                try:
                    for part in module.split("."):
                        current = getattr(current, part)
                except AttributeError:
                    continue
                if isinstance(current, nn.Module):
                    return current
            raise AttributeError(f"no module at path {module!r}")
        return module

    def path_of(self, module: nn.Module) -> str:
        for root in (self.model, self.backbone):
            for name, candidate in root.named_modules():
                if candidate is module:
                    return name
        return "<external module>"

    @contextmanager
    def capture(self, names: Iterable[str] | None = None, *, store_inputs: bool = False):
        """Record activations of the model while the block is active.

        ``names`` selects dotted module paths (default: embeddings, every block, final
        norm, LM head).  Yields an :class:`Activations` mapping.

        Example::

            with acc.capture() as acts:
                bundle(input_ids)
            per_layer = acts.stack("layers.4")
        """
        if names is None:
            names = ["embed_tokens", *[f"layers.{i}" for i in range(self.num_blocks)], "embedding_norm", "lm_head"]
        activations: Activations = Activations()
        handles = []

        for name in names:
            try:
                module = self.resolve(name)
            except AttributeError:
                logger.warning("capture: %s not found, skipping", name)
                continue

            def make_hook(label: str):
                def hook(mod, inputs, output):
                    value = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(value, torch.Tensor):
                        activations.setdefault(label, []).append(value.detach())
                    if store_inputs and isinstance(inputs, (tuple, list)) and inputs:
                        first = inputs[0]
                        if isinstance(first, torch.Tensor):
                            activations.setdefault(f"{label}.input", []).append(first.detach())

                return hook

            handles.append(module.register_forward_hook(make_hook(name)))

        try:
            yield activations
        finally:
            for handle in handles:
                handle.remove()

    def clear_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------- surgery
    def set_input_embeddings(self, module: nn.Module, *, tie_head: bool | None = None) -> nn.Module:
        """Replace the token embedding matrix.

        Parameters
        ----------
        module:
            Any module mapping ``(batch, seq)`` token ids to ``(batch, seq, hidden)``.
        tie_head:
            ``True`` re-ties the LM head to the new matrix (default: keep the previous
            tying state), ``False`` leaves the head untouched.
        """
        was_tied = self.is_head_tied()
        self.backbone.embed_tokens = module
        if hasattr(module, "num_embeddings"):
            self.config.vocab_size = module.num_embeddings
        if tie_head is None:
            tie_head = was_tied
        if tie_head:
            self.tie_head_to_embed()
        logger.info("replaced input embeddings with %s", type(module).__name__)
        return module

    def set_output_head(self, module: nn.Module, *, tie: bool = False) -> nn.Module:
        """Replace the LM head (the output embedding).

        ``tie=False`` (default) detaches the new head from the input embedding matrix,
        which is what you want for an adapter or a probe.
        """
        head = self.lm_head
        if head is None:
            raise AttributeError("this model has no lm_head")
        setattr(self.model, "lm_head", module)
        if tie:
            self.tie_head_to_embed()
        logger.info("replaced lm_head (%s -> %s)", type(head).__name__, type(module).__name__)
        return module

    # kept as an alias because "output embedding" is the term used in the task
    replace_output_head = set_output_head

    def tie_head_to_embed(self) -> None:
        """Share the input embedding weight with the LM head."""
        embed_weight = getattr(self.embed_tokens, "weight", None)
        if not isinstance(embed_weight, nn.Parameter):
            logger.warning("input embeddings expose no weight Parameter; the head is left untouched")
            return
        self.lm_head.weight = embed_weight
        self.config.tie_word_embeddings = True
        logger.info("lm_head tied to embed_tokens")

    def untie_head(self) -> None:
        """Give the LM head its own weight (copy of the embedding matrix)."""
        with torch.no_grad():
            self.lm_head.weight.copy_(self.embed_tokens.weight)
        self.config.tie_word_embeddings = False
        logger.info("lm_head untied (weights copied)")

    def is_head_tied(self) -> bool:
        embed = getattr(self.embed_tokens, "weight", None)
        head = getattr(self.lm_head, "weight", None)
        return embed is not None and head is not None and embed.data_ptr() == head.data_ptr()

    def set_embedding_matrix(self, matrix: torch.Tensor) -> None:
        """Overwrite the embedding weights in place (shape must match)."""
        weight = self.embed_tokens.weight
        if tuple(weight.shape) != tuple(matrix.shape):
            raise ValueError(f"embedding matrix must be {tuple(weight.shape)}, got {tuple(matrix.shape)}")
        with torch.no_grad():
            weight.copy_(matrix.to(weight.device, weight.dtype))

    def resize_embeddings(self, new_vocab: int, *, keep_tied: bool | None = None) -> nn.Embedding:
        """Grow or shrink the vocabulary of the input *and* output embedding.

        New rows are initialised with the mean of the existing rows (better than random
        for added tokens).  The LM head is rebuilt so its output width matches; its
        existing weights are kept for the shared rows.

        ``keep_tied=None`` keeps the previous tying state.  The tokenizer is *not*
        resized -- that is a separate step (``tokenizer.add_tokens(...)`` +
        ``tokenizer.resize_token_embeddings`` semantics) and changes the ids, so do it
        before this call if you need it.
        """
        old_embed = self.embed_tokens
        if not isinstance(old_embed, nn.Embedding):
            raise TypeError("resize_embeddings needs an nn.Embedding as input embedding")
        old_head = self.lm_head
        old_head_weight = getattr(old_head, "weight", None)
        old_vocab, hidden = old_embed.num_embeddings, old_embed.embedding_dim
        was_tied = self.is_head_tied()
        tie = was_tied if keep_tied is None else keep_tied

        new_embed = nn.Embedding(
            new_vocab, hidden, padding_idx=old_embed.padding_idx,
            device=old_embed.weight.device, dtype=old_embed.weight.dtype,
        )
        new_head = nn.Linear(hidden, new_vocab, bias=False, device=old_head.weight.device, dtype=old_head.weight.dtype)
        keep = min(new_vocab, old_vocab)
        with torch.no_grad():
            new_embed.weight[:keep].copy_(old_embed.weight[:keep])
            if old_head_weight is not None and old_head_weight.shape[0] >= keep:
                new_head.weight[:keep].copy_(old_head_weight[:keep])
            if new_vocab > old_vocab:
                mean_row = old_embed.weight.mean(dim=0, keepdim=True)
                new_embed.weight[old_vocab:].copy_(mean_row)
                new_head.weight[old_vocab:].copy_(mean_row)

        self.backbone.embed_tokens = new_embed
        setattr(self.model, "lm_head", new_head)
        if tie:
            self.lm_head.weight = new_embed.weight
        self.config.vocab_size = new_vocab
        if isinstance(getattr(self.model, "vocab_size", None), int):
            self.model.vocab_size = new_vocab
        pad = getattr(self.config, "pad_token_id", None)
        if pad is not None and pad >= new_vocab:
            self.config.pad_token_id = None
        logger.info("resized vocabulary %d -> %d (tied=%s)", old_vocab, new_vocab, tie)
        return new_embed

    def scale_embeddings(self, factor: float) -> None:
        with torch.no_grad():
            self.embed_tokens.weight.mul_(factor)
        logger.info("scaled embeddings by %.3f", factor)

    def zero_row(self, token_id: int) -> None:
        """Zero one embedding row (e.g. the pad token)."""
        with torch.no_grad():
            self.embed_tokens.weight[token_id].zero_()

    def replace_block(self, index: int, module: nn.Module) -> nn.Module:
        """Swap decoder block ``index`` for ``module``.

        The replacement must accept the Hugging Face block signature::

            forward(hidden_states, position_embeddings=None, attention_mask=None,
                    position_ids=None, past_key_values=None, seq_idx=None, **kwargs)

        :class:`lfm3.modules.BlockWrapper` implements it; see ``docs/LFM3_LAB_GUIDE.md``.
        """
        old = self.blocks[index]
        self.blocks[index] = module
        logger.info("replaced block %d (%s -> %s)", index, type(old).__name__, type(module).__name__)
        return old

    def bypass_block(self, index: int, mode: str = "identity", alpha: float = 1.0) -> nn.Module:
        """Short-circuit one block for ablations.

        ``mode`` is ``identity`` (pass through), ``zero`` (output zeros) or ``scale``
        (multiply the block output by ``alpha``).
        """
        from .modules import BlockBypass

        return self.replace_block(index, BlockBypass(mode=mode, alpha=alpha))

    def attach_after_final_norm(self, module: nn.Module, *, head: nn.Module | None = None) -> nn.Module:
        """Insert ``module`` between the final RMSNorm and the LM head.

        ``module`` maps ``(batch, seq, hidden)`` to the same shape (residual or
        bottleneck style).  If ``head`` is given, the LM head is replaced as well, so
        the pair can be trained together.  The original head stays reachable via
        ``acc.lm_head`` only if you kept a reference -- save one first.
        """
        from .modules import HeadAdapter

        wrapper = HeadAdapter(module, head=head or self.lm_head)
        setattr(self.model, "lm_head", wrapper)
        logger.info("inserted %s between final norm and lm_head", type(module).__name__)
        return wrapper

    def attach_module(self, name: str, module: nn.Module) -> nn.Module:
        """Attach an arbitrary module to the top-level model under ``name``."""
        setattr(self.model, name, module)
        logger.info("attached module %s (%s)", name, type(module).__name__)
        return module

    def restore_block(self, index: int, module: nn.Module) -> None:
        self.blocks[index] = module

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, strict: bool = False) -> Any:
        state = torch.load(path, map_location=self.bundle.device, weights_only=False)
        return self.model.load_state_dict(state, strict=strict)

    # ------------------------------------------------------------- reports
    def describe_layers(self) -> str:
        counts = {t: self.layer_types.count(t) for t in sorted(set(self.layer_types))}
        lines = [f"blocks: {self.num_blocks}  {counts}"]
        for i, layer_type in enumerate(self.layer_types):
            block = self.blocks[i]
            operator = "attn" if hasattr(block, "self_attn") else "conv"
            lines.append(f"  [{i:2d}] {layer_type:18s} {type(block).__name__:18s} operator={operator}")
        return "\n".join(lines)

    def describe_tree(self, max_depth: int = 3) -> str:
        """Module tree with parameter counts, e.g. for a paper appendix."""
        lines = []

        def walk(module: nn.Module, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            own = sum(p.numel() for p in module.parameters(recurse=False))
            lines.append(f"{prefix}{type(module).__name__}  ({own/1e6:.2f}M own)")
            for name, child in module.named_children():
                walk(child, prefix + "  ", depth + 1)

        walk(self.model, "", 0)
        return "\n".join(lines)

    def embedding_report(self) -> dict[str, Any]:
        weight = self.embed_tokens.weight
        with torch.no_grad():
            norms = weight.float().norm(dim=-1)
            dead = int((norms == 0).sum())
        return {
            "shape": tuple(weight.shape),
            "dtype": str(weight.dtype),
            "tied_to_head": self.is_head_tied(),
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "zero_rows": dead,
            "padding_idx": getattr(self.config, "pad_token_id", None),
        }


__all__ = ["ModelAccess", "Activations", "ParamRow"]