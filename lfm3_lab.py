#!/usr/bin/env python
"""Command line entry point for the LFM3 lab.

Examples
--------
::

    # architecture report without downloading weights
    python lfm3_lab.py arch

    # full report of the real checkpoint
    python lfm3_lab.py info

    # every experiment on the real model
    python lfm3_lab.py demo --all

    # every experiment on a tiny random model (seconds, no checkpoint download)
    python lfm3_lab.py demo --all --tiny

    # text generation
    python lfm3_lab.py generate --prompt "Once upon a time" --tokens 40

    # omni model (downloads ~4.3 GB)
    python lfm3_lab.py info --model omni

Common options (``--model``, ``--tiny``, ``--device``, ``--dtype``, ``--attn``) may be
given before or after the sub-command.
"""

from __future__ import annotations

import argparse

from lfm3.config import LFM3_1_7B_BASE, LFM3_OMNI_2B, LoadOptions
from lfm3.experiments import DEMOS, run_all
from lfm3.inspect import describe, describe_architecture, parameter_budget


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def build_load_options(args: argparse.Namespace) -> LoadOptions:
    opts = LoadOptions(device=args.device, dtype=args.dtype, attn_implementation=args.attn)
    if getattr(args, "tiny", False):
        opts.random_init = True
        opts.load_tokenizer = False
        opts.device = opts.device or "cpu"
        opts.dtype = "float32"
        opts.attn_implementation = "eager"
    return opts


def load_bundle(args: argparse.Namespace):
    from lfm3.loader import load_base, load_omni, load_tiny, load_tiny_omni

    if getattr(args, "tiny", False):
        return load_tiny_omni() if args.model == "omni" else load_tiny()

    opts = build_load_options(args)
    if args.model == "omni":
        return load_omni(LFM3_OMNI_2B, opts)
    return load_base(LFM3_1_7B_BASE, opts)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_arch(args: argparse.Namespace) -> None:
    cfg = LFM3_OMNI_2B if args.model == "omni" else LFM3_1_7B_BASE
    print(describe_architecture(cfg))
    if args.budget:
        print()
        print(parameter_budget(getattr(cfg, "text_config", cfg)))


def cmd_info(args: argparse.Namespace) -> None:
    bundle = load_bundle(args)
    print(describe(bundle))
    if args.budget:
        lab_cfg = bundle.lab_config
        print()
        print(parameter_budget(getattr(lab_cfg, "text_config", lab_cfg)))


def cmd_demo(args: argparse.Namespace) -> None:
    if args.list:
        for name, fn in DEMOS.items():
            print(f"{name:12s} {(fn.__doc__ or '').splitlines()[0]}")
        return

    bundle = load_bundle(args)
    if args.all:
        run_all(bundle)
        return
    for name in args.demo:
        if name not in DEMOS:
            raise SystemExit(f"unknown demo {name!r}; choose from {', '.join(DEMOS)}")
        print()
        print("=" * 78)
        print(f"### {name}")
        print("=" * 78)
        DEMOS[name](bundle)


def cmd_generate(args: argparse.Namespace) -> None:
    bundle = load_bundle(args)
    if bundle.tokenizer is None:
        print("generation needs a tokenizer: drop --tiny or load the real checkpoint")
        return
    ids = bundle.tokenizer(args.prompt, return_tensors="pt").input_ids.to(bundle.device)
    print(f"prompt: {args.prompt!r}")
    out = bundle.generate(
        ids,
        max_new_tokens=args.tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,
        use_cache=True,
    )
    text = bundle.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(text)
    print("-" * 60)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add the shared options.

    The sub-parsers get ``argparse.SUPPRESS`` defaults so that a global option given
    *before* the sub-command is not overwritten by the sub-parser.
    """
    default = argparse.SUPPRESS if suppress else None

    def value(fallback):
        return default if suppress else fallback

    parser.add_argument("--model", choices=["base", "omni"], default=value("base"), help="checkpoint family")
    parser.add_argument("--tiny", action="store_true", default=value(False), help="random tiny model (only config files are fetched)")
    parser.add_argument("--device", default=value(None), help="torch device (default: cuda if available)")
    parser.add_argument("--dtype", default=value("bfloat16"), help="bfloat16 / float16 / float32")
    parser.add_argument(
        "--attn", default=value("sdpa"), help="attention backend: sdpa / eager / flash_attention_2"
    )
    parser.add_argument("--budget", action="store_true", default=value(False), help="print the parameter budget too")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common(common, suppress=True)

    parser = argparse.ArgumentParser(description="LFM3 lab (Liquid AI LFM3 research helper)")
    add_common(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("arch", parents=[common], help="architecture report from the config (no weights)").set_defaults(
        func=cmd_arch
    )
    sub.add_parser("info", parents=[common], help="full report of the loaded checkpoint").set_defaults(func=cmd_info)

    demo = sub.add_parser("demo", parents=[common], help="run experiments")
    demo.add_argument("demo", nargs="*", help="demo names (default: all)")
    demo.add_argument("--all", action="store_true", help="run every demo")
    demo.add_argument("--list", action="store_true", help="list the available demos")
    demo.set_defaults(func=cmd_demo)

    gen = sub.add_parser("generate", parents=[common], help="sample a continuation")
    gen.add_argument("--prompt", default="Once upon a time")
    gen.add_argument("--tokens", type=int, default=32)
    gen.add_argument("--sample", action="store_true")
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())