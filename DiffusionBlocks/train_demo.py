"""
Verifikations-Tests und Mini-Trainingsdemo für den AR-DiffusionBlocks-Transformer.

Ausführen:
    python3 ar_diffusion_blocks/train_demo.py
    python3 ar_diffusion_blocks/train_demo.py --only-verify
    python3 ar_diffusion_blocks/train_demo.py --steps 1200
"""

import argparse
import sys
from pathlib import Path
import torch

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from config import ARConfig
from DiffusionBlocks.Dif_ar import ARDiffusionBlocksModel
from DiffusionBlocks.noise_schedule import get_block_index, get_inference_sigmas


DEMO_TEXT = (
    "DiffusionBlocks teilt einen Transformer in unabhängige Blöcke auf. "
    "Jeder Block lernt das Entrauschen in seinem eigenen Noise-Intervall. "
    "Im Training braucht immer nur ein Block Gradienten, was Speicher spart. "
    "Bei der Inferenz arbeitet sich das Modell von starkem zu schwachem Rauschen vor. "
)


class CharTokenizer:
    """Minimaler Zeichen-Tokenizer ohne externe Abhängigkeiten."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def _randomize_zero_init_layers(model: ARDiffusionBlocksModel) -> None:
    """
    Hilfsfunktion nur für Tests: AdaLN-Zero initialisiert lm_head & adaLN mit 0.
    Für Äquivalenz- und Leakage-Tests brauchen wir nicht-triviale Gewichte.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.1)


def run_verification_tests(device: torch.device) -> None:
    """Prüft die 4 mathematischen Kern-Eigenschaften von AR DiffusionBlocks."""
    torch.manual_seed(42)
    cfg = ARConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=64,
        num_heads=4,
        num_layers=8,
        num_blocks=4,
        dropout=0.0,
    )
    model = ARDiffusionBlocksModel(cfg).to(device)
    tokens = torch.randint(0, cfg.vocab_size, (2, 17), device=device)

    # 1. Block-Isolation-Test: Nur Block 1 darf Gradienten bekommen
    model.train()
    model.zero_grad()
    loss, info = model.training_loss(tokens, block_idx=1)
    loss.backward()

    for b_idx, block in enumerate(model.blocks):
        grads = [p.grad is not None for p in block.parameters()]
        if b_idx == 1:
            assert all(grads), "Block 1 sollte überall Gradienten haben!"
        else:
            assert not any(grads), f"Block {b_idx} darf KEINE Gradienten haben!"
    print("[OK] 1. Block-Isolation: Nur der aktive Block erhält Gradienten.")

    # Für Tests 2 & 3 nicht-triviale Gewichte setzen und Dropout deaktivieren
    _randomize_zero_init_layers(model)
    model.eval()

    # 2. Zero-Leakage-Test: Zukunftstokens dürfen frühere Logits nicht beeinflussen
    x_ids = tokens[:, :-1]
    z_noisy = torch.randn(2, x_ids.shape[1], cfg.dim, device=device)
    sigma = torch.tensor([1.5, 1.5], device=device)
    test_block_idx = get_block_index(sigma[0], model.block_sigmas)

    with torch.no_grad():
        logits_orig = model.denoise_train_block(x_ids, z_noisy, sigma, block_idx=test_block_idx)
        assert logits_orig.abs().max() > 0.1, "Test-Logits sollten nicht trivial 0 sein."

        x_mod = x_ids.clone()
        x_mod[:, 6:] = (x_mod[:, 6:] + 1) % cfg.vocab_size
        z_mod = z_noisy.clone()
        z_mod[:, 6:] = torch.randn_like(z_mod[:, 6:])
        logits_mod = model.denoise_train_block(x_mod, z_mod, sigma, block_idx=test_block_idx)

    max_leak = (logits_orig[:, :6] - logits_mod[:, :6]).abs().max().item()
    assert max_leak < 1e-5, f"Zukunfts-Leakage erkannt: {max_leak}"
    print(f"[OK] 2. Zero-Leakage: Kausale 2S×2S-Maske verhindert Zukunfts-Leaks (diff={max_leak:.2e}).")

    # 3. Train-Inferenz-Äquivalenz: denoise_train_block(pos i) == denoise_step(prefix[:i+1])
    with torch.no_grad():
        max_eq_diff = 0.0
        for i in range(x_ids.shape[1]):
            logits_inf = model.denoise_step(x_ids[:, : i + 1], z_noisy[:, i], sigma)
            diff = (logits_orig[:, i] - logits_inf).abs().max().item()
            max_eq_diff = max(max_eq_diff, diff)

    assert max_eq_diff < 1e-4, f"Train-Inferenz-Abweichung zu groß: {max_eq_diff}"
    print(f"[OK] 3. Train-Inferenz-Äquivalenz: Exakte Übereinstimmung (max_diff={max_eq_diff:.2e}).")

    # 4. Noise-Schedule-Abdeckung: 4 Inferenz-Schritte besuchen exakt Blöcke [0, 1, 2, 3]
    sigmas = get_inference_sigmas(num_steps=cfg.num_blocks)
    visited = [get_block_index(s, model.block_sigmas) for s in sigmas]
    assert visited == [0, 1, 2, 3], f"Erwartet [0, 1, 2, 3], erhalten {visited}"
    print(f"[OK] 4. Noise-Schedule-Abdeckung: Inferenz-Schritte besuchen Blöcke {visited}.")


def run_training_demo(steps: int, device: torch.device) -> None:
    """Trainiert ein kleines AR-DiffusionBlocks-Modell auf dem Demo-Text und generiert Text."""
    torch.manual_seed(42)
    tokenizer = CharTokenizer(DEMO_TEXT)
    encoded = torch.tensor(tokenizer.encode(DEMO_TEXT), dtype=torch.long, device=device)

    seq_len = 64
    batch_size = 16
    cfg = ARConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        dim=128,
        num_heads=4,
        num_layers=8,
        num_blocks=4,
        dropout=0.0,
        batch_size=batch_size,
        steps=steps,
        lr=3e-3,
    )
    torch.manual_seed(cfg.seed)
    model = ARDiffusionBlocksModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )

    print(f"\nStarte Mini-Training ({cfg.steps} Schritte, {cfg.num_blocks} Blöcke, Device={device})...")
    block_ce = {b: [] for b in range(cfg.num_blocks)}

    model.train()
    for step in range(1, cfg.steps + 1):
        starts = torch.randint(0, len(encoded) - cfg.max_seq_len - 1, (cfg.batch_size,), device=device)
        batch = torch.stack([encoded[s : s + cfg.max_seq_len + 1] for s in starts])

        optimizer.zero_grad()
        loss, info = model.training_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        block_ce[info["block_idx"]].append(info["ce_loss"])

        if step % (cfg.steps // 4) == 0 or step == cfg.steps:
            summary = " | ".join(
                f"B{b}: {sum(v)/len(v):.3f}" if v else f"B{b}: -"
                for b, v in block_ce.items()
            )
            print(f"Schritt {step:4d}/{cfg.steps} | Ø CE-Loss -> {summary}")
            block_ce = {b: [] for b in range(cfg.num_blocks)}

    prompt = "DiffusionBlocks "
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=90,
        num_steps=cfg.inference_steps,
        temperature=cfg.temperature,
    )
    print("\nGenerierter Text:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{tokenizer.decode(generated_ids[0].tolist())}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="AR DiffusionBlocks Demo & Verifikation")
    parser.add_argument("--steps", type=int, default=1200, help="Anzahl Trainingsschritte")
    parser.add_argument("--only-verify", action="store_true", help="Nur Verifikationstests ausführen")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_verification_tests(device)
    if not args.only_verify:
        run_training_demo(args.steps, device)


if __name__ == "__main__":
    main()
