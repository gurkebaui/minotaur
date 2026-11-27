from dataclasses import dataclass
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from DiffusionBlocks.Dif_tse import TimestepEmbedder, apply_rope, compute_rope_freqs
from DiffusionBlocks.noise_schedule import (
    edm_loss_weight,
    edm_preconditioning,
    get_block_sigmas,
    get_inference_sigmas,
    sample_sigma_for_block,
)


@dataclass
class LoopedARConfig:
    """Konfiguration für den autoregressiven Looped DiffusionBlocks-Transformer (Prelude-Core-Coda)."""

    vocab_size: int = 256
    max_seq_len: int = 256
    dim: int = 256
    num_heads: int = 4
    prelude_layers: int = 2
    core_layers: int = 4
    coda_layers: int = 2
    mlp_dim: int | None = None
    dropout: float = 0.0
    freq_dim: int = 256
    max_period: float = 10000.0
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02

    # Noise-Schedule & EDM
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2

    # Looping & Inferenz
    core_loops: int = 1
    inference_steps: int = 16
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) muss durch num_heads ({self.num_heads}) teilbar sein.")
        if self.core_layers < 1:
            raise ValueError(f"core_layers ({self.core_layers}) muss >= 1 sein.")
        if self.core_loops < 1:
            raise ValueError(f"core_loops ({self.core_loops}) muss >= 1 sein.")
        if self.mlp_dim is None:
            raw = int(self.dim * 8 / 3)
            self.mlp_dim = ((raw + 31) // 32) * 32


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Modulation: x * (1 + scale) + shift, gebroadcastet über die Sequenzlänge."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (wie in Llama-2)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """Feed-Forward-Netzwerk mit SwiGLU-Aktivierung (wie in Llama-2)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention mit RoPE und flexibler Kausal- oder Trainings-Maske."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(out)


class DiTTransformerLayer(nn.Module):
    """Llama-2-Transformer-Layer mit DiT AdaLN-Zero für Noise-Conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.mlp = SwiGLU(dim, mlp_dim, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(
            6, dim=-1
        )
        h_attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cos, sin, attn_mask)
        x = x + gate_msa.unsqueeze(1) * h_attn

        h_mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h_mlp
        return x


def build_training_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Erzeugt die (2S, 2S) Attention-Maske für das AR-Training (Appendix E.4)."""
    tril = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    zeros = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)
    return torch.cat([torch.cat([tril, zeros], dim=1), torch.cat([tril, eye], dim=1)], dim=0)


class LoopedDiffusionBlock(nn.Module):
    """Middle-Loop-Architektur: Prelude (einmal) -> Core (N-fach geloopt) -> Coda (einmal)."""

    def __init__(self, config: LoopedARConfig) -> None:
        super().__init__()
        make_layer = lambda: DiTTransformerLayer(
            config.dim,
            config.num_heads,
            config.mlp_dim,
            config.dropout,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.prelude = nn.ModuleList([make_layer() for _ in range(config.prelude_layers)])
        self.core = nn.ModuleList([make_layer() for _ in range(config.core_layers)])
        self.coda = nn.ModuleList([make_layer() for _ in range(config.coda_layers)])

    def all_layers(self):
        yield from self.prelude
        yield from self.core
        yield from self.coda

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        num_loops: int = 1,
    ) -> torch.Tensor:
        for layer in self.prelude:
            x = layer(x, cond, cos, sin, attn_mask=attn_mask)

        x_in = x
        h = x_in
        for loop_idx in range(num_loops):
            if loop_idx > 0:
                # Input-Re-Injection + Varianz-Dämpfung gegen Drift über mehrere Loops
                h = 0.5 * (h + x_in)
            for layer in self.core:
                h = layer(h, cond, cos, sin, attn_mask=attn_mask)

        for layer in self.coda:
            h = layer(h, cond, cos, sin, attn_mask=attn_mask)
        return h


class LoopedARDiffusionModel(nn.Module):
    """
    Autoregressiver Looped-Transformer nach DiffusionBlocks (Section 5.5 / Appendix E.5):
    - Single-Pass-Training über alle Noise-Level (kein BPTT nötig).
    - Beliebig skalierbare Euler-Denoising-Schritte in der Inferenz.
    """

    def __init__(self, config: LoopedARConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_embedder = TimestepEmbedder(
            config.dim, freq_dim=config.freq_dim, max_period=config.max_period
        )
        self.net = LoopedDiffusionBlock(config)

        self.final_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, 2 * config.dim))
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        head_dim = config.dim // config.num_heads
        cos, sin = compute_rope_freqs(head_dim, config.max_seq_len + 1, base=config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.block_sigmas = get_block_sigmas(
            num_blocks=1,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            p_mean=config.p_mean,
            p_std=config.p_std,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init_std = self.config.init_std
        nn.init.normal_(self.token_emb.weight, std=init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for layer in self.net.all_layers():
            nn.init.zeros_(layer.adaLN[-1].weight)
            nn.init.zeros_(layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.lm_head.weight)

    def get_normalized_embeddings(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids is None:
            return F.normalize(self.token_emb.weight, p=2, dim=-1)
        return F.normalize(self.token_emb(token_ids), p=2, dim=-1)

    def denoise_train(
        self,
        x_ids: torch.Tensor,
        z_noisy: torch.Tensor,
        sigma: torch.Tensor,
        core_loops: int | None = None,
    ) -> torch.Tensor:
        bsz, seq_len = x_ids.shape
        device = x_ids.device
        loops = self.config.core_loops if core_loops is None else core_loops

        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.config.sigma_data)
        clean_emb = self.get_normalized_embeddings(x_ids)
        noisy_in = z_noisy * c_in[:, None, None]

        h = torch.cat([clean_emb, noisy_in], dim=1)
        positions = torch.cat(
            [torch.arange(seq_len, device=device), torch.arange(1, seq_len + 1, device=device)],
            dim=0,
        )
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        cond = self.time_embedder(c_noise)
        attn_mask = build_training_mask(seq_len, device)

        h = self.net(h, cond, cos, sin, attn_mask=attn_mask, num_loops=loops)
        h_noisy = self.final_norm(h[:, seq_len:, :])
        model_out = h_noisy * c_out[:, None, None] + z_noisy * c_skip[:, None, None]

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        return self.lm_head(modulate(model_out, shift, scale))

    def training_loss(
        self, tokens: torch.Tensor, core_loops: int | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x_ids, y_ids = tokens[:, :-1], tokens[:, 1:]
        bsz, seq_len = x_ids.shape

        y_emb = self.get_normalized_embeddings(y_ids)
        sigma = sample_sigma_for_block(
            block_idx=0,
            batch_size=bsz,
            block_sigmas=self.block_sigmas,
            gamma=0.0,
            p_mean=self.config.p_mean,
            p_std=self.config.p_std,
            device=tokens.device,
        )
        z_noisy = y_emb + sigma[:, None, None] * torch.randn_like(y_emb)

        logits = self.denoise_train(x_ids, z_noisy, sigma, core_loops=core_loops)
        ce = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            y_ids.reshape(-1),
            reduction="none",
        ).view(bsz, seq_len)

        loss = (ce * edm_loss_weight(sigma, self.config.sigma_data)[:, None]).mean()
        return loss, {"loss": float(loss.item()), "ce_loss": float(ce.mean().item())}

    def denoise_step(
        self,
        prefix_ids: torch.Tensor,
        z_t: torch.Tensor,
        sigma: torch.Tensor,
        core_loops: int | None = None,
    ) -> torch.Tensor:
        device = prefix_ids.device
        seq_len = prefix_ids.shape[1]
        loops = self.config.core_loops if core_loops is None else core_loops

        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.config.sigma_data)
        clean_emb = self.get_normalized_embeddings(prefix_ids)
        noisy_in = (z_t * c_in[:, None]).unsqueeze(1)

        h = torch.cat([clean_emb, noisy_in], dim=1)
        positions = torch.arange(seq_len + 1, device=device)
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        cond = self.time_embedder(c_noise)

        h = self.net(h, cond, cos, sin, attn_mask=None, num_loops=loops)
        h_last = self.final_norm(h[:, -1, :])
        model_out = h_last * c_out[:, None] + z_t * c_skip[:, None]

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        return self.lm_head(model_out * (1.0 + scale) + shift)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        num_steps: int | None = None,
        core_loops: int | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        num_steps = self.config.inference_steps if num_steps is None else num_steps
        temperature = self.config.temperature if temperature is None else temperature

        was_training = self.training
        self.eval()
        device = prompt_ids.device
        bsz = prompt_ids.shape[0]

        sigmas = get_inference_sigmas(
            num_steps=num_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            p_mean=self.config.p_mean,
            p_std=self.config.p_std,
            device=device,
        )
        norm_emb_table = self.get_normalized_embeddings()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            context = ids[:, -self.config.max_seq_len :]
            z = torch.randn(bsz, self.config.dim, device=device) * math.sqrt(
                1.0 + sigmas[0].item() ** 2
            )

            for i in range(len(sigmas) - 1):
                sigma_val = sigmas[i].item()
                next_sigma_val = sigmas[i + 1].item()
                sigma = torch.full((bsz,), sigma_val, device=device)

                logits = self.denoise_step(context, z, sigma, core_loops=core_loops)
                denoised = F.softmax(logits, dim=-1) @ norm_emb_table
                d = (z - denoised) / sigma[:, None]
                z = z + (next_sigma_val - sigma_val) * d

            sigma_final = torch.full((bsz,), sigmas[-1].item(), device=device)
            logits = self.denoise_step(context, z, sigma_final, core_loops=core_loops)

            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)

        if was_training:
            self.train()
        return ids


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LoopedARConfig(
        vocab_size=64,
        max_seq_len=32,
        dim=64,
        num_heads=4,
        prelude_layers=2,
        core_layers=4,
        coda_layers=2,
        core_loops=1,
        inference_steps=16,
    )
    model = LoopedARDiffusionModel(cfg).to(device)
    tokens = torch.randint(0, cfg.vocab_size, (2, 17), device=device)

    # 1. Single-Pass-Training & Backward
    model.train()
    loss, info = model.training_loss(tokens)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters()), "Alle Layer müssen Gradienten erhalten!"
    print(f"[OK] 1. Single-Pass-Training: loss={info['loss']:.3f}, ce={info['ce_loss']:.3f}")

    # Für Leakage- & Äquivalenz-Tests nicht-triviale Gewichte setzen
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.1)
    model.eval()

    x_ids = tokens[:, :-1]
    z_noisy = torch.randn(2, x_ids.shape[1], cfg.dim, device=device)
    sigma = torch.tensor([1.5, 1.5], device=device)

    # 2. Zero-Leakage (für core_loops=1 und core_loops=2)
    with torch.no_grad():
        for loops in (1, 2):
            logits_orig = model.denoise_train(x_ids, z_noisy, sigma, core_loops=loops)
            x_mod = x_ids.clone()
            x_mod[:, 6:] = (x_mod[:, 6:] + 1) % cfg.vocab_size
            z_mod = z_noisy.clone()
            z_mod[:, 6:] = torch.randn_like(z_mod[:, 6:])
            logits_mod = model.denoise_train(x_mod, z_mod, sigma, core_loops=loops)
            max_leak = (logits_orig[:, :6] - logits_mod[:, :6]).abs().max().item()
            assert max_leak < 1e-5, f"Leakage bei core_loops={loops}: {max_leak}"
            print(f"[OK] 2. Zero-Leakage (core_loops={loops}): diff={max_leak:.2e}")

        # 3. Train-Inferenz-Äquivalenz
        logits_orig = model.denoise_train(x_ids, z_noisy, sigma, core_loops=2)
        max_eq_diff = 0.0
        for i in range(x_ids.shape[1]):
            logits_inf = model.denoise_step(x_ids[:, : i + 1], z_noisy[:, i], sigma, core_loops=2)
            max_eq_diff = max(max_eq_diff, (logits_orig[:, i] - logits_inf).abs().max().item())
        assert max_eq_diff < 1e-4, f"Train-Inferenz-Abweichung zu groß: {max_eq_diff}"
        print(f"[OK] 3. Train-Inferenz-Äquivalenz (core_loops=2): max_diff={max_eq_diff:.2e}")

    # 4. Skalierbare Test-Time-Loops (4 vs. 16 Euler-Schritte)
    prompt = tokens[:1, :6]
    out_4 = model.generate(prompt, max_new_tokens=6, num_steps=4)
    out_16 = model.generate(prompt, max_new_tokens=6, num_steps=16)
    assert out_4.shape == (1, 12) and out_16.shape == (1, 12)
    print(f"[OK] 4. Test-Time-Looping: 4 & 16 Euler-Schritte erfolgreich (Device={device}).")
