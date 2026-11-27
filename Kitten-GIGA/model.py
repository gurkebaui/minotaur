import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KittenConfig


def precompute_rope_freqs(d_k, max_seq_len, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_emb(x, cos, sin):
    T = x.shape[-2]
    cos = cos[:T, :].unsqueeze(0).unsqueeze(1)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(1)
    x_rot = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)
    return (x * cos) + (x_rot * sin)


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class CSA(nn.Module):
    """Causal Self-Attention mit RoPE."""

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        cos, sin = precompute_rope_freqs(self.d_k, cfg.max_seq_len, device="cpu", base=cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        cos, sin = self.cos.to(x.device), self.sin.to(x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            raw = int(8 * d_model / 3)
            hidden_dim = ((raw + 63) // 64) * 64
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RMSNorm(nn.Module):
    def __init__(self, cfg, eps=None):
        super().__init__()
        self.eps = cfg.rms_norm_eps if eps is None else eps
        self.weight = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LatentMoE(nn.Module):
    """Latent MoE mit Per-Loop-Router (gegen Expert Selection Collapse über Loops)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        l = cfg.moe_latent_dim
        h = cfg.moe_expert_dim
        e = cfg.num_experts

        self.latent_dim = l
        self.num_experts = e
        self.top_k = min(cfg.top_k, e)

        self.routers = nn.ModuleList([nn.Linear(d, e, bias=False) for _ in range(cfg.max_loops)])
        self.down_proj = nn.Linear(d, l, bias=False)
        self.up_proj = nn.Linear(l, d, bias=False)

        self.w1 = nn.Parameter(torch.randn(e, l, h) * 0.02)
        self.w3 = nn.Parameter(torch.randn(e, l, h) * 0.02)
        self.w2 = nn.Parameter(torch.randn(e, h, l) * 0.02)

        self.shared_expert = SwiGLU_FFN(d, hidden_dim=cfg.moe_shared_dim)
        self.aux_loss = None

    def forward(self, x, loop_idx=0):
        B, T, _ = x.shape
        router = self.routers[min(loop_idx, len(self.routers) - 1)]
        probs = F.softmax(router(x), dim=-1)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        if self.training:
            f = F.one_hot(topk_idx, self.num_experts).float().mean(dim=-2).mean(dim=(0, 1))
            p = probs.mean(dim=(0, 1))
            self.aux_loss = self.num_experts * (f * p).sum()
        else:
            self.aux_loss = None

        z_flat = self.down_proj(x).reshape(-1, self.latent_dim)
        idx_flat = topk_idx.reshape(-1, self.top_k)
        w_flat = topk_w.reshape(-1, self.top_k)

        out_latent = torch.zeros_like(z_flat)
        for e in range(self.num_experts):
            tok_idx, k_pos = torch.where(idx_flat == e)
            if tok_idx.numel() == 0:
                continue
            z_e = z_flat[tok_idx]
            y_e = (F.silu(z_e @ self.w1[e]) * (z_e @ self.w3[e])) @ self.w2[e]
            w_e = w_flat[tok_idx, k_pos].unsqueeze(-1).to(y_e.dtype)
            out_latent.index_add_(0, tok_idx, (y_e * w_e).to(out_latent.dtype))

        return self.up_proj(out_latent.view(B, T, self.latent_dim)) + self.shared_expert(x)


class TransformerBlock(nn.Module):
    """Dichter Standard-Block für Prelude & Coda."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg)
        self.attn = CSA(cfg)
        self.ln2 = RMSNorm(cfg)
        self.ffn = SwiGLU_FFN(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LoopedCoreBlock(nn.Module):
    """Core-Block mit IterAdaLN (Loop-Konditionierung), Causal Self-Attention und LatentMoE."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg)
        self.attn = CSA(cfg)
        self.ln2 = RMSNorm(cfg)
        self.moe = LatentMoE(cfg)
        self.iter_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, 6 * cfg.d_model))
        nn.init.zeros_(self.iter_adaln[-1].weight)
        nn.init.zeros_(self.iter_adaln[-1].bias)

    def forward(self, x, loop_cond, loop_idx=0):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.iter_adaln(loop_cond).chunk(6, dim=-1)
        x = x + (1.0 + gate_a) * self.attn(modulate(self.ln1(x), shift_a, scale_a))
        x = x + (1.0 + gate_m) * self.moe(modulate(self.ln2(x), shift_m, scale_m), loop_idx=loop_idx)
        return x


class Transformer(nn.Module):
    """Autoregressiver Middle-Loop Latent-MoE Transformer (Prelude -> Core x K -> Coda)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.d_model = cfg.d_model

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.prelude_layers)])
        self.core = nn.ModuleList([LoopedCoreBlock(cfg) for _ in range(cfg.core_layers)])
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.coda_layers)])

        self.loop_emb = nn.Embedding(cfg.max_loops, cfg.d_model)
        nn.init.normal_(self.loop_emb.weight, std=0.02)

        self.ln_f = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.last_aux_loss = None

    def forward(self, x, num_loops=None):
        loops = self.cfg.num_loops if num_loops is None else num_loops
        h = self.tok_emb(x)

        for block in self.prelude:
            h = block(h)

        x_in = h
        aux_losses = []
        for loop_idx in range(loops):
            if loop_idx > 0:
                h = 0.5 * (h + x_in)  # Input Re-Injection gegen Varianz-Drift
            loop_cond = self.loop_emb.weight[min(loop_idx, self.cfg.max_loops - 1)]
            for block in self.core:
                h = block(h, loop_cond, loop_idx=loop_idx)
                if block.moe.aux_loss is not None:
                    aux_losses.append(block.moe.aux_loss)

        self.last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        for block in self.coda:
            h = block(h)

        return self.lm_head(self.ln_f(h))


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = KittenConfig(vocab_size=128, max_seq_len=32, d_model=128, n_heads=8)
    model = Transformer(cfg).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=device)

    # 1. Forward + Backward (inkl. aux_loss)
    model.train()
    logits = model(x, num_loops=2)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
    if model.last_aux_loss is not None:
        loss = loss + cfg.moe_aux_loss_coef * model.last_aux_loss
    loss.backward()
    print(f"[OK] 1. Forward/Backward ({params:.2f}M Params, Device={device}): loss={loss.item():.3f}, aux={model.last_aux_loss.item():.3f}")

    # 2. Flexibles Test-Time-Looping & Kausalität
    model.eval()
    with torch.no_grad():
        logits_2 = model(x, num_loops=2)
        logits_4 = model(x, num_loops=4)
        x_mod = x.clone()
        x_mod[:, 6:] = (x_mod[:, 6:] + 1) % cfg.vocab_size
        leak = (logits_4[:, :6] - model(x_mod, num_loops=4)[:, :6]).abs().max().item()
    assert logits_2.shape == logits_4.shape == (2, 16, cfg.vocab_size)
    assert leak < 1e-5, f"Kausalitäts-Leak erkannt: {leak}"
    print(f"[OK] 2. Multi-Loop (2 & 4 Loops) + Kausalität (diff={leak:.2e}).")
