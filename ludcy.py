import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


def precompute_rope_freqs(d_k, max_seq_len, device):
  inv_freq = 1.0 / (
      10000 ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
  )
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


class CSA(nn.Module):

  def __init__(self, cfg):
    super().__init__()
    self.d_model = cfg.d_model
    self.n_heads = cfg.n_heads
    self.d_k = cfg.d_model // cfg.n_heads

    self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
    self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
    self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
    self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    cos, sin = precompute_rope_freqs(self.d_k, cfg.max_seq_len, device="cpu")
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)

  def forward(self, x):
    B, T, C = x.shape
    q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    q = apply_rotary_emb(q, self.cos.to(x.device), self.sin.to(x.device))
    k = apply_rotary_emb(k, self.cos.to(x.device), self.sin.to(x.device))

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    mask = torch.triu(
        torch.full((T, T), float("-inf"), device=x.device), diagonal=1
    )
    scores = scores + mask

    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.out_proj(out)


class SwiGLU_FFN1(nn.Module):
  """Standard SwiGLU used in the 3-layer trunk and the u-branch."""

  def __init__(self, d_model):
    super().__init__()
    hidden_dim = int(8 * d_model / 3)
    hidden_dim = ((hidden_dim + 63) // 64) * 64

    self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
    self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
    self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

  def forward(self, x):
    gate = F.silu(self.w_gate(x))
    x_gated = gate * self.w_up(x)
    return self.w_down(x_gated)


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = int(8 * d_model / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.u_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x, u):
        B, T, C = x.shape

        # Causal cumulative mean: token t only sees u from 0..t (no future leakage)
        counts = torch.arange(1, T + 1, device=u.device, dtype=u.dtype).view(1, T, 1)
        u_causal = torch.cumsum(u, dim=1) / counts  # Shape: (B, T, d_model)

        u_proj = F.silu(self.u_gate(u_causal))      # Shape: (B, T, hidden_dim)

        # Per-token rank-1 update applied efficiently without materializing (B, T, H, D):
        # (W_up - outer(u_proj, u_causal)) @ x  ==  W_up(x) - u_proj * dot(u_causal, x)
        dot_ux = (u_causal * x).sum(dim=-1, keepdim=True)  # Shape: (B, T, 1)
        up = self.w_up(x) - (u_proj * dot_ux)              # Shape: (B, T, hidden_dim)

        gate = F.silu(self.w_gate(x))
        return self.w_down(gate * up)



class RMSNorm(nn.Module):

  def __init__(self, cfg, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(cfg.d_model))

  def forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    rms = torch.rsqrt(variance + self.eps)
    return x * rms * self.weight


class TransformerBlock1(nn.Module):

  def __init__(self, cfg):
    super().__init__()
    self.ln1 = RMSNorm(cfg)
    self.attn = CSA(cfg)
    self.ln2 = RMSNorm(cfg)
    self.ffn = SwiGLU_FFN1(cfg.d_model)

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x


class TransformerBlock(nn.Module):

  def __init__(self, cfg):
    super().__init__()
    self.ln1 = RMSNorm(cfg)
    self.attn = CSA(cfg)
    self.ln2 = RMSNorm(cfg)
    self.ffn = SwiGLU_FFN(cfg.d_model)

  def forward(self, x, u):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x), u)
    return x


class Transformer(nn.Module):

  def __init__(self, cfg):
    super().__init__()
    assert cfg.n_layers > 3, "n_layers must be greater than 3 to branch"
    self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

    # 1. Shared 3-layer trunk
    self.trunk_blocks = nn.ModuleList(
        [TransformerBlock1(cfg) for _ in range(3)]
    )

    # 2. Branch A: Computes update representation u
    self.u_blocks = nn.ModuleList(
        [TransformerBlock1(cfg) for _ in range(cfg.n_layers - 3)]
    )

    # 3. Branch B: Uses u to update w_up in each layer
    self.main_blocks = nn.ModuleList(
        [TransformerBlock(cfg) for _ in range(cfg.n_layers - 3)]
    )

    self.ln_u = RMSNorm(cfg)
    self.ln_f = RMSNorm(cfg)
    self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    self.lm_head.weight = self.tok_emb.weight

  def forward(self, x):
    h = self.tok_emb(x)

    # Run first 3 shared layers
    for block in self.trunk_blocks:
      h = block(h)

    # Branch 1: Compute u first so it is ready to update w_up
    u = h
    for block in self.u_blocks:
      u = block(u)
    u = self.ln_u(u)

    # Branch 2: Run main layers with w_up dynamically updated by u
    for block in self.main_blocks:
      h = block(h, u)

    h = self.ln_f(h)
    logits = self.lm_head(h)
    return logits, u


if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  cfg = ModelConfig()
  t = Transformer(cfg).to(device)

  x = torch.randint(0, 128, (4, 16), device=device)
  y, u = t(x)

  print("Input shape: ", x.shape)
  print("Logits shape:", y.shape)
  print("u shape:     ", u.shape)
