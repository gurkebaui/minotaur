from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import DiTConfig

def sinusodialTimestepEmbedding( timesteps, dim, cfg):
        #timesteps = [B]
        half_dim = dim // 2

        #freq_i = 1000 ^-2i / d_model aber so wie es der comp mag
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) *
            (math.log(cfg.max_period) / half_dim)
        )

        args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        #[B, half_dim]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) 
        return embedding 


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = int(8 * d_model / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64 #auf ein vielfaches von 64 runden

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))

        x_gated = gate * self.w_up(x)

        return self.w_down(x_gated)



class DiTBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(cfg.d_model, num_heads=cfg.num_heads, batch_first=True)

        self.ffn = SwiGLU_FFN(cfg.d_model)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 4*cfg.d_model, bias=True)
        )
    def forward(self, x, t_emb):
        #x shape [B, NumPatches, d_model]
        #t_emb [B, d_mdoel]
        ln_par = self.adaLN_modulation(t_emb)
        #[b, 4*d]
        gamma1, beta1, gamma2, beta2 = torch.chunk(ln_par, 4, dim=-1)
        #[b,d]
        gamma1 = 1 + gamma1.unsqueeze(1)#[b,1,d]
        beta1 = beta1.unsqueeze(1)
        gamma2 = 1+ gamma2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)

        norm_x = self.norm1(x) * gamma1 + beta1
        attn_out, _ = self.attn(norm_x, norm_x, norm_x) 
        x = attn_out + x

        norm_x = self.norm2(x) * gamma2 + beta2
        x = self.ffn(norm_x) + x
        return x


      


class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.d_model = cfg.d_model
        self.proj = nn.Conv2d(self.in_channels, self.d_model, self.patch_size, self.patch_size)

    def forward(self, x):
        #x [B, Channesl, Hight, Width]
        x = self.proj(x)
        #[B, d_model, H_patch, W_patch]
        x = x.flatten(2, -1)
        #[B, d_model, H*W]
        x = x.transpose(1, -1)
        #[B, H*W, d_model]
        return x
class PosEmbedding(nn.Module):
    def __init__(self, cfg):# x from patch_emb
        super().__init__()
        
        num_patches = (cfg.H // cfg.patch_size) * (cfg.W // cfg.patch_size)
        #[B, W*H, d_model] [seq_len]
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, cfg.d_model) * 0.02)
    def forward(self, x): #x from PatchEmb
        return x + self.pos_emb


class DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_embedder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )

        self.patch_emb = PatchEmbedding(cfg)
        self.pos_emb = PosEmbedding(cfg)

        self.blocks = nn.ModuleList(
            [DiTBlock(cfg) for _ in range(cfg.num_layers)]
        )

        self.final_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2*cfg.d_model, bias=True)
        )

        out_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.final_proj = nn.Linear(cfg.d_model, out_dim)

    def unpatchify(self, x):
        #[B, num_patches, patch_pixel]
        c = self.cfg.in_channels
        p = self.cfg.patch_size
        h = self.cfg.H // p
        w = self.cfg.W // p

        #[B, h*w, p*p*c]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, w*p)
        return x

    def forward(self, x_t, t_idx):
        #timestep emb
        t_sin = sinusodialTimestepEmbedding(t_idx, self.cfg.d_model, self.cfg)
        t_emb = self.t_embedder(t_sin)#[B, d]

        x = self.patch_emb(x_t)
        x = self.pos_emb(x)#[b, num_patches, d]

        for block in self.blocks:
            x = block(x, t_emb)

        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        scale = 1.0 + scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        x = self.final_norm(x) * scale + shift

        x = self.final_proj(x)

        prediction_noise = self.unpatchify(x)
        return prediction_noise

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DiTConfig()
    t = DiT(cfg).to(device)
    
    x_t = torch.randn(2, cfg.in_channels, cfg.H, cfg.W, dtype=torch.float32, device=device)
    
    x_idx = torch.tensor([10, 500], dtype=torch.long, device=device)
    
    y = t(x_t, x_idx)
    print("Input shape: ", x_t.shape)
    print("Output shape: ", y.shape)
