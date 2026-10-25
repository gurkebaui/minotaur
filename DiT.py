from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import DiTConfig

def sinusodialTimestepEmbedding(self, timesteps, dim, cfg):
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
class DiTBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(cfg.d_model, num_heads=cfg.num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model *4),
            nn.SiLU(),
            nn.Linear(4*cfg.d_model, cfg.d_model)
        )

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
        x = self.attn(norm_x, norm_x, norm_x) + x

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
    def __init__(self, cfg, x):# x from patch_emb
        super().__init__()
        
        num_patches = (cfg.H // cfg.patch_size) * (cfg.W // cfg.patch_size)
        #[B, W*H, d_model] [seq_len]
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, cfg.d_model) * 0.02)
    def forward(self, x): #x from PatchEmb
        return x + self.pos_emb