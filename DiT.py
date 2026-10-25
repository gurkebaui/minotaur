from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import DiTConfig


class DiT(nn.Module):

    def __init__(self, t, x_t, x_0, cfg):
        super().__init__()
        self.t = t

        hidden_dim = int(8 * cfg.d_model / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64 
        self.w_gate = nn.Linear(self.t.size(), hidden_dim, bias=False)        
        self.w_up = nn.Linear(self.t.size, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, cfg.d_model, bias=False)


    '''def TimestepEmbedding(self, cfg):
        

        gate = F.silu(self.w_gate(self.t))
        
        t_gated = gate * self.w_up(self.t)
        
        return self.w_down(t_gated)'''#depreciated

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