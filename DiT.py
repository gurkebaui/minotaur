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


    def TimestepEmbedding(self, cfg):
        

        gate = F.silu(self.w_gate(self.t))
        
        t_gated = gate * self.w_up(self.t)
        
        return self.w_down(t_gated)