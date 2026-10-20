from typing import Any

import torch
import torch.nn as nn
import math
from data import CustomDataset
from config import ModelConfig 



def precompute_rope_freqs(d_k, max_seq_len, device):
    inv_freq = 1.0 / (10000 ** torch.arange(0, d_k, 2, dtype=torch.float32, device="cpu") / d_k)
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def apply_rotary_emb(x, cos, sin):
    T = x.shape[-2]
    cos = cos[:T, :].unsqueeze(0).unsqueeze(1)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(1)

    x_rot = torch.cat((-x[..., x.shape[-1]//-2:], x[..., :x.shape[-1]//2]), dim=-1)
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
        B, T, C = x.shape #what are btc here? they in torch?

        #nfu
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)


        q = apply_rotary_emb(q, self.cos.to(x.device), self.sin.to(x.device))
        k = apply_rotary_emb(k, self.cos.to(x.device), self.sin.to(x.device))


        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k) #what happens
        
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)#nsh
        scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)




class RMSNorm(nn.Module):
    def __init__(self, cfg, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)

        return x * rms * self.weight 

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.dataset = CustomDataset()
        self.vocab_size = cfg.vocab_size
        self.d_model = cfg.d_model

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        #self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) #depreciated

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(0, T, device=x.device)

        h = self.tok_emb(x) 
        
        for block in self.blocks:
            h = block(h)

            
        h = self.ln_f(h)

        logits = self.lm_head(h)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg)
        self.attn = CSA(cfg)
        self.ln2 = RMSNorm(cfg)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4*cfg.d_model),
            nn.GELU(),
            nn.Linear(4*cfg.d_model, cfg.d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    t = Transformer(cfg).to(device)
    x = torch.randint(0,128, (4, 16), device=device)
    y = t(x)
    print("Input shape: ", x.shape)
    print("output shape: ", y.shape)
