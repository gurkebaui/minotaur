import torch
import torch.nn as nn
import math
from data import CustomDataset
from config import ModelConfig 

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

    def forward(self, x):
        B, T, C = x.shape #what are btc here? they in torch?

        #nfu
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k) #what happens
        
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)#nsh
        scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.dataset = CustomDataset()
        self.vocab_size = cfg.vocab_size
        self.d_model = cfg.d_model

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(0, T, device=x.device)

        h = self.tok_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            h = block(h)

            
        h = self.ln_f(h)

        logits = self.lm_head(h)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CSA(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
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
