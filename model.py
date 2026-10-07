import torch
import torch.nn as nn
import math

class CSA(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads #nsh

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape #what are btc here? they in torch?

        #nfu
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k) #what happens
        
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)#nsh
        scores = scores + mask

        attn_weights = torch.softmax(scores, dim=1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class Transformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=256, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CSA(d_model=d_model, n_heads=8)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(0, T, device=x.device)

        h = self.tok_emb(x) + self.pos_emb(pos)

        h = h + self.attn(self.ln1(h))
        h = h + self.ffn(self.ln2(h))

        logits = self.lm_head(h)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = Transformer().to(device)
    x = torch.randint(0,128, (4, 16), device=device)
    y = t(x)
    print("Input shape: ", x.shape)
    print("output shape: ", y.shape)
