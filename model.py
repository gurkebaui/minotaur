import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 128
        self.d_model = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb_1 = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(device)
        self.scores = self.scores + mask
        self. attn_weights = torch.softmax(self.scores, dim=-1)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4*self.d_model),
            nn.GELU(),
            nn.Linear(4*self.d_model, self.d_model)
        )
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        

    def forward(self, x):
        pass

t = Transformer()
x = torch.randn(4, 128)
y = t(x)
print(y)
print(y.shape)
    
