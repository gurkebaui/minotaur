import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_1 = nn.Linear(128, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.m_attn = nn.MultiheadAttention(512, 2)
        self.l_norm_2 = nn.LayerNorm(512)
        self.fc_1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.emb_1(x)
        x = self.layer_norm(x)
        x = self.m_attn(x)
        x = self.l_norm_2(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x

t = Transformer()
x = torch.randn(4, 128)
y = t(x)
print(y)
print(y.shape)
    
