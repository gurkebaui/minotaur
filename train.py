import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer

class Trainer(nn.Module):
    def __init__(self, vocab_size=128, seq_len=32, batch_size=8, d_model=128):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Transformer(vocab_size=vocab_size, d_model=d_model, max_seq_len=seq_len).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        self.vocab_size = vocab_size

        self.fixed_seq = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=self.device)

        print(f"Starte Training auf {self.device}...")
        self.model.train()

    def train(self):
        for epoch in range(100):
            
            x = self.fixed_seq[:,:-1]
            y = self.fixed_seq[:,1:]

            self.optimizer.zero_grad()

            logits = self.model(x)

            loss = self.criterion(logits.view(-1, self.vocab_size), y.reshape(-1))

            loss.backward()

            self.optimizer.step()

            if(epoch +1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/100 | Loss {loss.item():.4f}")
        
        print(f"\nFinished Training!")

if __name__ == "__main__":
    t = Trainer()
    t.train()
