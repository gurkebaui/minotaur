import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from data import CustomDataset
from torch.utils.data import Dataset, DataLoader

class Trainer(nn.Module):
    def __init__(self, vocab_size=50257, seq_len=64, batch_size=8, d_model=128):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = CustomDataset()
        self.model = Transformer(vocab_size=len(self.dataset.tokenizer), d_model=d_model, max_seq_len=seq_len).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        self.vocab_size = len(self.dataset.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

        #self.fixed_seq = self.data()

        print(f"Starte Training auf {self.device}...")
        self.model.train()

    def train(self):
        for epoch in range(5):
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                #print(f"Min tokenid {batch_x.min().item()} max id {batch_x.max().item()}")
                

                logits = self.model(batch_x)

                loss = self.criterion(logits.view(-1, self.vocab_size), batch_y.reshape(-1))

                loss.backward()

                self.optimizer.step()

            if(epoch +1) % 1 == 0:
                print(f"Epoch {epoch+1:3d}/100 | Loss {loss.item():.4f}")
        
        print(f"\nFinished Training!")

if __name__ == "__main__":
    t = Trainer()
    t.train()
