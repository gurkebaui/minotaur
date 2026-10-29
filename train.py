import torch
import torch.nn as nn
from torch.optim import Muon, AdamW
import torch.optim as optim
from model import Transformer
from data import CustomDataset
from eval_data import EvalDataset
from torch.utils.data import Dataset, DataLoader
import statistics
from config import ModelConfig
import os
from datetime import datetime
import math


def get_lr(step, cfg):
    max_iters = cfg.max_steps
    warmup_iters = cfg.warmup_steps
    learning_rate = cfg.max_lr
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters

    if step > max_iters:
        return cfg.min_lr

    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (learning_rate - cfg.min_lr)

class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_dataset = EvalDataset(cfg)
        self.dataset = CustomDataset(cfg)
        self.model = Transformer(cfg).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=cfg.learning_rate)
        self.vocab_size = len(self.dataset.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=cfg.batch_size, shuffle=False)
        self.epochs = cfg.epochs


        #Why is muon like this? i don't want to anymore
        '''
        muon_params = []
        adamw_params = [] 
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_embed_or_head = any(x in name for x in ['wte', 'wpe', 'embed', 'classifier', 'output'])

            if param.ndim == 2 and not is_embed_or_head:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self.muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
        self.adamw_opt = AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
        '''


        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        os.makedirs("checkpoints", exist_ok=True)
        self.checkpoint_path = f"checkpoints/{cfg.model_name}_{timestamp}.pth"

        #self.fixed_seq = self.data()

        print(f"Starte Training auf {self.device}...")
        self.model.train()

    def train(self):
        global_step = 0
        for epoch in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)


                lr = get_lr(global_step, cfg)
                for param_group in self.optimizer.param_groups: # bzw. adamw_opt / muon_opt
                    param_group['lr'] = lr

                #self.muon_opt.zero_grad()
                #self.adamw_opt.zero_grad()
                self.optimizer.zero_grad()

                #print(f"Min tokenid {batch_x.min().item()} max id {batch_x.max().item()}")
                

                logits = self.model(batch_x)

                loss = self.criterion(logits.view(-1, self.vocab_size), batch_y.reshape(-1))

                loss.backward()
                self.optimizer.step()

                #self.muon_opt.step()
                #self.adamw_opt.step()
                global_step += 1

            if(epoch +1) % 1 == 0:
                print(f"Epoch {epoch+1:3d}/3 | Loss {loss.item():.4f} {datetime.now().strftime('%H:%M:%S')} at lr: {lr:.3f}")
        
        print(f"\nFinished Training! at {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        print("saving...")
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"saved at {self.checkpoint_path}")

    def eval(self):
        self.model.eval()

        correct = 0
        total = 0
        losses = []
        print("evaluation")
        with torch.no_grad():
            for batch_x, batch_y in self.eval_dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits.view(-1, self.vocab_size), batch_y.reshape(-1))

                losses.append(loss.item())
                #print(f"Loss. {loss.item():.4f}")
                

        avg_loss = statistics.fmean(losses)
        print(f"Avg Loss: {avg_loss:.4f}")
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"PPL: {ppl:.2f}")
                
        

if __name__ == "__main__":
    cfg = ModelConfig()
    t = Trainer(cfg)
    t.train()
    t.eval()
