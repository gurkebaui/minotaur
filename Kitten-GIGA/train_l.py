from datetime import datetime
import math
import os
import statistics
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from config import ModelConfig
from data import CustomDataset
from eval_data import EvalDataset
from Transformer.ludcy import Transformer
import torch
import torch.nn as nn
from torch.optim import AdamW, Muon
from torch.utils.data import DataLoader


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


class Trainer:

  def __init__(self, cfg):
    self.cfg = cfg
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.eval_dataset = EvalDataset(cfg)
    self.dataset = CustomDataset(cfg)
    self.model = Transformer(cfg).to(self.device)
    self.criterion = nn.CrossEntropyLoss()

    self.vocab_size = len(self.dataset.tokenizer)
    self.dataloader = DataLoader(
        self.dataset, batch_size=cfg.batch_size, shuffle=True
    )
    self.eval_dataloader = DataLoader(
        self.eval_dataset, batch_size=cfg.batch_size, shuffle=False
    )
    self.epochs = cfg.epochs

    # Separate 2D hidden weights (Muon) from embeddings, head, and 1D norms/biases (AdamW)
    muon_params = []
    adamw_params = []
    for name, param in self.model.named_parameters():
      if not param.requires_grad:
        continue
      is_embed_or_head = any(
          k in name
          for k in ["tok_emb", "lm_head", "wte", "wpe", "embed", "output"]
      )

      if param.ndim == 2 and not is_embed_or_head:
        muon_params.append(param)
      else:
        adamw_params.append(param)

    self.muon_base_lr = cfg.muon_lr
    self.adamw_base_lr = cfg.adamw_lr
    self.muon_opt = Muon(muon_params, lr=self.muon_base_lr, momentum=cfg.muon_momentum)
    self.adamw_opt = AdamW(
        adamw_params, lr=self.adamw_base_lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs("checkpoints", exist_ok=True)
    self.checkpoint_path = f"checkpoints/{cfg.model_name}_{timestamp}.pth"

    print(f"Starte Training auf {self.device}...")

  def train(self):
    self.model.train()
    global_step = 0

    for epoch in range(self.epochs):
      for batch_x, batch_y in self.dataloader:
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Schedule LR while preserving the Muon / AdamW ratio
        lr = get_lr(global_step, self.cfg)
        lr_scale = lr / self.cfg.max_lr

        for param_group in self.muon_opt.param_groups:
          param_group["lr"] = self.muon_base_lr * lr_scale

        for param_group in self.adamw_opt.param_groups:
          param_group["lr"] = self.adamw_base_lr * lr_scale

        self.muon_opt.zero_grad()
        self.adamw_opt.zero_grad()

        # Unpack (logits, u) from the branched Transformer
        logits, _ = self.model(batch_x)

        loss = self.criterion(
            logits.view(-1, self.vocab_size), batch_y.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.muon_opt.step()
        self.adamw_opt.step()
        global_step += 1

      print(
          f"Epoch {epoch+1:3d}/{self.epochs} | Loss {loss.item():.4f} |"
          f" {datetime.now().strftime('%H:%M:%S')} | lr: {lr:.5f}"
      )

    print(
        "\nFinished Training! at"
        f" {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} in {global_step} Steps"
    )
    print("saving...")
    torch.save(self.model.state_dict(), self.checkpoint_path)
    print(f"saved at {self.checkpoint_path}")

  def eval(self):
    self.model.eval()
    losses = []
    print("evaluation")
    with torch.no_grad():
      for batch_x, batch_y in self.eval_dataloader:
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        logits, _ = self.model(batch_x)
        loss = self.criterion(
            logits.view(-1, self.vocab_size), batch_y.reshape(-1)
        )
        losses.append(loss.item())

    avg_loss = statistics.fmean(losses)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Avg Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")


if __name__ == "__main__":
  cfg = ModelConfig()
  t = Trainer(cfg)
  t.train()
  t.eval()
