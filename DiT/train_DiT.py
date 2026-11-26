import torch
import torch.nn.functional as F
import torch.nn as nn
from config import DiTConfig
import torch.optim as optim
from DiT import DiT
from sheduler import EulerScheduler

def train_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DiTConfig()

    model = DiT(cfg).to(device)
    scheduler = EulerScheduler(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    model.train()

    print(f"Starte Training auf {device}")

    for step in range(cfg.max_steps):
        optimizer.zero_grad()

        x_0 = torch.randn(4, cfg.in_channels, cfg.H, cfg.W, device=device)
        batch_size = x_0.shape[0]

        t_idx = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
        print(f"t_idx: {t_idx}")
        
        noise = torch.randn_like(x_0)

        x_t, target_noise = scheduler.add_noise(x_0, noise, t_idx)

        predicted_noise = model(x_t, t_idx)

        loss = F.mse_loss(predicted_noise, target_noise)

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")

if __name__ == "__main__":
    train_loop()
