import torch
from config import DiTConfig

class EulerScheduler:
    def __init__(self, cfg):
        self.num_train_timesteps = cfg.num_train_timesteps

    def add_noise(self, x_0, noise, t_idx):
        t = t_idx.float() / self.num_train_timesteps
        t = t.view(-1, 1, 1, 1) #[B, C, H, W]
        x_t = (1.0 - t) * x_0 + t * noise #interpolieren (flow matching)
        return x_t, t

    def get_inference_timesteps(self, cfg):
        return torch.linspace(1.0, 0.0, cfg.num_steps)