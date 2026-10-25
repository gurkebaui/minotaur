import torch
from config import DiTConfig


class EulerScheduler:
    def __init__(self, cfg):
        self.num_train_timesteps = cfg.num_train_timesteps

    def add_noise(self, x_0, noise, t_idx):
        #Shape [B, C, H, W]
        t = t_idx.float() / self.num_train_timesteps
        t= t.view(-1, 1, 1, 1)
        x_t = (1 - t) * x_0 + noise * t
        return x_t, noise
