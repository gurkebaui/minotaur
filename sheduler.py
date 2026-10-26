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

    def sample_step(self, x_t, predicted_noise, t_idx, prev_t_idx):
        #alpha_t, sigma_t = self.get_coefficients(t_idx)
        t = t_idx.float() / self.num_train_timesteps
        t_prev = prev_t_idx.float() / self.num_train_timesteps
        t = t.view(-1, 1, 1, 1)
        t_prev = t_prev.view(-1, 1, 1, 1)
        x_0 = (x_t - predicted_noise * t) / ((1 - t) + 1e-8)
        x_prev = (1 - t_prev) * x_0 + predicted_noise * t_prev
        return x_prev
