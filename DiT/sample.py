import torch
from DiT import DiT
from config import DiTConfig
from sheduler import EulerScheduler

class Sampler():
    def __init__(self):
        cfg = DiTConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DiT(cfg).to(self.device)
        self.scheduler = EulerScheduler(cfg)
        self.num_samples = cfg.num_samples
        self.steps = cfg.inference_steps
        self.model.eval()
        
    def sample(self, x):
        timestep = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, self.steps, device=self.device).long()
        print(f"sample {self.num_samples} Bilder in {self.steps} steps...")

        for i in range(len(timestep)-1):
            t_idx = timestep[i].expand(self.num_samples)
            prev_t_idx = timestep[i + 1].expand(self.num_samples)

            predicted_noise = self.model(x, t_idx)

            x = self.scheduler.sample_step(x, predicted_noise, t_idx, prev_t_idx)

        print("fertig")

        return x
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = Sampler()
    cfg = DiTConfig()
    
    x = torch.randn(cfg.num_samples, cfg.in_channels, cfg.H, cfg.W, device=device)
    y = s.sample(x)
    print(y)
