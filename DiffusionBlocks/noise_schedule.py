"""
Noise-Schedule und EDM-Mathematik für DiffusionBlocks.

Enthält:
- Equi-Probability Partitioning der Log-Normal-Verteilung (Abschnitt 3.3 im Paper)
- Überlappende Noise-Intervalle mit Faktor gamma (Appendix C)
- EDM Preconditioning & Loss-Gewichtung (Karras et al., 2022 / Appendix C)
"""

import numpy as np
from scipy.stats import norm
import torch


def get_block_sigmas(
    num_blocks: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    """
    Teilt [sigma_min, sigma_max] per Equi-Probability Partitioning in `num_blocks` Intervalle.

    Jeder Block bekommt exakt 1/B der gesamten Wahrscheinlichkeitsmasse der
    Log-Normal-Verteilung log(sigma) ~ N(p_mean, p_std^2).
    Rückgabe in absteigender Reihenfolge (Inferenz-Reihenfolge):
        [sigma_0 = sigma_max, sigma_1, ..., sigma_B = sigma_min]
    Block b ist zuständig für [block_sigmas[b + 1], block_sigmas[b]].
    """
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)

    block_sigmas = []
    for i in range(num_blocks + 1):
        # Von cdf_max (hohes Rauschen) absteigend bis cdf_min (wenig Rauschen)
        p = cdf_max - (cdf_max - cdf_min) * (i / num_blocks)
        sigma = float(np.exp(p_mean + p_std * norm.ppf(p)))
        block_sigmas.append(sigma)
    return block_sigmas


def get_inference_sigmas(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Erzeugt den diskreten Denoising-Schedule für die Inferenz (absteigend von sigma_max nach sigma_min).
    """
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    cdf_points = np.linspace(cdf_max, cdf_min, num_steps)
    sigmas = np.exp(p_mean + p_std * norm.ppf(cdf_points))
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def sample_sigma_for_block(
    block_idx: int,
    batch_size: int,
    block_sigmas: list[float],
    gamma: float = 0.1,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Sampelt `batch_size` Noise-Level sigma aus dem Intervall von Block `block_idx`.

    Mit gamma > 0 wird das Intervall im Log-Raum leicht erweitert (Appendix C),
    damit die Übergänge zwischen benachbarten Blöcken glatter werden.
    """
    sigma_high = block_sigmas[block_idx]
    sigma_low = block_sigmas[block_idx + 1]
    global_max = block_sigmas[0]
    global_min = block_sigmas[-1]

    if gamma > 0.0:
        log_low = np.log(sigma_low)
        log_high = np.log(sigma_high)
        log_range = log_high - log_low
        sigma_low = max(float(np.exp(log_low - gamma * log_range)), global_min)
        sigma_high = min(float(np.exp(log_high + gamma * log_range)), global_max)

    cdf_low = norm.cdf((np.log(sigma_low) - p_mean) / p_std)
    cdf_high = norm.cdf((np.log(sigma_high) - p_mean) / p_std)

    u = np.random.uniform(cdf_low, cdf_high, size=batch_size)
    sigmas = np.exp(p_mean + p_std * norm.ppf(u))
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def get_block_index(sigma: float | torch.Tensor, block_sigmas: list[float]) -> int:
    """
    Findet den Block-Index b in [0, B-1], in dessen Intervall [sigma_{b+1}, sigma_b] das gegebene sigma liegt.
    """
    s = float(sigma.item() if isinstance(sigma, torch.Tensor) else sigma)
    num_blocks = len(block_sigmas) - 1

    if s >= block_sigmas[0]:
        return 0
    if s <= block_sigmas[-1]:
        return num_blocks - 1

    for b in range(num_blocks):
        if block_sigmas[b + 1] <= s <= block_sigmas[b]:
            return b
    return num_blocks - 1


def edm_preconditioning(
    sigma: torch.Tensor, sigma_data: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Berechnet die EDM-Skalierungsfaktoren (Karras et al., 2022) für stabiles Training über alle Noise-Level.

    Rückgabe: (c_skip, c_out, c_in, c_noise)
    """
    sigma_sq = sigma**2
    data_sq = sigma_data**2
    denom = torch.sqrt(sigma_sq + data_sq)

    c_skip = data_sq / (sigma_sq + data_sq)
    c_out = sigma * sigma_data / denom
    c_in = 1.0 / denom
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def edm_loss_weight(
    sigma: torch.Tensor, sigma_data: float = 0.5, max_weight: float | None = 10.0
) -> torch.Tensor:
    """
    EDM Loss-Gewichtung w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.
    Gleicht den Sampling-Bias der Log-Normal-Verteilung aus (geclippt gegen Low-Noise-Spikes bei CE).
    """
    w = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    return w.clamp(max=max_weight) if max_weight is not None else w
