import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from config import ARConfig
from DiffusionBlocks.noise_schedule import (
    edm_loss_weight,
    edm_preconditioning,
    get_block_index,
    get_block_sigmas,
    get_inference_sigmas,
    sample_sigma_for_block,
)
from DiffusionBlocks.Dif_trans import DiffusionBlock, LatentMoE, RMSNorm, build_training_mask, modulate
from DiffusionBlocks.Dif_tse import TimestepEmbedder, compute_rope_freqs


class ARDiffusionBlocksModel(nn.Module):
    """
    Autoregressiver Transformer mit DiffusionBlocks.

    - Im Training wird pro Schritt nur EIN Block `self.blocks[block_idx]` ausgeführt und aktualisiert.
    - In der Inferenz werden die Blöcke nacheinander von sigma_max bis sigma_min per Euler-Schritt
      durchlaufen, um das nächste Token-Embedding zu entrauschen.
    """

    def __init__(self, config: ARConfig) -> None:
        super().__init__()
        self.config = config
        layers_per_block = config.num_layers // config.num_blocks

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_embedder = TimestepEmbedder(
            config.dim, freq_dim=config.freq_dim, max_period=config.max_period
        )

        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    layers_per_block,
                    config.dim,
                    config.num_heads,
                    config.moe_expert_dim,
                    config.dropout,
                    rms_norm_eps=config.rms_norm_eps,
                    latent_dim=config.moe_latent_dim,
                    num_experts=config.num_experts,
                    top_k=config.top_k,
                    shared_dim=config.moe_shared_dim,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.final_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, 2 * config.dim))
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # RoPE bis max_seq_len + 1, da noisy_i bei Position i+1 liegt (bis max_seq_len)
        head_dim = config.dim // config.num_heads
        cos, sin = compute_rope_freqs(head_dim, config.max_seq_len + 1, base=config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.block_sigmas = get_block_sigmas(
            num_blocks=config.num_blocks,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            p_mean=config.p_mean,
            p_std=config.p_std,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        init_std = self.config.init_std
        nn.init.normal_(self.token_emb.weight, std=init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LatentMoE):
                nn.init.normal_(module.w1, std=init_std)
                nn.init.normal_(module.w3, std=init_std)
                nn.init.normal_(module.w2, std=init_std)

        # AdaLN-Zero: Modulations- und Output-Layer mit Null initialisieren (stabiler Start)
        for block in self.blocks:
            for layer in block.layers:
                nn.init.zeros_(layer.adaLN[-1].weight)
                nn.init.zeros_(layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.lm_head.weight)

    def get_normalized_embeddings(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        L2-normalisierte Token-Embeddings (verhindert Embedding-Kollaps, siehe Appendix C).
        Ohne Argument wird die gesamte normalisierte Embedding-Matrix (V, D) zurückgegeben.
        """
        if token_ids is None:
            return F.normalize(self.token_emb.weight, p=2, dim=-1)
        return F.normalize(self.token_emb(token_ids), p=2, dim=-1)

    def denoise_train_block(
        self,
        x_ids: torch.Tensor,
        z_noisy: torch.Tensor,
        sigma: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor:
        """
        Führt EINEN Block `block_idx` auf den konkatenierten [clean, noisy]-Sequenzen aus.

        x_ids:   (B, S) saubere Kontext-Tokens t_0 .. t_{S-1}
        z_noisy: (B, S, D) verrauschte Embeddings der Ziel-Tokens t_1 .. t_S
        sigma:   (B,) Noise-Level pro Batch-Element
        """
        bsz, seq_len = x_ids.shape
        device = x_ids.device

        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.config.sigma_data)
        clean_emb = self.get_normalized_embeddings(x_ids)
        noisy_in = z_noisy * c_in[:, None, None]

        # Konkatenation: erst saubere Tokens (Pos 0..S-1), dann verrauschte Tokens (Pos 1..S)
        h = torch.cat([clean_emb, noisy_in], dim=1)  # (B, 2S, D)
        pos_clean = torch.arange(seq_len, device=device)
        pos_noisy = torch.arange(1, seq_len + 1, device=device)
        positions = torch.cat([pos_clean, pos_noisy], dim=0)
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]

        cond = self.time_embedder(c_noise)
        attn_mask = build_training_mask(seq_len, device)

        # NUR der ausgewählte Block wird ausgeführt!
        h = self.blocks[block_idx](h, cond, cos, sin, attn_mask=attn_mask)

        # Nur die verrauschten Positionen (zweite Hälfte) werden zu Logits dekodiert
        h_noisy = self.final_norm(h[:, seq_len:, :])
        model_out = h_noisy * c_out[:, None, None] + z_noisy * c_skip[:, None, None]

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        logits = self.lm_head(modulate(model_out, shift, scale))
        return logits

    def training_loss(
        self, tokens: torch.Tensor, block_idx: int | None = None
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """
        Berechnet den DiffusionBlocks-Loss für einen einzelnen zufällig gewählten (oder festen) Block.

        tokens: (B, S + 1) Token-IDs. x = tokens[:, :-1], Ziel y = tokens[:, 1:].
        """
        if block_idx is None:
            block_idx = random.randrange(self.config.num_blocks)

        x_ids = tokens[:, :-1]
        y_ids = tokens[:, 1:]
        bsz, seq_len = x_ids.shape

        y_emb = self.get_normalized_embeddings(y_ids)
        sigma = sample_sigma_for_block(
            block_idx=block_idx,
            batch_size=bsz,
            block_sigmas=self.block_sigmas,
            gamma=self.config.gamma,
            p_mean=self.config.p_mean,
            p_std=self.config.p_std,
            device=tokens.device,
        )
        z_noisy = y_emb + sigma[:, None, None] * torch.randn_like(y_emb)

        logits = self.denoise_train_block(x_ids, z_noisy, sigma, block_idx)
        ce = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            y_ids.reshape(-1),
            reduction="none",
        ).view(bsz, seq_len)

        w = edm_loss_weight(sigma, self.config.sigma_data)[:, None]
        loss = (ce * w).mean()

        aux_losses = [
            layer.mlp.aux_loss
            for layer in self.blocks[block_idx].layers
            if getattr(layer.mlp, "aux_loss", None) is not None
        ]
        if aux_losses and self.config.moe_aux_loss_coef > 0:
            aux_loss = torch.stack(aux_losses).mean()
            loss = loss + self.config.moe_aux_loss_coef * aux_loss
        else:
            aux_loss = torch.zeros((), device=tokens.device)

        return loss, {
            "loss": float(loss.item()),
            "ce_loss": float(ce.mean().item()),
            "aux_loss": float(aux_loss.item()),
            "block_idx": block_idx,
        }

    def denoise_step(
        self, prefix_ids: torch.Tensor, z_t: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Einzelner Denoising-Schritt für die Inferenz: sagt das nächste Token aus
        dem sauberen Präfix `prefix_ids` (B, L) und dem verrauschten Zustand `z_t` (B, D) vorher.
        """
        device = prefix_ids.device
        seq_len = prefix_ids.shape[1]
        block_idx = get_block_index(sigma[0], self.block_sigmas)

        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.config.sigma_data)
        clean_emb = self.get_normalized_embeddings(prefix_ids)
        noisy_in = (z_t * c_in[:, None]).unsqueeze(1)  # (B, 1, D)

        h = torch.cat([clean_emb, noisy_in], dim=1)  # (B, L + 1, D)
        positions = torch.arange(seq_len + 1, device=device)
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]

        cond = self.time_embedder(c_noise)
        # Standard-kausale Attention (is_causal=True): jedes Token sieht sich und alle links davon
        h = self.blocks[block_idx](h, cond, cos, sin, attn_mask=None)

        h_last = self.final_norm(h[:, -1, :])
        model_out = h_last * c_out[:, None] + z_t * c_skip[:, None]

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        logits = self.lm_head(model_out * (1.0 + scale) + shift)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        num_steps: int | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive Textgenerierung: Für jedes neue Token wird ein `num_steps`-stufiger
        Euler-Denoising-Prozess von sigma_max nach sigma_min durchlaufen.
        """
        if num_steps is None:
            num_steps = self.config.inference_steps or self.config.num_blocks
        if temperature is None:
            temperature = self.config.temperature

        was_training = self.training
        self.eval()

        device = prompt_ids.device
        bsz = prompt_ids.shape[0]
        sigmas = get_inference_sigmas(
            num_steps=num_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            p_mean=self.config.p_mean,
            p_std=self.config.p_std,
            device=device,
        )
        norm_emb_table = self.get_normalized_embeddings()  # (V, D)

        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            context = ids[:, -self.config.max_seq_len :]

            # Startzustand: reines Gauß-Rauschen passend zur Startvarianz (1 + sigma_0^2)
            z = torch.randn(bsz, self.config.dim, device=device) * math.sqrt(
                1.0 + sigmas[0].item() ** 2
            )

            # Euler-Schritte über die Blöcke von sigma_max nach sigma_min
            for i in range(len(sigmas) - 1):
                sigma_val = sigmas[i].item()
                next_sigma_val = sigmas[i + 1].item()
                sigma = torch.full((bsz,), sigma_val, device=device)

                logits = self.denoise_step(context, z, sigma)
                probs = F.softmax(logits, dim=-1)
                denoised = probs @ norm_emb_table  # Erwartetes sauberes Embedding

                d = (z - denoised) / sigma[:, None]
                dt = next_sigma_val - sigma_val
                z = z + dt * d

            # Letzter Denoising-Schritt bei sigma_min liefert die finalen Token-Logits
            sigma_final = torch.full((bsz,), sigmas[-1].item(), device=device)
            logits = self.denoise_step(context, z, sigma_final)

            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=1)

        if was_training:
            self.train()
        return ids
