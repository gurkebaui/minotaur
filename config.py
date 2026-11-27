from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Konfiguration für den Standard-/Ludcy-Transformer und dessen Training."""

    # Architektur
    vocab_size: int = 50257
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 64
    seq_len: int = 64
    model_name: str = "ludcy_small"
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    conv_kernel_size: int = 7

    # Daten & Tokenizer
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    max_samples: int = 10000

    # Training & Optimizer
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 3
    max_steps: int = 10000
    warmup_steps: int = 200
    max_lr: float = 1e-1
    min_lr: float = 1e-5
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.90, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Generierung
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    checkpoint_path: str = "checkpoints/ludcy_small_2026-11-23_16-21.pth"


@dataclass
class DiTConfig:
    """Konfiguration für den Bild-DiT und den Euler-Scheduler."""

    # Architektur
    d_model: int = 128
    max_period: int = 10000
    patch_size: int = 4
    in_channels: int = 3
    W: int = 32
    H: int = 32
    num_heads: int = 8
    num_layers: int = 3
    pos_emb_std: float = 0.02

    # Diffusion & Training
    num_train_timesteps: int = 1000
    num_steps: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 21
    log_every: int = 10

    # Generator
    num_samples: int = 4
    inference_steps: int = 50


@dataclass
class ARConfig:
    """Konfiguration für den Autoregressiven DiffusionBlocks-Transformer."""

    # Architektur & Embeddings
    vocab_size: int = 50257
    max_seq_len: int = 64
    dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    num_blocks: int = 4
    mlp_dim: int | None = None
    dropout: float = 0.0
    freq_dim: int = 128
    max_period: float = 10000.0
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02

    # Latent MoE
    num_experts: int = 16
    top_k: int = 2
    moe_latent_dim: int | None = None
    moe_expert_dim: int | None = None
    moe_shared_dim: int | None = None
    moe_aux_loss_coef: float = 0.01

    # Noise-Schedule & EDM
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2
    gamma: float = 0.1

    # Training & Optimizer
    seed: int = 42
    batch_size: int = 64
    steps: int = 8000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    min_warmup_steps: int = 10
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Daten & Evaluation
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    num_train_stories: int = 50_000
    num_val_stories: int = 1_000
    bpe_vocab_size: int = 4096
    eval_batches_per_block: int = 8
    log_every: int | None = None
    val_every: int | None = None

    # Inferenz / Generierung
    inference_steps: int | None = None
    temperature: float = 0.0
    max_new_tokens: int = 120

    def __post_init__(self) -> None:
        if self.num_layers % self.num_blocks != 0:
            raise ValueError(
                f"num_layers ({self.num_layers}) muss durch num_blocks ({self.num_blocks}) teilbar sein."
            )
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) muss durch num_heads ({self.num_heads}) teilbar sein."
            )
        if self.mlp_dim is None:
            # Standard-SwiGLU-Größe (~8/3 * dim), auf Vielfaches von 32 gerundet
            raw = int(self.dim * 8 / 3)
            self.mlp_dim = ((raw + 31) // 32) * 32
        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError(
                f"top_k ({self.top_k}) muss zwischen 1 und num_experts ({self.num_experts}) liegen."
            )
        if self.moe_latent_dim is None:
            self.moe_latent_dim = max(16, self.dim // 2)
        if self.moe_expert_dim is None:
            raw_exp = int(self.moe_latent_dim * 8 / 3)
            self.moe_expert_dim = ((raw_exp + 31) // 32) * 32
        if self.moe_shared_dim is None:
            self.moe_shared_dim = self.moe_expert_dim
        if self.inference_steps is None:
            self.inference_steps = self.num_blocks
