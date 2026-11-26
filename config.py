from dataclasses import dataclass

@dataclass
class ModelConfig:
    #Architektur
    vocab_size: int = 50257
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 64
    seq_len: int = 64
    model_name: str = "ludcy_small"

    #Training praram
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_samples: int = 10000
    batch_size: int = 32
    epochs: int = 3
    max_steps: int = 10000
    warmup_steps: int = 200
    max_lr: float = 1e-1
    min_lr: float = 1e-5



@dataclass
class DiTConfig:
    num_train_timesteps: int = 1000
    num_steps: int = 20
    d_model: int = 128
    max_period: int = 10000
    patch_size: int = 4
    in_channels: int = 3
    W: int = 32
    H: int = 32
    num_heads: int = 8
    num_layers: int = 3
    lr: float = 1e-4
    max_steps: int = 21

    #generator
    num_samples: int = 4
    inference_steps: int = 50
    

@dataclass
class ARConfig:
    """Konfiguration für den Autoregressiven DiffusionBlocks-Transformer."""

    vocab_size: int = 256
    max_seq_len: int = 256
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 12
    num_blocks: int = 4
    mlp_dim: int | None = None
    dropout: float = 0.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2
    gamma: float = 0.1

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





