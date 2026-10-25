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
    model_name: str = "Emilia"

    #Training praram
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_samples: int = 100000
    batch_size: int = 32
    epochs: int = 3



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
    max_steps: int = 101
    

