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

