# minotaur

Sammlung von PyTorch-Experimenten: ein kleines Transformer-Training
(`model.py`, `train.py`, `generate.py`), ein DiT-Sampler (`DiT.py`) und ein
Forschungs-Labor für die LFM3-Modelle von Liquid AI.

##Skripte

| Datei | Inhalt |
|---|---|
| `model.py`, `train.py`, `generate.py` | kleiner GPT-Style-Transformer auf TinyStories |
| `DiT.py`, `train_DiT.py`, `sample.py` | Diffusion-Transformer |
| `data.py`, `eval_data.py` | Datensätze |
| `config.py` | zentrale Dataclass-Configs |
| `checkpoints/` | gespeicherte Läufe (gitignored) |

## Referenzen

- **DiffusionBlocks Paper**: Makoto Shing, Masanori Koyama, Takuya Akiba (ICLR 2026). *DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation*. [arXiv:2506.14202](https://arxiv.org/abs/2506.14202)
- **Offizielles Repository (ViT & DiT)**: [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks)
