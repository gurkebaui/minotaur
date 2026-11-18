# minotaur

Sammlung von PyTorch-Experimenten: ein kleines Transformer-Training
(`model.py`, `train.py`, `generate.py`), ein DiT-Sampler (`DiT.py`) und ein
Forschungs-Labor für die LFM3-Modelle von Liquid AI.

## LFM3 research lab

Der Ordner `lfm3/` kapselt das Laden und Verändern von **LFM3-1.7B-Base** und
**LFM3-Omni-2B**. Ziel ist maximale Kontrolle für Experimente: Blöcke tauschen,
Embeddings ersetzen, kleine Modelle vor den LM-Kopf hängen, N-Gramm-Schichten,
LoRA-Fine-Tuning.

```bash
python lfm3_lab.py arch                    # Architekturbericht, ohne Download
python lfm3_lab.py info --budget           # Report des echten Checkpoints
python lfm3_lab.py demo --all --tiny       # alle Experimente in Sekunden (ohne Checkpoint)
python lfm3_lab.py generate --prompt "Once upon a time" --tokens 40 --sample
```

```python
from lfm3.loader import load_base

bundle = load_base()
acc = bundle.access

acc.parameter_groups()                   # wo liegen die 1.73 Mrd. Parameter?
with acc.capture() as acts:              # Aktivierungen pro Block
    bundle(input_ids)
acc.bypass_block(7)                      # Block ablaten
acc.attach_after_final_norm(adapter)     # kleines Modell vor den LM-Kopf
```

Dokumentation:

- [`docs/LFM3_1.7B_BASE.md`](docs/LFM3_1.7B_BASE.md) — Architektur des Textmodells
- [`docs/LFM3_OMNI_2B.md`](docs/LFM3_OMNI_2B.md) — Architektur des Omni-Modells
- [`docs/LFM3_LAB_GUIDE.md`](docs/LFM3_LAB_GUIDE.md) — Code-Dokumentation und Rezepte

## Weitere Skripte

| Datei | Inhalt |
|---|---|
| `model.py`, `train.py`, `generate.py` | kleiner GPT-Style-Transformer auf TinyStories |
| `DiT.py`, `train_DiT.py`, `sample.py` | Diffusion-Transformer |
| `data.py`, `eval_data.py` | Datensätze |
| `config.py` | zentrale Dataclass-Configs |
| `checkpoints/` | gespeicherte Läufe (gitignored) |
