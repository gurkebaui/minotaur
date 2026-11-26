# Autoregressiver Transformer mit DiffusionBlocks

Eine kompakte, leicht verständliche PyTorch-Implementierung des **Autoregressive (AR) Transformer** aus Sakana AIs Paper *„DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation“* (Abschnitte 3, 5.4, Appendix B, C und E.4).

## Dateien

| Datei | Inhalt |
| :--- | :--- |
| [`noise_schedule.py`](./noise_schedule.py) | Equi-Probability Noise-Partitionierung, Überlappung $\gamma$, EDM-Preconditioning & Gewichtung |
| [`ar_transformer.py`](./ar_transformer.py) | Llama-2-Style AR-Transformer mit AdaLN-Zero, RoPE, $2S \times 2S$-Maske, Training & Euler-Generierung |
| [`train_demo.py`](./train_demo.py) | 4 automatische Verifikationstests + kurzes Trainings- & Textgenerierungs-Beispiel |
| [`train_tinystories.py`](./train_tinystories.py) | Echtes Training auf `roneneldan/TinyStories` (BPE-Tokenizer, bf16, Validation & Generierung) |

## Schnellstart

```bash
# Tests + Mini-Training (1200 Schritte) + Textgenerierung
python3 ar_diffusion_blocks/train_demo.py

# Nur die 4 mathematischen Verifikationstests ausführen
python3 ar_diffusion_blocks/train_demo.py --only-verify

# Kleines Modell (~17M Parameter) auf TinyStories trainieren
python3 ar_diffusion_blocks/train_tinystories.py --steps 8000

# Gespeicherten TinyStories-Checkpoint zum Generieren nutzen
python3 ar_diffusion_blocks/train_tinystories.py --generate-only --prompt "One day, a little cat "
```

## Wie es funktioniert (kurz erklärt)

1. **Block-Aufteilung & Equi-Probability Noise-Bereiche:**
   Die $L$ Transformer-Layer werden in $B$ Blöcke geteilt (Standard für AR: $B=4$). Jeder Block bekommt per inverser Log-Normal-CDF genau $1/B$ der Wahrscheinlichkeitsmasse aus $[\sigma_{\min}, \sigma_{\max}] = [0.002, 80.0]$.

2. **Modifizierte $2S \times 2S$-Attention-Maske (`build_training_mask`):**
   Beim Training werden saubere Kontext-Tokens $x = (t_0, \dots, t_{S-1})$ und verrauschte Ziel-Embeddings $z_\sigma = \text{norm\_emb}(y) + \sigma \epsilon$ konkateniert:
   ```text
   clean -> clean : kausal (untere Dreiecksmatrix)
   clean -> noisy : komplett blockiert (kein Zukunfts-Leakage)
   noisy -> clean : kausal (noisy_i sieht nur clean_0..clean_i)
   noisy -> noisy : nur Diagonale (noisy_i sieht nur sich selbst)
   ```

3. **Training vs. Inferenz:**
   - **Training (`training_loss`):** Wähle zufällig einen Block $b$, sample $\sigma$ aus Block $b$'s Intervall (mit Überlappung $\gamma = 0.1$) und aktualisiere **nur** `model.blocks[b]`.
   - **Inferenz (`generate`):** Für jedes neue Token startet $z_0$ als Rauschen und durchläuft in 4 Euler-Schritten die Blöcke $0 \to 1 \to 2 \to 3$ von $\sigma_{\max}$ nach $\sigma_{\min}$.

## Wichtige Hyperparameter (`ARConfig`)

| Parameter | Standard | Bedeutung |
| :--- | :--- | :--- |
| `num_blocks` | `4` | Anzahl unabhängiger Blöcke (Paper: $B=4$ für AR-Modelle) |
| `gamma` | `0.1` | Überlappung der Noise-Intervalle im Log-Raum (glattere Übergänge) |
| `sigma_data` | `0.5` | EDM-Datenskala für Preconditioning und Loss-Gewichtung |
| `num_steps` | `4` | Anzahl Euler-Denoising-Schritte pro generiertem Token |
