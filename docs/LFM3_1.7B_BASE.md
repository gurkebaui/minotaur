# LFM3-1.7B-Base — Architektur-Referenz

Stand: November 2026 · Modell: [`LiquidAI/LFM3-1.7B-Base`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) ·
Lizenz: LFM Open License v1.0

Diese Dokumentation beschreibt die Architektur **so, wie sie im Checkpoint steckt**
(`config.json`, `modeling_lfm3.py`, `lfm3_conv.py`). Alle Zahlen sind aus dem
Checkpoint bzw. der Config verifiziert; die Parameterzahlen stammen aus
`lfm3.inspect.parameter_budget()` und stimmen mit den 1.7288 Mrd. Parametern des
tatsächlichen `state_dict` überein.

> Kurzfassung: LFM3-1.7B-Base ist ein Decoder-only Transformer mit **tri-hybrider**
> Schichtung — 13 gated Short-Convolutions, 5 Sliding-Window-Attention-Layer und
> 6 Full-Attention-Layer. Dazu kommen QK-RMSNorm, eine sigmoidale Output-Gate pro Kopf,
> partielle RoPE (nur die Hälfte jeder Kopfdimension) und Sandwich-Norms.

---

## 1. Überblick

| Eigenschaft | Wert |
|---|---|
| Checkpoint | Pre-trained Text-Base (kein Instruction-Tuning) |
| Parameter | 1.7288 Mrd. (BF16, Datei 3.46 GB) |
| Architektur | 24 Layer: 13 × `conv`, 5 × `sliding_attention`, 6 × `full_attention` |
| Kontext | 262 144 Token (256K), trainiert auf 22 Trd. Token |
| Vokabular | 128 000 Embedding-Zeilen (Tokenizer: 125 033 Tokens) |
| Gewichtstyp | `bfloat16` |
| Embedding-Tying | `tie_word_embeddings: true` (Input = Output) |
| Attention-Backends | `sdpa` (Default), `eager`, `flash_attention_2`, `flex_attention` |
| Floats | RMSNorm-Varianz in FP32, RoPE in FP32, Softmax in FP32 |

Warum „tri-hybrid"? Die Schichten mischen drei Operatoren:

1. **Full Attention** (global, 6×) — sieht jedes Token.
2. **Sliding-Window Attention** (lokal, 5×) — sieht die letzten 256 Token.
3. **Gated Short Convolution** (lokal, 13×) — sieht die letzten 7 Token, ist aber
   rekurrent und braucht keinen KV-Cache, nur einen kleinen Fester-Zustand.

Das ist die Kernidee: globale Abhängigkeiten sparsam (nur 6 Layer), lokale Arbeit
günstig (18 Layer). Für Edge-Geräte ist das der Unterschied zwischen „KV-Cache pro Token"
und „333 KB Fixzustand".

---

## 2. Architektur im Überblick

```
                       ┌───────────────────────────────┐
  token ids  (B, T)    │ embed_tokens  128000 × 2048   │  (tied)
                       └───────────────┬───────────────┘
                                       │ (B, T, 2048)
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Block 0   conv            ┐                                  │
        │ Block 1   conv             │ 13 × gated short conv            │
        │ Block 2   full_attention   │                                  │
        │ Block 3   conv             │ 5 × sliding attention            │
        │ Block 4   sliding_attn     │                                  │
        │ Block 5   conv             │ 6 × full attention               │
        │ ...                        │                                  │
        │ Block 22  full_attention  ┘                                  │
        │ Block 23  conv                                               │
        └──────────────────────────────┬───────────────────────────────┘
                                       │ (B, T, 2048)
                                       ▼
                       ┌───────────────────────────────┐
                       │ embedding_norm  (RMSNorm)     │  ← "final norm"
                       └───────────────┬───────────────┘
                                       │ (B, T, 2048)  = "hidden state vor dem Output-Embedding"
                                       ▼
                       ┌───────────────────────────────┐
                       │ lm_head  2048 → 128000        │  (weight == embed_tokens.weight)
                       └───────────────┬───────────────┘
                                       ▼
                                 logits (B, T, 128000)
```

Jeder Block sieht intern so aus (Pre-Norm + Sandwich-Norm):

```
x ──► operator_norm ──► [ attention | short conv ] ──► post_operator_norm ──► (+) ──► x'
                                                                              │
x' ─► ffn_norm ──► SwiGLU-FFN ──► post_ffn_norm ──► (+) ────────────────────────► x''
```

`operator_norm` / `ffn_norm` sind RMSNorm, `post_operator_norm` / `post_ffn_norm` sind
RMSNorm, die den *Ausgang* des Operators normieren, bevor er in den Residual-Stream
addiert wird. Das ist die „Sandwich-Norm" (`use_sandwich_norm: true`) und stabilisiert
sowohl Pre-Training als auch Fine-Tuning.

---

## 3. Tokenisierung und Embedding

| Token | ID |
|---|---|
| `<|pad|>` | 124 893 |
| `<|startoftext|>` (BOS) | 124 894 |
| `<|im_end|>` | (Tokenizer) |
| EOS | 124 900 |
| `<|audio_out|>`, `<|audio|>`, `<|image|>` | 125 031 / 125 032 / 125 030 (Omni) |

Der Tokenizer (`tokenizer.json`, `tokenizer_config.json`) hat 125 033 Tokens; die
Embedding-Matrix ist auf 128 000 Zeilen aufgefüllt. Die zusätzlichen Zeilen sind nie
genutzt und werden beim Fine-Tuning oft eingefroren.

```python
ids = tokenizer("Once upon a time", return_tensors="pt").input_ids   # (1, T)
h   = model.model.embed_tokens(ids)                                  # (1, T, 2048)
```

Das Embedding ist **mit dem LM-Kopf geteilt** (`tie_word_embeddings: true`). Wenn du
eine der beiden Matrizen ersetzt, musst du das Tying beachten — `ModelAccess` macht das
explizit (`tie_head_to_embed()`, `untie_head()`).

---

## 4. Die Bausteine im Detail

### 4.1 RMSNorm

```python
variance = x.to(float32).pow(2).mean(-1, keepdim=True)
x = x * torch.rsqrt(variance + eps)     # eps = 1e-5
return weight * x.to(input_dtype)
```

Keine Bias, keine Mittelwert-Subtraktion. Die Varianz wird in FP32 gerechnet, damit
BF16-Inferenz stabil bleibt.

### 4.2 Full Attention (`full_attention`)

| Projektion | Shape | Gewicht |
|---|---|---|
| `q_proj` | 2048 → 16 × 128 = 2048 | 4.19 M |
| `k_proj` | 2048 → 4 × 128 = 512 | 1.05 M |
| `v_proj` | 2048 → 4 × 128 = 512 | 1.05 M |
| `o_proj` | 2048 → 2048 | 4.19 M |
| `gate_proj` | 2048 → 16 | 32.8 K |
| `q_norm`, `k_norm` | RMSNorm(128) | 2 × 128 |

Rechenweg:

```
q = q_norm(q_proj(x).view(B, T, 16, 128)).transpose(1, 2)     # (B, 16, T, 128)
k = k_norm(k_proj(x).view(B, T,  4, 128)).transpose(1, 2)     # (B,  4, T, 128)
v =      v_proj(x).view(B, T,  4, 128).transpose(1, 2)        # (B,  4, T, 128)

q, k = apply_rope(q, k, cos, sin)                             # partielle RoPE, FP32
k, v = repeat_kv(k, 4), repeat_kv(v, 4)                       # GQA: 4 KV-Köpfe → 16

scores = q @ k^T / sqrt(128) + causal_mask                    # (B, 16, T, T)
attn   = softmax(scores, dim=-1, dtype=float32)               # FP32-Weichmax
out    = attn @ v                                             # (B, 16, T, 128)

gate = sigmoid(gate_proj(x)).unsqueeze(-1)                    # (B, T, 16, 1)
out  = o_proj((out.transpose(1, 2) * gate).reshape(B, T, 2048))
```

Drei Dinge, die LFM3 von Llama unterscheiden:

1. **QK-Norm** auf beiden, Q und K, über `head_dim=128` — verhindert, dass einzelne
   Heads die Aufmerksamkeit aufsättigen.
2. **Sigmoidale Ausgabe-Gate pro Kopf**: `gate_proj` berechnet *einen Skalar pro Kopf*
   aus dem Block-Eingang und multipliziert ihn auf den Attention-Output. Das ist eine
   billige, dynamische „Lautstärke" pro Kopf.
3. **Partielle RoPE**: nur die erste Hälfte der 128 Dimensionen wird rotiert
   (`partial_rotary_factor = 0.5`), die zweite Hälfte bleibt positionslos. Die
   Frequenzen der rotierten Hälfte folgen trotzdem der *vollen* Dimension (keine
   gestauchte Frequenzskala), der Rest ist 0.

### 4.3 RoPE: partielle Rotation mit Zero-Tail

```python
rope_angles = int(partial_rotary_factor * head_dim) // 2      # 0.5 * 128 / 2 = 32
inv_freq    = 1 / theta ** (arange(0, 2*rope_angles, 2) / head_dim)   # 32 Werte
inv_freq    = cat([inv_freq, zeros(head_dim//2 - rope_angles)])       # + 32 Nullen

freqs = position_ids[..., None] * inv_freq                  # (B, T, 64)
emb   = cat([freqs, freqs], dim=-1)                         # (B, T, 128)
cos, sin = emb.cos(), emb.sin()
```

| Layer-Typ | `rope_theta` | `partial_rotary_factor` |
|---|---|---|
| `full_attention` | 1 000 000 | 0.5 |
| `sliding_attention` | 10 000 | 1.0 (implizit) |
| `conv` | — | — (keine RoPE) |

Die kurzen Reichweiten (Sliding) bekommen eine kleinere Basis (10 000), die globalen
Layer eine große (1 000 000). RoPE wird in FP32 gerechnet und erst danach auf BF16
zurückkonvertiert.

### 4.4 Sliding-Window Attention (`sliding_attention`)

Gleiche Aufmerksamkeit wie oben, aber mit zusätzlicher Maske:

```
erlaubt  ⟺  j ≤ i   und   j > i - sliding_window
```

Bei `sliding_window = 256` sieht ein Token sich selbst plus die 255 Vorgänger. Die
Layer haben trotzdem volle RoPE-Dimensionen (kein `partial_rotary_factor`), aber die
kleinere Theta-Basis.

### 4.5 Gated Short Convolution (`conv`)

Der Operator mit dem größten Parameteranteil (13 Layer × 63.99 M = 832 M, ~48 %).

```python
projected = in_proj(x)                     # (B, T, 3·2048) = (B, T, 6144)
B_, C_, x_ = projected.chunk(3, dim=-1)    # drei Ströme: Gate, Ausgangs-Gate, Value
h = B_ * x_                                # (B, T, 2048)  — sigmoidfrei, elementweise

# kausale, tiefenweise Faltung mit Kernel 7 (links auffüllen, rechts abschneiden)
h = conv1d(h, weight[2048, 1, 7], groups=2048)[..., :T]

y = C_ * h                                 # (B, T, 2048) — zweite Gating-Stufe
out = out_proj(y)                          # (B, T, 2048)
```

| Gewicht | Shape | Params |
|---|---|---|
| `in_proj` | 6144 × 2048 | 12.58 M |
| `conv.weight` | 2048 × 1 × 7 | 14.3 K |
| `out_proj` | 2048 × 2048 | 4.19 M |

Eigenschaften:

- **Kausal**: Position `t` sieht nur `t-6 … t`. Kein Attention-Score, keine Maskenmatrix.
- **Tiefe Faltung** (`groups=hidden_size`): jeder Kanal hat seinen eigenen 7-Tap-Filter.
- **Rekurrent**: Für die Generierung reicht ein Zustand der letzten 6 Eingaben pro Kanal
  (`conv_L_cache = 7`, 2048 × 6 × 2 Byte ≈ 24 KB pro Layer). Kein KV-Cache, der mit
  der Sequenzlänge wächst.
- **Zwei Gates**: `B_ * x_` (vor der Faltung) und `C_ * h` (nach der Faltung). Das ist
  die „gated" Variante aus LFM2.

### 4.6 SwiGLU-Feed-Forward

```python
w1: 2048 → 7680
w3: 2048 → 7680
w2: 7680 → 2048
ffn(x) = w2(silu(w1(x)) * w3(x))
```

Keine Biases. 7680 = 3.75 × 2048, nicht die klassischen 8/3·d — LFM3 skaliert das FFN
etwas breiter.

---

## 5. Forward-Pass mit Shapes

Beispiel `B = 1, T = 4`:

| Stufe | Shape | Anmerkung |
|---|---|---|
| `input_ids` | (1, 4) | Token-IDs |
| `embed_tokens` | (1, 4, 2048) | Input-Embedding |
| Block 0 … 23 | (1, 4, 2048) | 24× Operator + FFN |
| `embedding_norm` | (1, 4, 2048) | **final norm** |
| `lm_head` | (1, 4, 128000) | Logits |

`ModelAccess.forward_stages()` liefert genau diese Zwischenstände als Dict
(`embeddings`, `layers.0 … layers.23`, `final`, `logits`).

---

## 6. Parameter-Budget

| Komponente | Parameter | Anteil |
|---|---:|---:|
| `embed_tokens` (= `lm_head`) | 262.14 M | 15.16 % |
| 6 × `full_attention` Layer | 346.28 M | 20.03 % |
| 5 × `sliding_attention` Layer | 288.56 M | 16.69 % |
| 13 × `conv` Layer | 831.81 M | 48.12 % |
| `embedding_norm` (final) | 2 K | 0.00 % |
| **Summe** | **1.7288 Mrd.** | 100 % |

Pro Layer:

| Layer-Typ | Parameter | davon Operator | davon FFN |
|---|---:|---:|---:|
| `full_attention` | 57.71 M | 9.52 M | 47.19 M |
| `sliding_attention` | 57.71 M | 9.52 M | 47.19 M |
| `conv` | 63.99 M | 16.79 M | 47.19 M |

Das FFN ist mit 47.19 M pro Layer (2.7 % der Gesamtparameter pro Layer) der größte
einzelne Block — bei allen 24 Layern zusammen 1.13 Mrd. Parameter, also ~65 % des
Modells.

---

## 7. Speicherbedarf

| Posten | Größe |
|---|---|
| Gewichte BF16 | 3.46 GB |
| KV-Cache pro Token (6 Full-Attention-Layer) | 12 KB |
| KV-Cache bei 32K Kontext | ≈ 384 MB |
| KV-Cache bei 256K Kontext | ≈ 3.1 GB |
| Sliding-Layer KV-Cache | 5 × 256 Token ≈ 2.6 MB (fix) |
| Conv-State (13 Layer) | ≈ 24 KB × 13 ≈ 320 KB |
| Aktivierungen (Training, B=1, T=1024) | einige 100 MB, abhängig von `attn_implementation` |

Der entscheidende Punkt für Edge: Der KV-Cache wächst **nur mit den 6 globalen Layern**,
nicht mit 24. Die 13 Conv-Layer tragen einen konstanten Zustand von ~320 KB bei.

---

## 8. Generierung und Cache

`model.generate()` nutzt einen `DynamicCache`, der zwei Dinge hält:

1. **KV-Zustand** für die 11 Attention-Layer (Sliding-Layer begrenzt auf 256 Token).
2. **Conv-State** für die 13 Short-Convolutions (letzte 6 Eingaben pro Kanal).

Beim dekodieren mit Cache bekommt die Short-Conv-Faltung nur das neue Token; der
Zustand wird in-place weitergeschoben. Deshalb ist die `conv_L_cache`-Konfig (7) so
wichtig: sie definiert die Breite des Zustands.

Für Forschung heißt das:

- **Teacher Forcing / Training**: ein Vorwärtsdurchlauf über die ganze Sequenz.
- **Generation**: Cache benutzen, sonst zahlst du 24× die volle Faltung pro Token.
- **Packed sequences** (mehrere Dokumente in einer Zeile): LFM3 unterstützt das über
  `position_ids`, die bei 0 neu starten; die Conv-Faltung resettet dann ihren Zustand.
  Der HF-Code lehnt gepackte Prompts bei `generate()` ab, weil der Cache keine
  Dokumentgrenzen kennt — für Training aber kein Problem.

---

## 9. Anknüpfungspunkte für Experimente

| Ziel | Wo du ansetzt | API |
|---|---|---|
| Embedding tauschen / erweitern | `model.model.embed_tokens` | `acc.set_input_embeddings`, `acc.set_embedding_matrix`, `acc.resize_embeddings` |
| Output-Embedding ersetzen | `model.lm_head` | `acc.set_output_head`, `acc.attach_after_final_norm` |
| Einzelnen Block abtauschen | `model.model.layers[i]` | `acc.replace_block`, `BlockWrapper` |
| Block ablaten | `model.model.layers[i]` | `acc.bypass_block(i)` |
| Aktivierungen pro Layer | Forward-Hooks | `acc.capture()`, `acc.forward_stages()` |
| Attention verändern | `layers[i].self_attn.{q,k,v,o,gate}_proj` | direktes `nn.Linear`-Surgery |
| Short-Conv verändern | `layers[i].conv.conv.weight` | direktes Tensor-Surgery |
| N-Gramm-Schicht | vor/zwischen `embedding_norm` und `lm_head` | `lfm3.modules.NgramLayer` |
| Kleines Modell anhängen | zwischen `embedding_norm` und `lm_head` | `BottleneckAdapter`, `TinyTransformerAdapter` |
| Fine-Tuning | LoRA auf Attention | `lfm3.modules.inject_lora` |
| Reine PyTorch-Blöcke | `lfm3.pytorch_blocks` | `ReferenceBackbone.from_hf(...)`, Parity-Test |

---

## 10. Referenzen

- Modellkarte: <https://huggingface.co/LiquidAI/LFM3-1.7B-Base>
- Architektur-Code: `modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py`
  (werden via `trust_remote_code=True` geladen und liegen unter
  `~/.cache/huggingface/modules/transformers_modules/...`).
- Labor-API: `docs/LFM3_LAB_GUIDE.md`
- Omni-Modell: `docs/LFM3_OMNI_2B.md`