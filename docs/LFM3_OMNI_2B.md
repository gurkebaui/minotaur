# LFM3-Omni-2B — Architektur-Referenz

Stand: November 2026 · Modell: [`LiquidAI/LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B) ·
Lizenz: LFM Open License v1.0

LFM3-Omni-2B ist das multimodale Mitglied der LFM3-Familie: **Text, Bild, Video und
Audio rein, Text und Sprache raus** — ein einziges end-to-end Modell für On-Device.
Diese Dokumentation beschreibt die Architektur aus `config.json`,
`modeling_lfm3_omni.py`, `modeling_lfm3_vl.py`, `modeling_liquid_codec.py` und den
Konfigurationsdateien. Alle Parameterzahlen stammen aus dem `model.safetensors`-Header
(2.1669 Mrd. Parameter) und sind mit den Shapes im Code abgeglichen.

> Kurzfassung: Ein **LFM3-1.7B-Text-Tower** (identisch zu `LFM3-1.7B-Base`, nur 128K
> Kontext) bekommt vier Anhängsel:
> 1. einen **Vision-Tower** (LFM3-VL, 14 Layer) + **Bild-Projektor**,
> 2. einen **Audio-Encoder** (Liquid, 17 Layer) + **Audio-Projektor**,
> 3. ein **Codebook-Embedding** für eingehende Sprach-Frames,
> 4. einen **Depth-Decoder** (3 Layer), der pro Sprach-Frame 12 Codebooks vorhersagt.

---

## 1. Überblick

| Eigenschaft | Wert |
|---|---|
| Parameter gesamt | 2.1669 Mrd. (BF16, Datei 4.33 GB) |
| Text-Tower | LFM3-1.7B, 24 Layer, 128K Kontext |
| Vision-Tower | 14 Layer, hidden 1024, Patch 16, Merge 2 |
| Audio-Encoder | 17 Layer, hidden 768, 128 Mel-Bins, Fenster 32 |
| Depth-Decoder | 3 Layer, hidden 1024, 12 Codebooks × 2017 Codes |
| Codec | LiquidCodec (24 kHz, 12 Codebooks, 1920 Samples/Frame) |
| Eingaben | Text, Bild, Video (+ Ton), 16-kHz-Audio |
| Ausgaben | Text, 24-kHz-Sprache, interleaved Text+Sprache |
| Kontext | 32K Token |
| Vokabular | 128 000 Text-Token |

---

## 2. Komponenten im Überblick

```
 Text ──► Tokenizer ──► embed_tokens (128000×2048) ──┐
                                                     │
 Bild ──► patchifier ─► Vision-Tower ─► mm_projector ┤
                                                     ├──► LFM3-1.7B Text-Tower ──► embedding_norm ──┬──► lm_head ──► Text-Logits
 Video ─► patchifier ─► Vision-Tower ─► mm_projector ┤                                              │
                                                     │                                              └──► hidden state ──► Depth-Decoder ──► 12 Codebooks ──► LiquidCodec ──► 24 kHz
 Audio ─► 128-Mel-Spektrogram ─► Audio-Encoder ─► audio_projector ──┘
```

Die Placeholder-Token im Textstrom verbinden die Modalitäten:

| Token | ID | Bedeutung |
|---|---:|---|
| `<|audio|>` | 125 031 | eine kodierte Eingabe-Sprach-Frames |
| `<|audio_out|>` | 125 032 | eine Ausgabe-Sprach-Frames |
| `<|image|>` | 125 030 | ein projizierter Bild-Token |
| `<|video|>` | 125 029 | ein projizierter Video-Token |
| `<|audio_start|>` | 125 012 | startet den Sprach-Ausgabemodus |
| `<|text_end|>` | 125 014 | beendet Text in gemischter Ausgabe |
| `<|mixed_start|>` | 125 015 | startet gemischte Ausgabe (Text + Sprache) |

---

## 3. Text-Tower

Der Text-Tower ist baugleich mit `LFM3-1.7B-Base` (siehe
[`LFM3_1.7B_BASE.md`](LFM3_1.7B_BASE.md)), mit zwei Unterschieden:

| | LFM3-1.7B-Base | Omni-Text-Tower |
|---|---|---|
| `max_position_embeddings` | 262 144 | 128 000 |
| `tie_word_embeddings` | true | true |
| Layer / Hidden / Vocab | 24 / 2048 / 128000 | gleich |

Die Shape-Kette ist identisch: `(B, T)` Token → `(B, T, 2048)` Hidden → 24 Blöcke →
`embedding_norm` → `(B, T, 2048)`.

Im Omni-Checkpoint liegt der Text-Tower unter `model.language_model` (also
`bundle.text_model`), die Placeholder-Embeddings werden **vor** dem Text-Tower in die
`inputs_embeds` gemischt (`masked_scatter`).

---

## 4. Vision-Tower (LFM3-VL)

### 4.1 Patchifier

Bilder/Video-Frames werden in Patches von 16×16 Pixeln zerlegt, jeder Patch wird
zeitlebens 2 Frames gestapelt (`temporal_patch_size = 2`):

```
patch_dim = 3 · 2 · 16² = 1536
patchifier: Linear(1536 → 1024)   (kein Bias)
embedding_norm: RMSNorm(1024)
```

### 4.2 Layer-Typen

14 Layer im Muster `conv, conv, conv, full, conv, full, …, conv, full`:

| Typ | Anzahl | Operator |
|---|---:|---|
| `conv` | 8 | gated depthwise 2D-Conv (Kernel 3) über das Patch-Gitter |
| `full_attention` | 6 | bidirektionale GQA-Attention pro Frame |

Die Attention ist **bidirektional** und **pro Frame** (Frames schauen nicht
aufeinander) — anders als der Text-Tower, der kausal ist.

### 4.3 Attention-Details

| Eigenschaft | Wert |
|---|---|
| Köpfe / KV-Köpfe | 16 / 8 (head_dim 64) |
| QK-Norm | `q_layernorm`, `k_layernorm` (RMSNorm über 64) |
| Gate | keins (nur Text-Tower hat die sigmoidale Gate) |
| RoPE | **axial**: Hälfte der Dimensionen für y, Hälfte für x, keine partielle Rotation |
| Maskierung | keine Kausalität, Attention nur innerhalb eines Frames |

Axiale RoPE: `spatial_dim = 64 / 2 = 32` Frequenzen, `cat([freq_h, freq_w])` → 64,
dann `cat([f, f])` für die cos/sin-Tabellen.

### 4.4 Conv-Block

```python
B_, C_, x_ = in_proj(x).chunk(3, dim=-1)          # 1024 → 3072
gated = B_ * x_
# auf dem (H/2, W/2, 2, 2, C)-Gitter zurück in (B, C, H, W) falten
conv = depthwise_conv2d(gated, kernel=3, groups=1024)
y = C_ * conv
out = out_proj(y)
```

Der `spatial_merge_size = 2` wird hier ausgenutzt: die Patches sind in 2×2-Blöcken
angeordnet, der Block wird für die Faltung momentarily zu einem Bildgitter
umsortiert und danach zurück in die Token-Reihenfolge.

### 4.5 MLP und Projektor

- **MLP**: SwiGLU `1024 → 2560 → 1024`.
- **Multi-Modal-Projector** (Bild → Text-Raum):

```
LayerNorm(1024)
Linear(1024 · 2² = 4096 → 4096)   + Bias
GELU
Linear(4096 → 2048)               + Bias
```

Ein 2×2-Block von Vision-Tokens wird also zu **einem** Text-Token zusammengefasst.
Bei einem 448×448-Bild (28×28 Patches) sind das 196 Vision-Tokens → 49 Text-Token.

---

## 5. Audio-Encoder (Liquid)

### 5.1 Eingang

16-kHz-Audio → Log-Mel-Spektrogramm mit 128 Bins (`num_mel_bins = 128`). Die
Form ist `(B, 128, T_mel)`.

### 5.2 Downsampler

Drei stride-2-Operationen auf der (Mel, Zeit)-Ebene:

```
conv_in : Conv2d(1 → 256, k=3, s=2, p=1)
conv_mid: Conv2d(256 → 256, k=3, s=2, p=1, groups=256)   (depthwise)
lin_mid : Conv2d(256 → 256, k=1)                          (pointwise)
conv_end: Conv2d(256 → 256, k=3, s=2, p=1, groups=256)
lin_end : Conv2d(256 → 256, k=1)
lin_out : Conv1d(128 · 256 / 8 = 4096 → 768, k=1)
```

Nach den drei Schritten ist die Zeitachse um 8 reduziert und die Mel-Achse von 128 auf
16. Das Ergebnis: **ein Vektor pro 8 Mel-Frames** (Shape `(B, T_mel/8, 768)`).

### 5.3 Encoder (LFM2-Style)

17 Layer, hidden 768, 8 Köpfe / 4 KV-Köpfe, `intermediate_size = 2560`,
`conv_L_cache = 3`, `sliding_window = 32`.

| Layer-Typ | Anzahl |
|---|---:|
| `conv` | 10 |
| `full_attention` | 7 |

Muster: `[conv, conv, full] × 5 + [full, full]`. Die Attention-Layer haben ein
Schiebefenster von 32 Frames (also ~32 · 8 = 256 Mel-Frames ≈ 5 s Kontext), was für
Streaming genügt.

### 5.4 Audio-Projektor

```
LayerNorm(768) → Linear(768 → 2048) → GELU → Linear(2048 → 2048)
```

Jeder Audio-Token landet damit als 2048-Vektor an der `<|audio|>`-Position.

---

## 6. Depth-Decoder (Sprachausgabe)

Sprache wird nicht direkt als Token, sondern als **12 Codebooks pro Frame**
dargestellt. Der Depth-Decoder sagt pro Frame alle Codebooks gemeinsam voraus.

### 6.1 Eingang

Der Hidden-State des Text-Towers an der Position vor dem `<|audio_out|>`-Frame wird
linear auf 12 Slots projiziert:

```
depth_linear: Linear(2048 → 12 · 1024 = 12288)   + Bias
```

Pro Slot wird entweder das bekannte Code-Embedding addiert oder (für maskierte
Codebooks) ein gelerntes `mask_embedding` (1024):

```
depth_in = depth_linear(h).unflatten(-1, (12, 1024))
       + where(masked, mask_embedding, codebook_embedding(codes))
```

### 6.2 Trunk

Ein **bidirektionaler** LFM3 mit 3 Layern über die 12 Slots:

| Eigenschaft | Wert |
|---|---|
| Hidden / Layer | 1024 / 3 |
| Köpfe / KV | 32 / 8 (head_dim 32) |
| FFN | 2816 |
| Attention | `full_attention`, bidirektional, `partial_rotary_factor = 1.0` |
| Max. Positionen | 12 (= Anzahl Codebooks) |
| Sandwich-Norm | nein |
| Cache | nein |

Bidirektional, weil alle Codebooks eines Frames *gemeinsam* aus dem Hidden-State
rekonstruiert werden; die Reihenfolge ist die Codebook-Achse, nicht die Zeit.

### 6.3 MaskGIT-Sampling

Beim Generieren wird pro Frame in mehreren Durchläufen dekodiert
(`maskgit_schedule = [4, 4, 4]`):

1. Durchlauf: 4 Codebooks vorhersagen, 8 maskiert lassen.
2. Durchlauf: weitere 4 vorhersagen (die bekannten Codebooks als Embedding).
3. Durchlauf: die letzten 4.

Jeder Codebook-Head ist ein eigenes Embedding mit RMSNorm:

```
logits_k = F.linear(rms_norm(h_k), codebook_embedding_k.weight)    # 1024 → 2017
```

### 6.4 Ausgabe

Die 12 Codes eines Frames werden zu einem 2048-Vektor summiert
(`codebook_embedding`, `12 · 2017 = 24 204` Zeilen) und als `<|audio_out|>`-Token in
den Textstrom geschrieben. Der LiquidCodec dekodiert diese Codes zu 24-kHz-Wellenform.

---

## 7. LiquidCodec (Sprach-Tokens)

Der Codec liegt separat unter `audio_tokenizer/` (147 MB, BF16):

| Eigenschaft | Wert |
|---|---|
| Samplerate | 24 000 Hz |
| STFT | `n_fft = 1280`, `hop = 320`, `win = 1280` |
| Frame-Stacking | 6 STFT-Frames pro Codec-Frame → 1920 Samples (80 ms, 12.5 Frames/s) |
| Encoder | LFM3, hidden 512, conv-dim 256 |
| Quantizer | FSQ mit `[8, 7, 6, 6]` → 2016 Codes + 1 EOS = 2017 |
| Codebooks | 12 (`n_q = 12`) |
| Decoder | LFM3, hidden 512, Sliding-Window 30 Frames, iSTFT |

`encode`: Wellenform → STFT → Frontend → Encoder → FSQ → `(B, 12, T_frames)`.
`decode`: Codes → FSQ-Decoder → LFM3-Decoder → iSTFT → Wellenform.

Der Codec ist **nicht** Teil des Omni-`state_dict`; er wird bei Bedarf separat aus
`audio_tokenizer/` geladen.

---

## 8. Token-Layout und Shapes

Beispiel „Bild + Frage + Sprachantwort":

```
Text-Prompt:  <|startoftext|> ... <|image|> <|image|> ... "What is this?" ...
                                │        │
                                │        └─ 49 projizierte Bild-Token (28×28 Patches / 2²)
                                └─ Placeholder, werden per masked_scatter ersetzt

Embedding-Stream (B, T, 2048):
  Text-Token       → embed_tokens
  <|image|>        → multi_modal_projector(vision_tower(pixel_values))
  <|audio|>        → audio_projector(audio_encoder(mel))
  <|audio_out|>    → sum(codebook_embedding(codes))

Text-Tower → hidden (B, T, 2048) → lm_head → Text-Logits
                                  → depth_decoder → 12 Codebooks → LiquidCodec → 24 kHz
```

Der Omni-Forward (`Lfm3OmniModel.forward`) macht genau das:

1. `inputs_embeds = embed_tokens(input_ids)`
2. Bild- und Video-Features per `masked_scatter` in die Placeholder-Positionen schreiben.
3. Audio-Features (Encoder + Projektor) in die `<|audio|>`-Positionen schreiben.
4. `audio_out_codes` (Summe der Codebook-Embeddings) in die `<|audio_out|>`-Positionen.
5. Text-Tower.
6. `Lfm3OmniForConditionalGeneration` rechnet Text-Logits (LM-Kopf) und optional
   Sprach-Logits (Depth-Decoder).

---

## 9. Generierungsmodi

| Modus | Start-Token | Verhalten |
|---|---|---|
| `TEXT` | — | nur Text |
| `AUDIO` | `<|audio_start|>` (125 012) | Text-Tower → Depth-Decoder → Codec, bis Audio-EOS-Code 2016 |
| `INTERLEAVED` | `<|mixed_start|>` (125 015) | 6 Text-Token, dann 12 Sprach-Frames, wiederholt |

`interleaved_n_text = 6`, `interleaved_n_audio = 12` sind die Blockgrößen im
gemischten Modus.

---

## 10. Parameter-Budget

| Komponente | Parameter | Anteil |
|---|---:|---:|
| Text-Tower (`model.language_model`) | 1 728.80 M | 79.78 % |
| Vision-Tower (`model.vision_tower`) | 164.20 M | 7.58 % |
| Audio-Encoder (`model.audio_encoder`) | 109.51 M | 5.05 % |
| Depth-Decoder (`model.depth_decoder`) | 83.90 M | 3.87 % |
| Codebook-Embedding | 49.57 M | 2.29 % |
| Multi-Modal-Projector | 25.17 M | 1.16 % |
| Audio-Projektor | 5.77 M | 0.27 % |
| **Summe** | **2 166.93 M** | 100 % |

Der Text-Tower ist also ~80 % des Modells; alle Modalitätsbausteine zusammen ~20 %.
Das ist der Grund, warum Omni auf dem gleichen Edge-Budget wie das Textmodell bleibt.

---

## 11. Anknüpfungspunkte für Experimente

| Ziel | Wo | API |
|---|---|---|
| Text-Tower allein nutzen | `model.model.language_model` | `bundle.text_model`, `bundle.access` |
| Vision-Features sehen | `model.model.vision_tower` | `get_image_features(pixel_values, grid_thw)` |
| Bild-Projektor ändern | `model.model.multi_modal_projector` | direktes Surgery |
| Audio-Encoder ändern | `model.model.audio_encoder` | `get_audio_features(...)` |
| Depth-Decoder ändern | `model.model.depth_decoder` | `depth_decoder(h, codes, masked)` |
| Codebook-Embedding | `model.model.codebook_embedding` | `get_audio_out_embeds(codes)` |
| Codec einzeln | `audio_tokenizer/` | `LiquidCodecModel.from_pretrained(...)` |
| Text-Experimente | wie 1.7B | `lfm3.access`, `lfm3.modules` |

Für reine Text-Forschung ist der Omni-Checkpoint meist die falsche Wahl (4.33 GB statt
3.46 GB, 128K statt 256K Kontext). Für multimodale Forschung ist er der Einstieg.

---

## 12. Referenzen

- Modellkarte: <https://huggingface.co/LiquidAI/LFM3-Omni-2B>
- Code: `modeling_lfm3_omni.py`, `modeling_lfm3_vl.py`, `modeling_liquid_codec.py`,
  `configuration_lfm3_omni.py`, `configuration_lfm3_vl.py`,
  `configuration_liquid_codec.py`
- Text-Tower: [`LFM3_1.7B_BASE.md`](LFM3_1.7B_BASE.md)
- Labor-API: `docs/LFM3_LAB_GUIDE.md`