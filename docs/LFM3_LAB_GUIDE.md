# LFM3-Labor — Code-Dokumentation

Diese Dokumentation erklärt **den Code in diesem Projekt** (`lfm3/` und `lfm3_lab.py`):
was jede Datei tut, welche API du für welche Aufgabe brauchst, und wie du eigene
Experimente baust. Die Architektur selbst steht in
[`LFM3_1.7B_BASE.md`](LFM3_1.7B_BASE.md) und [`LFM3_OMNI_2B.md`](LFM3_OMNI_2B.md).

---

## 1. Voraussetzungen

| Paket | Version im Projekt | Wofür |
|---|---|---|
| `torch` | 2.11.0+cu130 | alles |
| `transformers` | 5.17.0 | LFM3 lädt seinen Modellcode selbst |
| `huggingface_hub` | 1.9.0 | Checkpoint-Download |
| `datasets` | vorhanden | optionale Fine-Tuning-Datensätze |

LFM3 ist **nicht** fester Bestandteil von `transformers`. Die Checkpoints bringen ihren
Modellcode mit (`modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py`), deshalb
läuft alles mit `trust_remote_code=True`. Der Code wird beim ersten Laden nach
`~/.cache/huggingface/modules/transformers_modules/...` kopiert.

### Der eine Kompatibilitäts-Patch

Die Checkpoints wurden mit `transformers` 5.12 exportiert. Seit 5.13 prüft die
Basisklasse `rope_parameters` für **jeden** Eintrag von `layer_types` — auch für die
`conv`-Layer, die gar keine RoPE haben. Ohne Patch scheitert schon
`AutoConfig.from_pretrained` mit `KeyError: 'conv'`.

`lfm3.loader.apply_compat_shims()` installiert deshalb beim Import einen kleinen,
idempotenten Patch: fehlende Layer-Typen werden für die Dauer der Validierung als
`None` eingetragen (die Methode überspringt `None`) und danach wieder entfernt. Der
Patch ist auf LFM3-artige Konfigurationen beschränkt und global einmal pro Prozess
aktiv. Du musst dich nicht darum kümmern — `load_base()` / `load_omni()` rufen ihn auf.

---

## 2. Dateien

```
lfm3/
├── __init__.py         # öffentliche API, lazy imports (schnelles `import lfm3`)
├── config.py           # Dataclass-Configs + Presets (LFM3_1_7B_BASE, TinyLFM3, ...)
├── loader.py           # Checkpoint-Laden, Kompatibilitäts-Patch, ModelBundle
├── access.py           # ModelAccess: Struktur, Hooks, Chirurgie
├── modules.py          # n-gram, Adapter, LoRA, Probes, Block-Wrapper
├── pytorch_blocks.py   # reine PyTorch-Referenz der LFM3-Blöcke
├── inspect.py          # lesbare Architektur-Berichte
└── experiments.py      # lauffähige Demos / Smoke-Tests
lfm3_lab.py             # CLI: arch / info / demo / generate
docs/
├── LFM3_1.7B_BASE.md   # Architektur 1.7B
├── LFM3_OMNI_2B.md     # Architektur Omni
└── LFM3_LAB_GUIDE.md   # diese Datei
```

---

## 3. Quickstart

```python
from lfm3.loader import load_base

bundle = load_base()                  # lädt 3.46 GB (einmalig), danach Cache
print(bundle.describe())

ids = bundle.tokenizer("Once upon a time", return_tensors="pt").input_ids
ids = ids.to(bundle.device)

logits = bundle(ids).logits           # (1, T, 128000)
next_id = logits[0, -1].argmax()
print(bundle.tokenizer.decode([next_id]))
```

Auf der Kommandozeile:

```bash
python lfm3_lab.py arch                       # Architektur aus der Config, ohne Download
python lfm3_lab.py info --budget              # vollständiger Report des echten Modells
python lfm3_lab.py demo --all                 # alle Experimente
python lfm3_lab.py demo --all --tiny          # Sekunden, Zufallsmodell, kein Download
python lfm3_lab.py generate --prompt "Once upon a time" --tokens 40 --sample
python lfm3_lab.py info --model omni          # Omni (lädt 4.33 GB)
```

---

## 4. Konfiguration (`lfm3/config.py`)

Das Projekt nutzt an anderer Stelle Dataclasses — das Labor folgt demselben Muster. Zwei
Ebenen:

### 4.1 Architektur: `Lfm3Config`

```python
from lfm3.config import Lfm3Config, LFM3_1_7B_BASE

cfg = LFM3_1_7B_BASE
cfg.hidden_size            # 2048
cfg.num_hidden_layers      # 24
cfg.resolved_layer_types() # ['conv', 'conv', 'full_attention', ...]
cfg.counts()               # {'conv': 13, 'sliding_attention': 5, 'full_attention': 6}
cfg.parameter_estimate()   # {'embeddings': 262144000, 'layers': 1466655488, 'total': 1728801536}
```

Felder, die du am häufigsten ändern wirst:

| Feld | Default | Wirkung |
|---|---|---|
| `vocab_size` | 128 000 | Zeilen der Embedding-Matrix |
| `hidden_size` | 2048 | Breite des Residual-Streams |
| `intermediate_size` | 7680 | FFN-Breite |
| `num_hidden_layers` | 24 | Anzahl Blöcke |
| `num_attention_heads` / `num_key_value_heads` | 16 / 4 | GQA |
| `head_dim` | 128 | Rotations-/Norm-Dimension |
| `sliding_window` | 256 | Fenster der Sliding-Layer |
| `conv_kernel_size` | 7 | Breite der Short-Conv |
| `partial_rotary_factor` | 0.5 | rotierter Anteil der Kopfdimensionen |
| `layer_types` | `None` | `None` = kanonisches Muster; Liste = frei |
| `tie_word_embeddings` | `True` | Input = Output |

`to_hf_overrides()` übersetzt die Felder in die kwargs des Upstream-Configs
(z. B. `conv_L_cache`, `rope_parameters`). Wenn du eine eigene Config willst:

```python
cfg = LFM3_1_7B_BASE.with_size(hidden_size=1024, num_hidden_layers=12)
bundle = load_base(cfg)
```

`with_size()` rechnet `intermediate_size`, `num_attention_heads`, `num_key_value_heads`
und `head_dim` passend neu.

### 4.2 Runtime: `LoadOptions`

```python
from lfm3.config import LoadOptions

opts = LoadOptions(
    device="cuda",              # None -> cuda, sonst cpu
    dtype="bfloat16",           # bfloat16 / float16 / float32
    attn_implementation="sdpa", # sdpa / eager / flash_attention_2 / flex_attention
    load_tokenizer=True,
)
bundle = load_base(LFM3_1_7B_BASE, opts)
```

Presets:

| Name | Zweck |
|---|---|
| `DEFAULT_LOAD` | BF16, CUDA wenn verfügbar, SDPA |
| `CPU_LOAD` | FP32, CPU, eager |
| `TINY_LOAD` | Zufallsmodell, CPU, FP32, kein Tokenizer |
| `LFM3_1_7B_BASE` | Architektur 1.7B |
| `LFM3_OMNI_2B` | Architektur Omni |
| `TinyLFM3`, `MiniLFM3` | kleine Zufallsmodelle (Text) |
| `TinyOmni` | kleines Zufalls-Omni-Modell (Vision + Audio + Depth) |

### 4.3 Config-Overrides

`load_base(..., overrides={...})` überschreibt einzelne HF-Config-Felder *nach* dem
Preset. Praktisch, um das Layer-Muster zu ändern, ohne eine neue Config-Klasse zu
bauen:

```python
bundle = load_base(overrides={
    "layer_types": ["full_attention"] * 24,   # 24 volle Attention-Layer
})
```

`layer_types` und `num_hidden_layers` werden gemeinsam gesetzt, weil der Upstream-
Validator sonst meckert.

---

## 5. Laden (`lfm3/loader.py`)

### 5.1 `load_base`

```python
def load_base(cfg=LFM3_1_7B_BASE, opts=DEFAULT_LOAD, *, overrides=None,
              tokenizer_id=None, **from_pretrained_kwargs) -> ModelBundle
```

- lädt Config + Gewichte + Tokenizer,
- setzt Gerät und Datentyp,
- gibt ein `ModelBundle` zurück.

`from_pretrained_kwargs` geht direkt an `AutoModelForCausalLM.from_pretrained`
(`device_map`, `quantization_config`, `revision`, …).

### 5.2 `load_omni`

Analog, aber mit `AutoModelForImageTextToText` und `AutoProcessor`. Das Omni-Bundle
hat zusätzlich `processor` (Bild-/Audio-Vorverarbeitung).

### 5.3 `ModelBundle`

| Feld / Methode | Bedeutung |
|---|---|
| `model` | HF-Modell (`Lfm3ForCausalLM` bzw. `Lfm3OmniForConditionalGeneration`) |
| `tokenizer` | Tokenizer (bei Tiny-Modellen `None`) |
| `processor` | Omni-Processor, sonst `None` |
| `config` | instanziierte Upstream-Config |
| `lab_config` | Labor-Config, aus der das Bundle gebaut wurde |
| `text_model` | Text-Backbone (`model.model` bzw. `model.model.language_model`) |
| `access` | `ModelAccess`-View |
| `to(device, dtype)` | Modell verschieben/umcasten |
| `forward(...)` / `__call__` | direkt ans HF-Modell |
| `generate(...)` | HF-Generierung |
| `save(path)` / `load(path)` | vollständiges `state_dict` sichern/laden |
| `describe()` | lesbarer Report |

---

## 6. `ModelAccess` — der experimentelle Griff (`lfm3/access.py`)

`ModelAccess` ist eine **View**, kein Wrapper: jeder Aufruf liest/schreibt die Module
von `bundle.model`. Änderungen sind sofort im nächsten Forward aktiv.

### 6.1 Struktur abfragen

```python
acc = bundle.access

acc.embed_tokens          # nn.Embedding
acc.blocks                # nn.ModuleList mit 24 Blöcken
acc.final_norm            # RMSNorm (upstream: embedding_norm)
acc.lm_head               # nn.Linear(2048, 128000)
acc.rotary                # Lfm3RotaryEmbedding
acc.layer_types           # ['conv', 'conv', 'full_attention', ...]
acc.layer_type(4)         # 'sliding_attention'
acc.num_blocks            # 24

acc.parameter_groups()    # {'model.layers': {'total': ..., 'trainable': ...}, ...}
acc.trainable_summary()   # "trainable 1.17M / 1729.97M (0.07%)"
acc.embedding_report()    # Shape, dtype, Tying, Norm-Statistik, Pad-ID
acc.describe_layers()     # Tabelle Block -> Typ
acc.describe_tree()       # Modulbaum mit Parametern
```

### 6.2 Forward-Stufen

```python
acc.embed(ids)                                  # (B, T, 2048)
out = acc.forward_hidden(input_ids=ids)         # HF-Backbone-Output
out.last_hidden_state                           # (B, T, 2048) nach final norm
acc.logits_from_hidden(out.last_hidden_state)   # (B, T, 128000)
acc.forward_logits(ids)                         # komplettes Modell inkl. Kopf

stages = acc.forward_stages(ids)
# {'embeddings': ..., 'layers.0': ..., ..., 'layers.23': ..., 'final': ..., 'logits': ...}
```

`forward_stages` benutzt Forward-Hooks statt `output_hidden_states`, weil die
`hidden_states`-Liste in `transformers` ≥ 5.13 als letzten Eintrag den *nach*-Norm
Zustand führt und damit keine saubere Blockliste ist.

### 6.3 Aktivierungen mitschneiden

```python
with acc.capture() as acts:          # embed, alle Blöcke, final norm, lm_head
    bundle(ids)

acts.last("layers.4")                # letzter Forward
acts.stack("layers.4")               # (Aufrufe, B, T, 2048)
acts.names()
print(acts.summary())
```

Selektiv:

```python
with acc.capture(["layers.2.self_attn", "layers.2.feed_forward"]) as acts:
    bundle(ids)
```

### 6.4 Chirurgie

| Methode | Wirkung |
|---|---|
| `set_input_embeddings(module, tie_head=None)` | Embedding ersetzen (optional neu tie-en) |
| `set_embedding_matrix(tensor)` | Gewichtsmatrix in-place überschreiben |
| `resize_embeddings(n, keep_tied=None)` | Vokabular vergrößern/verkleinern (Input **und** Output), neue Zeilen = Mittelwert |
| `scale_embeddings(f)` | Matrix skalieren |
| `zero_row(token_id)` | eine Zeile nullen (z. B. Pad) |
| `set_output_head(module, tie=False)` | LM-Kopf ersetzen |
| `replace_output_head(...)` | Alias |
| `tie_head_to_embed()` / `untie_head()` | Tying setzen/lösen |
| `replace_block(i, module)` | Block ersetzen |
| `bypass_block(i, mode)` | Block ablaten (`identity` / `zero` / `scale`) |
| `attach_after_final_norm(module)` | `final_norm → module → lm_head` |
| `attach_module(name, module)` | beliebiges Untermodul anhängen |
| `freeze_all()` / `unfreeze_all()` | Gradienten |
| `freeze_before(i)` / `unfreeze_after(i)` | partielle Fine-Tunes |
| `save(path)` / `load(path)` | nur `state_dict` |

Beispiel: Embedding tauschen und Kopf sauber halten.

```python
import torch.nn as nn

new_embed = nn.Embedding(128000, 2048)
acc.set_input_embeddings(new_embed, tie_head=True)   # lm_head zeigt danach auf new_embed.weight
```

Beispiel: Output-Embedding ersetzen (Kopf wird vom Input-Embedding gelöst).

```python
acc.untie_head()                       # Kopie der Matrix, eigenes Gewicht
acc.set_output_head(nn.Linear(2048, 128000, bias=False))
```

### 6.5 Blöcke ersetzen

HF ruft jeden Block so auf:

```python
layer(hidden_states, position_embeddings=..., attention_mask=...,
      position_ids=..., past_key_values=..., seq_idx=..., **kwargs)
```

Ein Ersatzblock muss diese Signatur erfüllen. Für reine `hidden -> hidden`-Module gibt
es `BlockWrapper`:

```python
from lfm3.modules import BlockWrapper
acc.replace_block(7, BlockWrapper(my_module))
```

Oder direkt mit einem `BlockBypass`:

```python
acc.bypass_block(7, mode="zero")       # Block 7 löschen
acc.bypass_block(7, mode="scale", alpha=0.5)
```

Der Originalblock bleibt über den Rückgabewert erhalten:

```python
old = acc.replace_block(7, BlockWrapper(my_module))
...
acc.blocks[7] = old                     # zurücksetzen
```

---

## 7. Experimentelle Module (`lfm3/modules.py`)

### 7.1 N-Gramm-Schicht

```python
from lfm3.modules import NgramConfig, NgramLayer, build_n_gram

cfg = NgramConfig(
    n=3,               # Kontextlänge
    num_buckets=4096,  # Hash-Tabelle
    rank=32,           # niedrigrangiger Bypass
    mode="logit_bias", # oder "hidden_add"
)
ngram = build_n_gram(bundle, cfg)       # liegt danach als bundle.ngram bereit

logits = acc.forward_with_ngram(ids, ngram)
```

`mode="logit_bias"`: `logits = lm_head(h) + ngram(ids)` — Form `(B, T, V)`.
`mode="hidden_add"`: `logits = lm_head(h + ngram(ids))` — Form `(B, T, H)`.

Der Output-Projektor ist **null-initialisiert**, d. h. eine frische Schicht ist ein
No-op und ändert das Modell nicht. Das ist gewollt: Du kannst sie gefahrlos
hinzuschalten und dann trainieren.

**Wichtig**: Die Schicht braucht die Token-IDs, die der LM-Kopf nicht bekommt. Nutze
`forward_with_ngram` für Training/Teacher-Forcing. Bei gecachter Generierung bekommt
das Modell nur das letzte Token; dort `hidden_add` verwenden oder ohne Cache dekodieren.

### 7.2 Kleines Modell vor dem Output-Embedding

```python
from lfm3.modules import BottleneckConfig, BottleneckAdapter, TinyTransformerAdapter

adapter = BottleneckAdapter(
    hidden_size=2048,
    BottleneckConfig(bottleneck=512, layers=2, residual=True),
)
acc.attach_after_final_norm(adapter)     # final_norm -> adapter -> lm_head
```

`residual=True` + Zero-Init bedeutet: direkt nach dem Anhängen ist das Modell
**verändert um nichts** (identisch). Danach kannst du es per Backprop trainieren.

`TinyTransformerAdapter` ist dasselbe mit 1–2 `nn.TransformerEncoderLayer`-Layern
(sequenzweit, nicht nur positionsweise).

### 7.3 LoRA / Fine-Tuning

```python
from lfm3.modules import LoRAConfig, inject_lora, mark_only_lora_trainable

report = inject_lora(bundle.model, LoRAConfig(rank=8, alpha=16.0))
mark_only_lora_trainable(bundle.model)

# report = {'wrapped': 44, 'trainable': 1_170_000, 'total': 1_729_970_000}
```

`inject_lora` ersetzt `nn.Linear`-Layer (Default: `q_proj`, `k_proj`, `v_proj`,
`o_proj`) durch `LoRALinear`. Die Basisgewichte werden eingefroren, nur `lora_A` und
`lora_B` trainieren.

```python
from lfm3.modules import lora_state_dict, load_lora_state_dict, LoRALinear

torch.save(lora_state_dict(bundle.model), "lora.pth")
load_lora_state_dict(bundle.model, torch.load("lora.pth"))
```

Die State-Dict-Keys ändern sich durch LoRA (`...base.weight` statt `...weight`), deshalb
gibt es `lora_state_dict`. Willst du die Update-Matrix in die Basis falten:

```python
merged = lora_linear.merged()    # nn.Linear mit eingebettetem Update
```

### 7.4 Probe / Klassifikator

```python
from lfm3.modules import ProbeConfig, ProbeHead

probe = ProbeHead(2048, 128000, ProbeConfig(hidden=512))
acc.set_output_head(probe)
```

### 7.5 Embedding-Adapter

```python
from lfm3.modules import EmbeddingConfig, EmbeddingAdapter

adapter = EmbeddingAdapter(bundle.access.embed_tokens,
                           EmbeddingConfig(rank=64))
acc.set_input_embeddings(adapter, tie_head=False)
```

`scale * E[x] + bias + B[x] @ L` — ein trainierbarer Bypass um die Embedding-Matrix.

---

## 8. Reine PyTorch-Blöcke (`lfm3/pytorch_blocks.py`)

Wenn du die `transformers`-Masken-/Cache-Maschinerie nicht willst:

```python
from lfm3.pytorch_blocks import ReferenceBackbone

ref = ReferenceBackbone.from_hf(bundle.access.backbone)   # Gewichte direkt übernehmen
h = ref.forward_hidden(input_ids=ids)                     # (B, T, 2048)
```

Module:

| Klasse | entspricht |
|---|---|
| `ReferenceRMSNorm` | `Lfm3RMSNorm` |
| `ReferenceMLP` | `Lfm3MLP` (SwiGLU) |
| `ReferenceShortConv` | `Lfm3ShortConv` (gated depthwise conv) |
| `ReferenceAttention` | `Lfm3Attention` (GQA, QK-Norm, partielle RoPE, Gate) |
| `ReferenceDecoderLayer` | `Lfm3DecoderLayer` (Sandwich-Norm) |
| `ReferenceBackbone` | `Lfm3Model` (Embeddings + Blöcke + final norm) |

Die Modul- und Parameternamen sind identisch zu HF, deshalb ist `from_hf()` ein
`load_state_dict`. Parity-Test:

```python
python lfm3_lab.py demo parity
```

Auf dem echten Modell (BF16) sind die Conv-Layer bitgenau, die Attention-Layer weichen
um ~1e-2 ab (BF16-Rundung über 24 Blöcke); der Cosine-Abstand liegt bei 0.99999.

---

## 9. CLI (`lfm3_lab.py`)

```
python lfm3_lab.py [--model base|omni] [--tiny] [--device ...] [--dtype ...]
                   [--attn ...] [--budget]
                   {arch,info,demo,generate} ...
```

| Befehl | Zweck |
|---|---|
| `arch` | Architekturbericht aus der Config, **ohne** Gewichte |
| `info` | vollständiger Report des geladenen Checkpoints |
| `demo` | Experimente ausführen (`--list`, `--all`, oder Namen) |
| `generate` | Fortsetzung samplen |

Gemeinsame Optionen dürfen vor oder nach dem Subcommand stehen.

`--tiny` lädt ein Zufallsmodell (gleiche Architektur, 6 Layer, 64 breit) — ideal, um
Code-Pfade in Sekunden zu testen, ohne 3.46 GB zu laden.

---

## 10. Rezepte

### 10.1 „Ich will wissen, was ein Block beiträgt"

```python
acc = bundle.access
ids = bundle.tokenizer("The capital of France is", return_tensors="pt").input_ids.to(bundle.device)

with torch.no_grad():
    base = acc.forward_logits(ids)

for i in range(acc.num_blocks):
    old = acc.blocks[i]
    acc.bypass_block(i, "identity")
    with torch.no_grad():
        out = acc.forward_logits(ids)
    print(i, acc.layer_type(i), float((out - base).abs().mean()))
    acc.blocks[i] = old
```

### 10.2 „Ich will ein kleines Modell vor den LM-Kopf hängen und trainieren"

```python
import torch
import torch.nn.functional as F
from lfm3.modules import BottleneckAdapter, BottleneckConfig

acc = bundle.access
ids = ...                                   # (B, T)
with torch.no_grad():
    reference = acc.forward_logits(ids)

adapter = BottleneckAdapter(2048, BottleneckConfig(bottleneck=256))
acc.attach_after_final_norm(adapter)

opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
for step in range(100):
    logits = acc.forward_logits(ids)
    loss = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(reference, -1),
                    log_target=True, reduction="batchmean")
    opt.zero_grad(); loss.backward(); opt.step()
```

Das Demo `lfm3_lab.py demo adapter` zeigt genau das mit Ausgabe.

### 10.3 „Ich will die Input-Embeddings ändern"

```python
acc = bundle.access
acc.scale_embeddings(1.1)                   # alle Zeilen skalieren
acc.zero_row(124_893)                       # Pad-Zeile nullen
acc.set_embedding_matrix(my_matrix)         # ganze Matrix ersetzen
acc.resize_embeddings(130_000)              # Vokabular erweitern (Input + Output)

# oder: trainierbarer Bypass
from lfm3.modules import EmbeddingAdapter, EmbeddingConfig
acc.set_input_embeddings(
    EmbeddingAdapter(acc.embed_tokens, EmbeddingConfig(rank=64)),
    tie_head=False,
)
```

### 10.4 „Ich will eine N-Gramm-Schicht"

```python
from lfm3.modules import NgramConfig, build_n_gram
import torch.nn.functional as F

ngram = build_n_gram(bundle, NgramConfig(n=3, num_buckets=8192, rank=64))
opt = torch.optim.Adam(ngram.parameters(), lr=1e-3)

for step in range(200):
    logits = bundle.access.forward_with_ngram(ids, ngram)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 128000), ids[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
```

### 10.5 „Ich will einen Block durch eigene Aufmerksamkeit ersetzen"

```python
import torch.nn as nn
from lfm3.modules import BlockWrapper

class MyAttentionBlock(nn.Module):
    def __init__(self, hidden=2048):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
    def forward(self, x):
        # hier ist x (B, T, H); die Block-Signatur kapselt BlockWrapper
        return self.o(self.q(x))

acc.replace_block(2, BlockWrapper(MyAttentionBlock()))
```

Wenn du RoPE/Masken/KV-Cache korrekt behandeln willst, nimm
`lfm3.pytorch_blocks.ReferenceAttention` als Vorlage.

### 10.6 „Ich will einen editierten Block gegen das Original prüfen"

```python
from lfm3.pytorch_blocks import ReferenceDecoderLayer

hf_layer = acc.blocks[2]
ref = ReferenceDecoderLayer(bundle.config, 2).to(bundle.device, bundle.dtype)
ref.load_state_dict(hf_layer.state_dict())

with torch.no_grad():
    a = hf_layer(hidden)      # ggf. position_embeddings / attention_mask
    b = ref(hidden)
print((a.float() - b.float()).abs().max())
```

### 10.7 „Ich will Omni-Code testen, ohne 4.3 GB zu laden"

```python
from lfm3.loader import load_tiny_omni

bundle = load_tiny_omni()          # Zufallsmodell mit echter Omni-Verdrahtung
bundle.processor                   # None (nur echten Checkpoint gibt es einen)
bundle.text_model                  # LFM3-1.7B Text-Tower
bundle.model.model.vision_tower    # Vision
bundle.model.model.audio_encoder   # Audio
bundle.model.model.depth_decoder   # Sprach-Codebooks
```

Für echte multimodale Experimente:

```python
bundle = load_omni()
bundle.processor                     # Bild-/Audio-Preprocessing
bundle.text_model                    # LFM3-1.7B Text-Tower
bundle.model.model.vision_tower      # Vision
bundle.model.model.audio_encoder     # Audio
bundle.model.model.depth_decoder     # Sprach-Codebooks
```

---

## 11. Troubleshooting

| Symptom | Ursache | Lösung |
|---|---|---|
| `KeyError: 'conv'` beim Config-Laden | transformers ≥ 5.13 + LFM3-Config | `apply_compat_shims()` läuft automatisch; du siehst den Fehler nur, wenn du `AutoConfig` direkt ohne Labor benutzt |
| `AssertionError: Padding_idx must be within num_embeddings` | verkleinertes Vokabular, aber Pad-ID aus dem Original | `to_hf_overrides()` setzt ungültige Special-IDs auf `None`; bei eigenen Configs prüfen |
| Modell auf CPU, LoRA auf GPU | Surgery nach `.to(device)` | `inject_lora` erbt device/dtype der Basis; bei eigenen Modulen `.to(bundle.device)` |
| `RuntimeError: Expected all tensors on the same device` | Embedding/Kopf auf anderem Gerät | `bundle.to(device)`, danach Surgery |
| Tokenizer passt nicht zum Zufallsmodell | Tiny-Modell hat 512 Vokabeln | `TinyLFM3` lädt keinen Tokenizer (`load_tokenizer=False`); `_is_tiny()` erkennt das |
| 3.46 GB Download | erster Lauf | Cache unter `~/.cache/huggingface`; `--tiny` für Tests |
| Zu wenig GPU-RAM | BF16 1.7B braucht ~3.5 GB + KV-Cache | `LoadOptions(device="cpu")` oder kleinere Config |
| `generate()` wird degeneriert | Base-Modell + Greedy | Sampling benutzen (`do_sample=True, temperature=0.8, top_p=0.95`) |
| gepackte Prompts bei `generate()` abgelehnt | LFM3 kann Dokumentgrenzen im Cache nicht isolieren | pro Dokument eine Zeile, oder ohne Cache |
| Omni-Prozessor fehlt | `random_init=True` | Prozessor nur bei echten Checkpoints laden |

---

## 12. Design-Entscheidungen (und warum)

- **View statt Wrapper**: `ModelAccess` mutiert das echte Modell. Kein „Copy-modify-
  forward"-Zustand, keine Drift zwischen View und Modell.
- **Configs als Dataclasses**: konsistent mit dem Rest des Projekts, typisiert, als
  Preset wiederverwendbar.
- **HF-Modellcode statt Reimplementierung als Hauptweg**: der Checkpoint-Code ist die
  Referenz; eine Reimplementierung wäre eine Fehlerquelle. Die `pytorch_blocks` sind
  bewusst *additiv* und per Parity-Test abgesichert.
- **Zero-Init bei neuen Köpfen**: neue Module starten als No-op. So ist ein
  Experiment reproduzierbar vom unveränderten Modell aus.
- **Lokale Module**: N-Gramm, Adapter und LoRA sind eigenständig testbar und nicht an
  `transformers`-Interna gekoppelt.
- **Ein CLI**: `lfm3_lab.py` deckt die häufigsten Wege ab, ohne dass du ein Notebook
  brauchst.

---

## 13. Weiterführend

- Architektur 1.7B: [`LFM3_1.7B_BASE.md`](LFM3_1.7B_BASE.md)
- Architektur Omni: [`LFM3_OMNI_2B.md`](LFM3_OMNI_2B.md)
- Modellkarten: `LiquidAI/LFM3-1.7B-Base`, `LiquidAI/LFM3-Omni-2B`
- HF-Dokumentation: <https://huggingface.co/docs/transformers>