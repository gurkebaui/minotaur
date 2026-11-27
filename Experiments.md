# Mini-Experimente & Evaluationsplan (RTX 4060 Ti 16 GB)

## 1. Grundprinzipien & Zeitbudget
* **Hardware-Target:** 1× **RTX 4060 Ti 16 GB (`sm_89`)**, Code-Struktur über Feature-Flags 1:1 skalierbar auf **H100 80 GB (`sm_90`)**.
* **Zeitlimit:** Jedes der 7 Experiment-Pakete dauert **maximal 3 Stunden** (Gesamtlaufzeit ~18,5 Stunden — bequem über 1–2 Nächte ausführbar).
* **Smart Bundling mit Fehlertoleranz:** Wir bündeln Module, die **getrennte, eindeutig messbare Signale** hinterlassen (z. B. VRAM-Aufschlüsselung, Profiler-Kernel-Zeit, Bits-per-Byte, Router-Entropie, Gradienten-Spikes). Fällt ein Paket durch, wissen wir anhand der Diagnose-Metrik sofort, *welches* Modul schuld war — ohne jedes Einzelteil separat testen zu müssen.

---

## 2. Einheitliches Proxy-Setup & Pflicht-Metriken

### Proxy-Modell & Daten
* **Proxy-Größe:** **~120M–150M Backbone-Parameter** (= Stage-1-Backbone aus [ModelB.md](file:///home/henry/Documents/Henry/minotaur/ModelB.md)), parametrisiert mit **muP** für 1:1-Hyperparameter-Übertragbarkeit auf 3B.
* **Trainings-Slice:** Fester, reproduzierbarer Seed/Slice (**300M–500M Tokens** FineWeb-Edu + Code-Mix).
* **Evaluations-Set:**
  1. Festes **10M-Token Validation-Set** (Text + Code).
  2. **Synthetische Reasoning-Probes** (für Exp 5 & 6): Multi-Step Arithmetic, Parity / State-Tracking, verschachtelte Klammern / Code-Indent.

### Pflicht-Metriken (automatisch in `metrics.jsonl` / `summary.csv` geloggt)
1. **Qualität:** `Val-Loss` und **`Bits-per-Byte (BPB)`** *(BPB ist Pflicht, weil SuperBPE eine andere Token-Anzahl pro Text hat!)*.
2. **Speed:** `Tokens/sec`, `Bytes/sec` (entscheidend bei SuperBPE) und `ms/Step` (aufgeteilt in Forward / Backward / Optimizer).
3. **VRAM-Breakdown (in MB):** Aufgeschlüsselt nach `Weights`, `Gradients`, `Optimizer States` und `Peak Activations`.
4. **Stabilität:** Anzahl `NaN/Inf`-Steps, `max(grad_norm)`, Loss-Spikes.
5. **Reasoning-Accuracy (ab Exp 5):** Accuracy auf den Reasoning-Probes bei Loop-Tiefe $R \in \{1, 2, 4, 8\}$.
6. **Code- & H100-Tauglichkeit:** Subjektive Komplexitätsbewertung + kurzer Check, ob ein Custom-Kernel auf `sm_90` schneller skaliert.

---

## 3. Die 7 gebündelten Experiment-Pakete (je ≤ 3 Stunden)

### Exp 0 (~1.5 h): Baseline & Systems-Bundle („Free Lunch“-Check)
* **Ziel:** Saubere Referenz-Baseline `B0` etablieren und prüfen, ob der gesamte Systems-Unterbau auf `sm_89` ohne Fehler zusammenspielt.
* **Gebündelte Module:**
  * `Run 0a (25 min)`: Standard 150M Dense (PyTorch SDPA, Standard Cross-Entropy, BF16+FP32 Master, AdamW).
  * `Run 0b (45 min)`: `FlashAttention-2 + Liger-Kernel + torch.compile + Cut Cross-Entropy (CCE) + Sequence Packing (EoS-Aligned) + Selective Checkpointing + QK-Norm + ReLU²`.
  * `Mini-Sweep (20 min)`: Jeweils 2-Minuten-Läufe mit einzeln deaktiviertem `Liger`, `CCE` und `torch.compile`, um exakte Speed-/VRAM-Deltas zu isolieren.
* **Fehler-Isolierung:**
  * Triton-/Compile-Crash $\to$ `torch.compile` oder `Liger-Kernel` Flag.
  * VRAM-Peak bei großem Vocab $\to$ `CCE` Flag.
* **Go / No-Go:** `0b` muss bei identischem oder besserem BPB mindestens **1.3× schneller** sein und **>40 % Aktivierungs-VRAM** sparen $\implies$ wird neue Baseline **`B0`**.

---

### Exp 1 (~2.5 h): Ultra-Low-VRAM & Präzisions-Bundle
* **Ziel:** Den maximalen VRAM- & Speed-Gewinn bei Gewichten, Gradienten und Optimizer-States auf `sm_89` verifizieren.
* **Gebündelte Module:**
  * `Run 1A (1.2 h — „Clean & Fast“)`: `B0` + **Pure BF16 + Stochastic Rounding** (keine FP32 Master Weights) + **8-bit Muon** (2D-Weights) / 8-bit AdamW (1D/Embed) + **LOMO Fused Backward-Optimizer Hook** + **WSD-Schedule + EMA**.
  * `Run 1B (1.3 h — „Max Compression / 7B-Ready“)`: `B0` + **Ada FP8/INT8 (`torchao`)** + **COAT** (Optimizer/Activation Compression) + **APOLLO-Mini** + **LOMO Fused Hook** + **LFM3 Head-Wise Gating**.
* **Fehler-Isolierung:**
  * `Optimizer-VRAM` > 10 % der Weights $\to$ `Fused Hook` oder `APOLLO-Mini` greift nicht.
  * Loss divergiert / NaNs in `1B`, aber nicht in `1A` $\to$ `Ada FP8` Scaling-Problem (prüfen, ob `Head-Wise Gating` oder `Stochastic Rounding` fehlt).
* **Go / No-Go:**
  * `1A` muss `B0` im **BPB pro Stunde** klar schlagen (durch Muon) bei halbiertem Weight/Optimizer-VRAM.
  * `1B` darf max. **+0.02 BPB** gegenüber `1A` verlieren, muss aber deutlich weniger VRAM brauchen und schneller laufen (falls `sm_89` FP8-Path greift).

---

### Exp 2 (~3.0 h): Architektur- & FLOP-Killer-Bundle
* **Ziel:** Testen, wie stark **CoLA + LFM3 Tri-Hybrid + SuperBPE + MTP** die echten FLOPs pro Byte Text drücken.
* **Gebündelte Module:**
  * `Run 2A (1.5 h — Backbone-Mix)`: `1A` + **CoLA** (Low-Rank Activation Autoencoder in MLPs/Proj) + **LFM3 Tri-Hybrid** (~50 % Double-Gated Short Conv $k=7$, ~25 % SWA $W=256$, ~25 % Global GQA/MLA + **Head-Wise Gating** + Partial RoPE).
  * `Run 2B (1.5 h — Token-Effizienz)`: `2A` + **SuperBPE-Tokenizer** + **Multi-Token Prediction (MTP, $k=2$)**.
* **Fehler-Isolierung:**
  * `2A` langsamer als `1A` $\to$ `conv1d(k=7)`-Kernel oder `CoLA`-Rank zu unoptimal für `torch.compile`.
  * `2A` hat guten Val-Loss, aber schlechtes Long-Range-Verhalten $\to$ zu wenig Global-GQA-Layer.
  * `2B` gewinnt bei `Bytes/sec`, verliert aber stark bei `BPB` $\to$ `SuperBPE`-Vokabular zu groß für 150M-Proxy oder `MTP`-Gewicht zu hoch.
* **Go / No-Go:** `2A`/`2B` müssen bei gleichem Wanduhr-Budget (1.5 h) einen **niedrigeren BPB** erreichen als `1A` (durch höheren Textdurchsatz pro Sekunde).

---

### Exp 3 (~3.0 h): Parameter-Entkopplung & Self-Growth-Bundle
* **Ziel:** Verifizieren von **UltraMemV2** (Fakten-Speicher) und **HyperCloning + GStack** (From-Scratch-Wachstum für [ModelB.md](file:///home/henry/Documents/Henry/minotaur/ModelB.md)).
* **Gebündelte Module:**
  * `Run 3A (1.5 h — UltraMemV2)`: `2A` (150M CoLA-Backbone) + **600M UltraMemV2 / PKM-Tabellen** im 1:4-Verhältnis (exakt wie 600M + 2.4B im 3B-Zielmodell).
  * `Run 3B (1.5 h — Self-Growth)`: Starte mit **50M CoLA-Backbone** (45 min) $\to$ **HyperCloning + GStack** auf 150M CoLA + **Zero-Init Sparse Upcycling** der 600M UltraMemV2-Tabellen (45 min).
* **Fehler-Isolierung:**
  * Hoher `ms/Step`-Sprung in `3A` $\to$ PKM-Lookup-Kernel ist Memory-Bandwidth-bound (Top-$k$ oder Tabellen-Layout anpassen).
  * Permanenter Loss-Spike nach Growth in `3B` $\to$ `Zero-Init` oder LR-Reset nach HyperCloning/GStack fehlerhaft.
* **Go / No-Go:**
  * `3A` muss den `BPB` von `2A` bei <20 % Geschwindigkeitsverlust deutlich schlagen.
  * `3B` muss am Ende der 1.5 h auf dem Niveau von `3A` landen (oder besser).

---

### Exp 4 (~3.0 h): Daten- & Lossy-Backward-Bundle
* **Ziel:** Validierung der großen Daten-Multiplikatoren für **Modell B** (`4A`) und **Modell A** (`4B`) inkl. unserer **Rho-1++ Zeitplan-Regel**.
* **Gebündelte Module:**
  * `Run 4A (1.5 h — Model B / From Scratch)`: Beste Architektur aus Exp 2/3 + **Self-Bootstrapped JEST** (Proxy = Stage-1-Checkpoint) + **Self-Scored Token-Selective Sparse Backward**:
    * `0–40 %`: 100 % Tokens
    * `40–90 %`: **55 % Cutoff** (nur Top-45 % Tokens im Backward)
    * `90–100 %`: wieder 100 % Tokens (WSD-Cooldown)
  * `Run 4B (1.5 h — Model A / External Allowed)`: **Offline Top-16 Logit-Distillation** (von einem stärkeren Teacher vorab auf SSD gecacht) + extern gescortes **Rho-1++** (selber 40–90 % Zeitplan).
* **Fehler-Isolierung:**
  * Kein Speedup in Phase `40–90 %` $\to$ Gather/Scatter-Overhead beim Token-Cutting frisst Backward-Ersparnis auf (dann lieber Block-Mask/Batch-Packing statt Token-Slicing testen).
  * Loss bricht ab 40 % ein und erholt sich bei 90–100 % nicht $\to$ 55 % Cutoff zu aggressiv (auf 35–40 % reduzieren).
* **Go / No-Go:** Phase `40–90 %` muss spürbar schneller laufen (≥ 1.25× Step-Speedup) und am Ende von `90–100 %` mindestens denselben `BPB` wie ein 100 %-Run erreichen (bei `4B` deutlich besser).

---

### Exp 5 (~3.0 h): Run 2 Core — Looped Transformer & MoR-Bundle
* **Ziel:** Testen der sparsamen Loop-Architektur aus [Run2.md](file:///home/henry/Documents/Henry/minotaur/Run2.md) per **Loop-Upcycling** auf dem besten Checkpoint aus Exp 2/3.
* **Gebündelte Module:**
  * `Run 5A (1.5 h — Stabiler Loop-Kern)`: **Huginn Sandwich** (Prelude $\to$ Core $\to$ Coda) + **Input-Injection** + **Zero-Init Per-Token Loop-Gate** + **Step-Embeddings & Per-Loop RMSNorms** + **Randomized $R \in [1, 4]$** + **TBPTT $k=2$ (`no_grad` für frühere Loops)** + **UltraMemV2 1× außerhalb des Loops**.
  * `Run 5B (1.5 h — Max Sparsity)`: `5A` mit **TBPTT $k=1$** + **Mixture-of-Recursions (MoR)** (~35 % aktive Tokens in Loop 2–4, selektives KV-Caching) + **loop-kompatibler Fused Grad-Hook**.
* **Fehler-Isolierung:**
  * `5A` divergiert bei $R=4$ $\to$ Loop-Gate oder Input-Injection fehlt/zu stark.
  * `5B` verliert stark gegenüber `5A` $\to$ entweder $k=1$ TBPTT zu kurz (zurück auf $k=2$) oder MoR-Router kollabiert (Router-Balance-Loss prüfen).
  * `5B` langsamer als `5A` $\to$ MoR Gather/Scatter lohnt sich bei 150M noch nicht (dann für 3B testen oder starres $R=2$ nehmen).
* **Go / No-Go:** Test-Time-Scaling-Check! Bei der Evaluation müssen $R=4$ und $R=8$ auf den **Reasoning-Probes** klar besser abschneiden als $R=1$.

---

### Exp 6 (~2.5 h): Run 2 Exp — LFM3-Conv-YOCO-Loop & CTM-Light
* **Ziel:** Isolierter Test der experimentellen Module aus [Run2_Exp.md](file:///home/henry/Documents/Henry/minotaur/Run2_Exp.md) direkt gegen den Gewinner aus Exp 5 (`5A` oder `5B`).
* **Gebündelte Module:**
  * `Run 6A (1.25 h — Modul E4: LFM3-Conv-YOCO-Loop)`: Prelude erzeugt 1× globalen **YOCO-KV-Cache**; Looped Core nutzt **Double-Gated Short Conv ($k=7$)** (auf voller Sequenz) + **YOCO Cross-Attention** (gesliced auf aktive MoR-Tokens) + **Head-Wise Gating**.
  * `Run 6B (1.25 h — CTM-Light E1 + E2 + optional E3)`: Subsampled Neural Synchronization ($P=256$) + **Certainty-Based Halting Loss** + optional Tick-History-EMA.
* **Fehler-Isolierung:**
  * `6A` verliert bei Global-Recall $\to$ statischer YOCO-Cache aus Prelude reicht nicht; mindestens 1 frischer Attention-Layer nötig.
  * `6B` verliert `Tokens/sec` > 10 % $\to$ Tick-History (`E3`) oder Sync-Gather (`E1`) zu teuer $\to$ nur `E2` (Certainty-Loss) behalten oder alles streichen.
* **Go / No-Go:** Strikte Hürden wie in [Run2_Exp.md](file:///home/henry/Documents/Henry/minotaur/Run2_Exp.md): **Max. 10 % Speed-Verlust**, **< 500 MB VRAM-Delta** (hochgerechnet auf 3B) und **messbarer Gewinn** auf Reasoning-Probes oder Early-Exit-Rate.

---

## 4. Code-Architektur-Blueprint (Modulares Experiment-Harness)

Damit wir nicht 7 verschiedene Codebasen pflegen müssen, sondern alle Experimente aus **einem sauberen Repo** fahren (und später 1:1 auf 3B/H100 hochdrehen):

```text
minotaur_exp/
├── config.py          # Zentrale Dataclass mit Feature-Flags (use_cola, use_ultramem, use_lfm3, loop_cfg, ...)
├── model/
│   ├── attention.py   # FA-2/FA-3 Auto-Switch, SWA, CLA/YOCO, Head-Wise Gating, Partial RoPE
│   ├── conv_mixer.py  # LFM3 Double-Gated Short Conv (k=7) + Ringbuffer-Decode
│   ├── blocks.py      # CoLA Low-Rank Linear/MLP, ReLU², QK-Norm, Sandwich Prelude/Core/Coda
│   ├── memory.py      # UltraMemV2 / PKM-Lookup-Layer (1x im Prelude/Coda)
│   ├── router.py      # MoR-Token-Router + Optional CTM-Light (E1 Sync, E2 Certainty)
│   └── growth.py      # HyperCloning (Width) + GStack (Depth) + Zero-Init Sparse Upcycling
├── optim/
│   ├── muon_8bit.py   # 8-bit Muon + APOLLO-Mini + LOMO Fused Grad-Hook (loop-aware counter!)
│   ├── precision.py   # Pure-BF16 + Stochastic Rounding / TorchAO Ada-FP8 + COAT
│   └── schedule.py    # WSD-Schedule, EMA/LAWA, Rho-1++ Sparse-Backward-Scheduler (0-40-90-100%)
├── data/
│   ├── pipeline.py    # Sequence Packing, EoS-Masking, SuperBPE-Support, JEST-Batch-Sampler
│   └── distill.py     # Offline Top-K Logit Loader (für Exp 4B / Model A)
└── train_eval.py      # Einheitlicher Runner: VRAM-Profiler, BPB-Logger, Reasoning-Probes
```

---

## 5. Entscheidungsmatrix nach dem Review (Assembly)

Nach Auswertung der 7 Experiment-Reports bauen wir die finalen 3B-Configs zusammen:
* **Model A (`ModelA.md`):** Gewinner aus `Exp 0 + 1 + 2 + 3A + 4B` (mit Teacher-Distillation & Warm-Start).
* **Model B (`ModelB.md`):** Gewinner aus `Exp 0 + 1 + 2 + 3B + 4A` (100 % From Scratch mit Self-Growth).
* **Run 2 (`Run2.md`):** Gewinner-Loop-Setup aus `Exp 5` (+ `6A`, falls `E4` LFM3-Conv-YOCO überzeugt!).
* **Run 2 Exp (`Run2_Exp.md`):** Wird nur aktiviert, falls `Exp 6B` (CTM-Light) seine Go/No-Go-Kriterien klar erfüllt.
