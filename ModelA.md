# Modell A: 3B Transformer in 4–7 Tagen auf RTX 4060 Ti 16 GB (Externe Modelle erlaubt)

## Rahmenbedingungen & Zielsetzung
* **Ziel:** Training eines leistungsstarken **3B-Parameter-Transformers** in **4 Tagen (max. 7 Tagen)** auf einer einzelnen **RTX 4060 Ti 16 GB (`sm_89`, Ada Lovelace, 288 GB/s)**.
* **Skalierbarkeit:** Der gesamte Code-Stack skaliert 1:1 per Flag auf eine **NVIDIA H100 80 GB (`sm_90`, Hopper, 3.35 TB/s)** (z. B. Wechsel von FlashAttention-2 auf FlashAttention-3, nativem Hopper-FP8 und deaktiviertem/selektivem Activation Checkpointing).
* **Regel für Modell A:** **Externe Modelle sind explizit erlaubt** — d. h. Warm-Start über Weight-/HyperCloning bestehender Open-Weights-Modelle, Pruning + Healing (Minitron), Offline-Logit-Distillation von großen Teacher-Modellen sowie externe Referenzmodelle für Batch-/Token-Scoring (JEST, Rho-1++).

---

## 4-Tage-Phasenplan (inkl. Sparse-Backward-Schedule)

| Trainingsphase | Fortschritt | Token-Selective Sparse Backward | Hauptfokus |
| :--- | :--- | :--- | :--- |
| **Phase 1: Warm-Start & Alignment** | **0 % – 30/50 %** (Tag 1) | **Aus (100 % der Tokens)** | Initialisierung per **HyperCloning / Minitron-Pruning** aus starkem Basismodell (z. B. 0.5B/1.5B $\to$ 3B oder 7B $\to$ 3B) + optionales **Sparse Upcycling** (UltraMemV2). Volle Gradienten stabilisieren die geklonten/geprunten Gewichte. |
| **Phase 2: High-Speed Healing & Distillation** | **30/50 % – 90 %** (Tag 2 – Tag 3.5) | **Aktiv (50–60 % Cutoff)** | **Token-Selective Sparse Backward** schneidet die einfachsten 50–60 % der Tokens im Backward-Pass ab (bewertet durch externes Referenzmodell). Parallel laufen **Offline Top-$K$ Logit-Distillation**, **JEST** und **FP8/INT8-Compute**. |
| **Phase 3: WSD-Cooldown & Polishing** | **90 % – 100 %** (Tag 3.5 – Tag 4) | **Aus (100 % der Tokens)** | Steiler LR-Decay (**WSD-Schedule**) auf höchstkuratierten Daten mit wieder **100 % aktiven Gradienten** für maximale Endqualität, abgeschlossen durch **Checkpoint Averaging / EMA / LAWA**. |

---

## 1. S-Rank Methoden (Pflicht-Stack)

### 1.1 Warm-Start via HyperCloning & GStack / Minitron Pruning + Healing *(Spezifisch für Modell A)*
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Anstatt 3B Parameter bei Random Init zu starten, wird entweder ein kleines, starkes Open-Weights-Modell (z. B. 0.5B–1.5B) per **HyperCloning** (Breiten-Expansion) und **GStack** (Tiefen-Stacking) funktions-erhaltend auf 3B hochskaliert, oder ein größeres Modell (z. B. 7B) per **Minitron** (Width/Depth-Pruning) auf 3B geschrumpft und anschließend geheilt.
* **Spart:** **Compute / Zeit (20–100× vs. Scratch)** – das frühe, extrem teure Erlernen von Grundsyntax und Weltwissen entfällt komplett.
* **Zahlt:** Abhängigkeit von einem externen Basismodell; kurzer initialer Loss-Spike bei Architektur-Anpassungen (falls kombiniert).
* **Qualitätsverlust:** **Stark Positiv** (schlägt jedes Scratch-Training im 4-Tage-Budget um Längen).
* **Quellen:**
  * HyperCloning: <https://arxiv.org/abs/2409.12903>
  * GStack (*Stacking Your Transformers*): <https://arxiv.org/abs/2405.15319>
  * Minitron (*Compact Language Models via Pruning and Knowledge Distillation*): <https://arxiv.org/abs/2407.14679>

### 1.2 Offline Top-$K$ Logit-Distillation *(Spezifisch für Modell A)*
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Ein starkes Teacher-Modell (z. B. 32B oder 70B) berechnet einmalig vorab die **Top-8 bis Top-16 Logits** (Wahrscheinlichkeitsverteilung) pro Trainingstoken und speichert sie komprimiert auf SSD. Während des 4060-Ti-Trainings lernt Modell A über eine Kombination aus Cross-Entropy und KL-Divergenz auf diesen gecachten Soft-Targets, ohne den Teacher im VRAM halten zu müssen.
* **Spart:** **Tokens / Compute (3–6× weniger Tokens für gleiche Qualität)**, **VRAM** (kein Teacher im GPU-Speicher nötig).
* **Zahlt:** **SSD-Speicherplatz** (mehrere 100 GB für gecachte Top-$K$-Logits) + einmaliger Offline-Vorlauf.
* **Qualitätsverlust:** **Stark Positiv** (viel reichhaltigeres Trainingssignal als reines One-Hot-Next-Token-Prediction).
* **Quellen:**
  * Minitron (Distillation-Pipeline): <https://arxiv.org/abs/2407.14679>

### 1.3 JEST (Joint Example Selection) mit externem Referenzmodell
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Statt Datenpunkte isoliert zu filtern, bewertet ein kleines vortrainiertes Referenzmodell ganze Super-Batches auf ihre **gemeinsame Lernbarkeit** (*Joint Learnability*) und stellt daraus optimale Sub-Batches für Modell A zusammen.
* **Spart:** **Iterationen / Compute (bis zu 5–13×)**, **Trainingszeit**.
* **Zahlt:** **Offline-/Proxy-Scoring-Compute** (vorab auf GPU/CPU ausführbar); Risiko von Distribution-Bias bei zu engem Referenz-Set.
* **Qualitätsverlust:** **Positiv** (deutlich schnellerer Benchmark-Fortschritt pro Step).
* **Quellen:**
  * JEST: <https://arxiv.org/abs/2406.17711>

### 1.4 Hyper-Synthetic & Kuratiertes Daten-Curriculum + Optimal Mixing
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Training ausschließlich auf hochdichten, semantisch deduplizierten und kuratierten Daten (z. B. synthetische Lehrbuch-Daten, FineWeb-Edu, DCLM) statt rohem Web-Crawl, kombiniert mit optimierten Domain-Mischungen.
* **Spart:** **Tokens / Compute (2–8× weniger Tokens)**.
* **Zahlt:** **Offline-CPU-Compute** für Filterung/Deduplizierung; geringere Abdeckung obskurer Web-Fakten.
* **Qualitätsverlust:** **Stark Positiv** bei Reasoning, Sprache und Code; minimaler Verlust bei reinem Trivia-Wissen.
* **Quellen:**
  * FineWeb-Edu / Hugging Face Data Curation & DoReMi / RegMix (Domain Mixing Benchmarks).

### 1.5 UltraMemV2 / PEER (Parameter von FLOPs entkoppeln)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Packt den Großteil der 3B Parameter (z. B. ~2.4B) in **Product-Key Memory (PKM) Lookup-Tabellen** bzw. Millionen winziger 1-Neuron-Experten, während nur ~600M als dichter Backbone rechnen. UltraMemV2 optimiert dabei speziell den Memory-Access-Engpass gegenüber klassischem MoE. *(Hinweis: Bei Warm-Start über Sparse Upcycling in einen bestehenden Backbone einklinkbar.)*
* **Spart:** **Compute / FLOPs (4–6×)**, **Trainingszeit**, **Memory-Bandwidth**.
* **Zahlt:** **Implementierungs-Komplexität** (PKM-Lookup-Kernel), VRAM für Lookup-Tabellen (in INT8 ~2.4 GB).
* **Qualitätsverlust:** **Positiv bis Minimal** (überlegen bei Fakten/Recall, bei gleichem FLOP-Budget deutlich stärker als Dense).
* **Quellen:**
  * UltraMemV2: <https://arxiv.org/abs/2508.18756>
  * PEER (*Mixture of a Million Experts*): <https://arxiv.org/abs/2407.04153>

### 1.6 CoLA / CoLA-M (Low-Rank Activation Pretraining)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Ersetzt dichte MLP- und Attention-Projektionen durch kompakte Low-Rank-Autoencoder-Bottlenecks ($h = B\,\sigma(Ax)$) und nutzt damit die intrinsische Low-Rank-Struktur von Transformer-Aktivierungen aus. CoLA-M minimiert zusätzlich die im Backward vorgehaltenen Aktivierungen. *(Hinweis: Bei Modell A primär relevant, wenn der Warm-Start-Backbone ebenfalls CoLA nutzt oder per SVD/Low-Rank-Faktorisierung konvertiert wird.)*
* **Spart:** **Compute / FLOPs (~2×)**, **VRAM** (Gewichte + Aktivierungen), **Zeit (~1.86×)**.
* **Zahlt:** Geringere rohe Layer-Breite; erfordert Anpassung bei Standard-Checkpoints.
* **Qualitätsverlust:** **Keiner bis Minimal** (praktisch identische Perplexity bei halbierten FLOPs).
* **Quellen:**
  * CoLA: <https://arxiv.org/abs/2502.10940>

### 1.7 SuperBPE (Superword Tokenization)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Erweitert BPE über Leerzeichen hinweg auf ganze Phrasen (*Superwords*), wodurch derselbe Text bis zu **33 % weniger Tokens** benötigt. *(Hinweis: Bei Warm-Start mit anderem Tokenizer muss das Embedding/Head-Layer kurz adaptiert werden.)*
* **Spart:** **Compute / FLOPs (~1.4–1.5× pro Byte Text)**, **Zeit**, **effektive Kontextlänge**.
* **Zahlt:** Größeres Vokabular (durch Cut Cross-Entropy im VRAM abgefedert); minimal schwächer bei reinem Zeichen-/Spelling-Vergleich.
* **Qualitätsverlust:** **Positiv** (+4 % Schnitt / +8.2 % MMLU im Paper bei geringerem Compute).
* **Quellen:**
  * SuperBPE: <https://arxiv.org/abs/2503.13423>

### 1.8 FlashAttention-2 (4060 Ti) $\to$ FlashAttention-3 (H100) & FlexAttention
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Tiling + Online-Softmax direkt im GPU-SRAM ohne Materialisierung der $O(N^2)$-Attention-Matrix. Auf der 4060 Ti läuft FA-2; auf der H100 schaltet der Code automatisch auf FA-3 mit asynchroner TMA-Ausführung und FP8-Support um.
* **Spart:** **VRAM** ($O(N^2) \to O(N)$), **HBM-Bandwidth**, **Zeit (2–4×)**.
* **Zahlt:** Minimaler Setup-Aufwand; FA-3 Features sind Hopper-exklusiv.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * FlashAttention-3: <https://arxiv.org/abs/2407.08608>
  * FlashAttention-Repo: <https://github.com/Dao-AILab/flash-attention>

### 1.9 Liger-Kernel & `torch.compile` (Kernel Fusion)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Verschmilzt nicht-Attention-Operationen (RMSNorm, RoPE, SwiGLU, Cross-Entropy) über Triton-Kernels direkt im SRAM und vermeidet HBM-Roundtrips — entscheidend gegen die 288 GB/s-Drossel der 4060 Ti.
* **Spart:** **VRAM (40–60 % weniger Aktivierungen)**, **HBM-Bandwidth**, **Zeit (+15–30 %)**.
* **Zahlt:** Initiale Kompilierungszeit beim Start.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * Liger-Kernel Paper: <https://arxiv.org/abs/2410.10989>
  * Liger-Kernel Repo: <https://github.com/linkedin/Liger-Kernel>

### 1.10 Cut Cross-Entropy (CCE)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Berechnet Log-Sum-Exp und Target-Logit blockweise im SRAM, ohne jemals die riesige $(N \times V)$-Logit-Matrix im VRAM zu materialisieren.
* **Spart:** **VRAM (drastisch: z. B. 24 GB $\to$ ~1 MB bei großem Vocab)**, **HBM-Bandwidth**.
* **Zahlt:** Minimaler Recompute-Compute im Backward.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * Cut Cross-Entropy: <https://arxiv.org/abs/2411.09009>

### 1.11 Sequence Packing + Document Masking (EoS-Aligned)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Packt Dokumente lückenlos hintereinander auf volle Sequenzlänge und trennt sie per Block-Diagonal-Attention-Maske, sodass kein einziger FLOP für Padding-Tokens verschwendet wird.
* **Spart:** **Compute (bis 100 % nützliche Tokens)**, **Zeit**, **VRAM**.
* **Zahlt:** Etwas komplexere Data-Loader- & Masken-Logik.
* **Qualitätsverlust:** **Keiner** (oft leicht positiv durch sauberes EoS-Alignment).
* **Quellen:**
  * Modded-NanoGPT: <https://github.com/KellerJordan/modded-nanogpt>

### 1.12 Selective Activation Checkpointing + Fused Backward-Optimizer Hook (LOMO-Prinzip)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Verwirft nur speicherintensive, billig neuberechenbare Aktivierungen (auf 4060 Ti aggressiver als auf H100) und kombiniert das mit **Fused Backward + Optimizer Step** (`register_post_accumulate_grad_hook`): Jeder Layer wird im Backward sofort geupdated und sein Gradient direkt freigegeben.
* **Spart:** **VRAM (50–75 % Aktivierungen + nahezu komplette globale Gradienten-Allokation)**.
* **Zahlt:** **Compute (+10–25 % Recomputation)**; klassische Gradient Accumulation muss über Block-Puffer oder kleinere Steps gelöst werden.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * LOMO (*LOw-Memory Optimization*): <https://arxiv.org/abs/2310.10195>

### 1.13 FP8 Training (Tile-/Block-Wise Scaling) & Pure-BF16 + Stochastic Rounding
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Auf der 4060 Ti (`sm_89`) reduziert `torchao` FP8 / Pure-BF16 mit **Stochastic Rounding** (ohne FP32 Master Weights) die Gewichte auf 1–2 Bytes/Param; auf der H100 (`sm_90`) greift zusätzlich das feingranulare DeepSeek-V3 Block-Wise FP8-Schema ($1\times 128$ / $128\times 128$ mit FP32-Akkumulation) für bis zu 2× Speedup.
* **Spart:** **VRAM (2–3×)**, **Bandwidth**, **Compute/Zeit (auf H100 1.5–2×)**.
* **Zahlt:** Höhere Empfindlichkeit gegenüber Outliern; präzises Scaling nötig.
* **Qualitätsverlust:** **Keiner bis Minimal**.
* **Quellen:**
  * DeepSeek-V3 (FP8 Framework): <https://arxiv.org/abs/2412.19437>
  * TorchAO: <https://github.com/pytorch/ao>

### 1.14 Muon Optimizer (8-bit) + WSD-Schedule + muP + Checkpoint Averaging (EMA / LAWA)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** **Muon** orthogonalisiert Momentum-Matrizen von 2D-Hidden-Weights per Newton-Schulz-Iteration (spart 1 Optimizer-State und konvergiert bis ~1.5–2× schneller). **WSD** hält die LR lange konstant und erlaubt flexiblen Cooldown. **muP** überträgt Hyperparameter von Mini-Proxys auf 3B/H100. **EMA / LAWA** mittelt späte Checkpoints für kostenlosen Loss-Gewinn.
* **Spart:** **Steps / Zeit (Muon)**, **Optimizer-VRAM (1 Buffer statt 2)**, **Tuning-Compute (muP)**, **Experimentierkosten (WSD)**.
* **Zahlt:** Minimaler Newton-Schulz-Compute (<5 %) bei Muon; etwas Disk/RAM für Checkpoint-Mittelung.
* **Qualitätsverlust:** **Positiv** (besserer End-Loss pro FLOP).
* **Quellen:**
  * Muon: <https://github.com/KellerJordan/Muon>
  * Modded-NanoGPT (Muon + WSD-Varianten): <https://github.com/KellerJordan/modded-nanogpt>
  * muP: <https://github.com/microsoft/mup>
  * WSD-Schedule (MiniCPM / Hu et al.) & LAWA (Latest Weight Averaging).

### 1.15 Multi-Head Latent Attention (MLA) & Multi-Token Prediction (MTP)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** **MLA** komprimiert K/V gemeinsam in einen Low-Rank-Latent-Vektor (schrumpft Aktivierungs-VRAM und Bandwidth). **MTP** lässt das Modell 2–4 zukünftige Tokens gleichzeitig vorhersagen (dichteres Lernsignal + gratis Speculative-Decoding-Heads). *(Bei Warm-Start: MTP-Heads lassen sich leicht anflanschen; MLA erfordert ein passendes Basismodell oder Attention-Umbau.)*
* **Spart:** **VRAM & Bandwidth (MLA)**, **Tokens / Sample-Effizienz (MTP)**.
* **Zahlt:** ~5–10 % Extra-Compute für MTP-Heads; etwas Projektions-Compute bei MLA.
* **Qualitätsverlust:** **Positiv** (MTP steigert Reasoning deutlich).
* **Quellen:**
  * DeepSeek-V3 (MLA & MTP): <https://arxiv.org/abs/2412.19437>
  * Better & Faster LLMs via Multi-token Prediction: <https://arxiv.org/abs/2404.19737>

---

## 2. A-Rank Methoden (Performance-, Architektur- & Lossy-Booster)

### 2.1 Token-Selective Sparse Backward (Lossy Rho-1++) *(Mit Zeitplan-Regel)*
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Der Forward-Pass läuft über alle Tokens (voller Kontext bleibt erhalten), aber im Backward-Pass werden die Gradienten für die einfachsten **50–60 % der Tokens** physisch abgeschnitten, basierend auf dem Excess-Loss gegenüber einem externen Referenzmodell.
* **Aktivierungs-Regel:**
  * **0 % – 30/50 %:** **Deaktiviert** (100 % der Tokens im Backward für stabile Basisrepräsentationen).
  * **30/50 % – 90 %:** **Aktiviert** mit **50–60 % Cutoff** (nur die informativsten 40–50 % Tokens treiben Updates).
  * **90 % – 100 % (Cooldown):** **Deaktiviert** (wieder 100 % der Tokens für saubere Endkalibrierung).
* **Spart:** **Backward-Compute (1.4–1.8× in der aktiven Phase)**, **Trainingszeit**.
* **Zahlt:** Gather/Scatter-Overhead für unregelmäßige Tensor-Slices; leicht höheres Gradienten-Rauschen.
* **Qualitätsverlust:** **Minimal bis Moderat** (durch das Deaktivieren vor 30/50 % und ab 90 % stark abgefedert).
* **Quellen:**
  * Rho-1 (*Not All Tokens Are What You Need*): <https://arxiv.org/abs/2404.07965>

### 2.2 Ada FP8/INT8 mit FP16-Accumulate + COAT (Compression of Optimizer & Activations)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Zwingt die Ada-Tensor-Cores (`sm_89`) in den maximalen Durchsatz-Modus (FP8/INT8 mit FP16-Accumulate) und kombiniert das mit **COAT** (Dynamic Range Expansion für Optimizer-States + Mixed-Granularity FP8-Aktivierungsquantisierung).
* **Spart:** **Compute / Zeit (1.4–1.9× auf 4060 Ti)**, **VRAM (~1.54×)**, **HBM-Bandwidth (2×)**.
* **Zahlt:** Risiko numerischer Instabilität auf `sm_89` ohne sauberes Block-Scaling und Stochastic Rounding.
* **Qualitätsverlust:** **Minimal bis Moderat** (lossy, aber im Cooldown durch höhere Präzision stabilisierbar).
* **Quellen:**
  * COAT: <https://arxiv.org/abs/2410.19313>
  * TorchAO: <https://github.com/pytorch/ao>

### 2.3 APOLLO / APOLLO-Mini + 8-bit Adam (Fallback für dichte / nicht-Muon-Layer)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** **APOLLO-Mini** nutzt Rank-1-Subspace-Scaling statt voller AdamW-States und drückt Optimizer-VRAM praktisch auf SGD-Niveau (~0.1–0.3 GB selbst bei mehreren Milliarden Parametern), während 8-bit Adam für Embeddings/1D-Layer genutzt wird.
* **Spart:** **VRAM (Optimizer-States um 50–90 % reduziert)**.
* **Zahlt:** Minimaler Subspace-Projektions-Compute; bei extrem knapper GPU ersetzt es 8-bit Muon, falls dessen 1-Byte-Buffer noch zu groß ist.
* **Qualitätsverlust:** **Keiner bis Minimal**.
* **Quellen:**
  * APOLLO: <https://arxiv.org/abs/2412.05270>
  * bitsandbytes (8-bit Optimizer): <https://github.com/bitsandbytes-foundation/bitsandbytes>

### 2.4 Sparse Upcycling (Dense $\to$ UltraMemV2 / MoE) *(Spezifisch für Modell A)*
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Nimmt einen vortrainierten Dense-Backbone (z. B. nach dem HyperCloning/Minitron-Warm-Start) und klinkt neue UltraMemV2- oder MoE-Kapazität mit Zero-Init der Output-Projektion ein, sodass das Modell nahtlos von 1B/1.5B-Dense auf 3B-Sparse weiterlernt.
* **Spart:** **Compute / Zeit** (spart das komplette frühe Pretraining der Sparse-Struktur).
* **Zahlt:** Zusätzlicher VRAM für Experten/Tabellen; kurze Einlaufphase für Router/Keys.
* **Qualitätsverlust:** **Keiner** (schneller besser als reines Dense-Weitertraining).
* **Quellen:**
  * UltraMemV2: <https://arxiv.org/abs/2508.18756>

### 2.5 Modded-NanoGPT Boosters (QK-Norm, $\text{ReLU}^2$, Value Embeddings, U-Net Skips, Logit Softcap)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Architektur-Tricks aus den NanoGPT-Speedruns: QK-Norm stabilisiert extreme Lernraten, $\text{ReLU}^2$ liefert schärfere Gradienten als GELU/SwiGLU bei weniger Compute, U-Net-Skips schaffen Gradienten-Autobahnen. *(Bei Modell A nur anwendbar, wenn der Warm-Start-Backbone kompatibel ist oder Adapter/Layer neu initialisiert werden.)*
* **Spart:** **Tokens / Zeit (deutlich schnellere Konvergenz)**.
* **Zahlt:** Leicht mehr VRAM (bei Value Embeddings).
* **Qualitätsverlust:** **Positiv**.
* **Quellen:**
  * Modded-NanoGPT: <https://github.com/KellerJordan/modded-nanogpt>

### 2.6 Hybrid Sliding-Window Attention & Cross-Layer Attention (CLA) / YOCO
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Mischt lokale Sliding-Window-Attention mit wenigen Global-Layern (z. B. 3:1) und teilt K/V-Projektionen über Layer-Blöcke hinweg (**CLA**) bzw. nutzt einen globalen KV-Cache (**YOCO**).
* **Spart:** **Compute** ($O(N^2) \to O(N \cdot W)$), **HBM-Bandwidth**, **Aktivierungs-VRAM**.
* **Zahlt:** Leicht reduzierter Long-Range-Recall bei zu aggressivem Ratio.
* **Qualitätsverlust:** **Minimal**.
* **Quellen:**
  * Cross-Layer Attention (CLA): <https://arxiv.org/abs/2405.14361>
  * YOCO (*You Only Cache Once*): <https://arxiv.org/abs/2405.05254>

### 2.7 Aggressives Mixture-of-Depths (MoD) & H-Net Dynamic Chunking
* **Rank:** **A**
* **Kurz-Zusammenfassung:** **MoD** lässt einen Router pro Layer die unwichtigsten 50 % der Tokens per Residual-Skip durchreichen. **H-Net** komprimiert vorhersagbare Sequenzabschnitte dynamisch in gröbere Chunks für die mittleren Layer.
* **Spart:** **Compute / FLOPs (1.5–2.5×)**, **Aktivierungs-VRAM**.
* **Zahlt:** Routing-Komplexität und dynamische Batching-Logik.
* **Qualitätsverlust:** **Minimal** (bei 50 % Skip) bis **Positiv** (H-Net bei Code/Multilingual).
* **Quellen:**
  * Mixture-of-Depths: <https://arxiv.org/abs/2404.02258>
  * H-Net (*Dynamic Chunking*): <https://arxiv.org/abs/2507.07955>

### 2.8 Sequence Length Warmup & ScheduleFree+
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Startet das Training mit kurzen Sequenzen (z. B. 512 $\to$ 2048 $\to$ 4096) zur Reduktion von $O(N^2)$-Attention in frühen Steps. **ScheduleFree+** ist eine Alternative zum WSD-Schedule ohne festen Horizon.
* **Spart:** **Compute / Zeit (15–30 % in frühen Phasen)** bzw. Horizon-Planung.
* **Zahlt:** Etwas längerer Schedule-Anlauf für Long-Context; bei ScheduleFree+ leicht mehr VRAM (Extra-Iterate-Buffer).
* **Qualitätsverlust:** **Keiner bis Positiv**.
* **Quellen:**
  * Schedule-Free Repo: <https://github.com/facebookresearch/schedule_free>
