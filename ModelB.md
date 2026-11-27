# Modell B: 3B Transformer in 4–7 Tagen auf RTX 4060 Ti 16 GB (100 % From Scratch)

## Rahmenbedingungen & Zielsetzung
* **Ziel:** Training eines **3B-Parameter-Transformers komplett from Scratch** in **4 Tagen (max. 7 Tagen)** auf einer einzelnen **RTX 4060 Ti 16 GB (`sm_89`, Ada Lovelace, 288 GB/s)**.
* **Skalierbarkeit:** Der gesamte Code-Stack skaliert 1:1 per Flag auf eine **NVIDIA H100 80 GB (`sm_90`, Hopper, 3.35 TB/s)** (FlashAttention-3, natives Hopper-FP8, reduziertes/selektives Checkpointing).
* **Regel für Modell B:** **100 % From Scratch — KEINE externen Modelle erlaubt!**
  * **Explizit ausgeschlossen:** Keine Teacher-Distillation, kein Minitron-Pruning fremder Checkpoints, kein Warm-Start von fremden Open-Weights-Modellen und kein externes Referenzmodell für JEST/Rho-1++.
  * Stattdessen nutzt Modell B **Self-Growth** (klein starten, selbst hochklonen), **Self-Bootstrapping** (eigenes Stage-1-Modell oder Online-EMA-Loss für JEST/Rho-1++) und **radikale Architektur-Entkopplung** (CoLA + UltraMemV2 + SuperBPE).

---

## 4-Tage-Phasenplan (inkl. Sparse-Backward-Schedule)

| Trainingsphase | Fortschritt | Token-Selective Sparse Backward | Hauptfokus |
| :--- | :--- | :--- | :--- |
| **Phase 1: Winziger CoLA-Backbone from Scratch** | **0 % – 30/50 %** (Tag 1 – Tag 1.5) | **Aus (100 % der Tokens)** | Training eines **150M–300M CoLA-Backbones** ab Random Init (~10× schneller als 3B Dense!) auf SuperBPE-Daten mit Sequence Length Warmup. Volle Gradienten bauen saubere Basisrepräsentationen auf. |
| **Phase 2: Self-Growth auf 3B** | **Übergang** (Tag 1.5 – Tag 2) | **Aus (kurzes Stabilisieren)** | **HyperCloning + GStack** expandieren den eigenen Stage-1-Backbone funktions-erhaltend auf **600M CoLA-Backbone** (~300M FLOP-Äquivalent); **Sparse Upcycling** klinkt **2.4B UltraMemV2-Tabellen** (Zero-Init Output) ein $\implies$ **3.0B Total**. |
| **Phase 3: Haupttraining mit Sparse Backward** | **30/50 % – 90 %** (Tag 2 – Tag 3.6) | **Aktiv (50–60 % Cutoff)** | **Self-Scored Token-Selective Sparse Backward** (Excess-Loss gegen eigenes Stage-1-Modell bzw. Online-EMA-Loss) schneidet die einfachsten 50–60 % der Tokens im Backward ab. Parallel laufen **Self-JEST**, **MTP** und **FP8/INT8**. |
| **Phase 4: WSD-Cooldown & Polishing** | **90 % – 100 %** (Tag 3.6 – Tag 4) | **Aus (100 % der Tokens)** | Steiler LR-Decay (**WSD-Schedule**) auf hochkuratierten Daten mit wieder **100 % aktiven Gradienten** für maximale Endqualität, abgeschlossen durch **Checkpoint Averaging / EMA / LAWA**. |

---

## 1. S-Rank Methoden (Pflicht-Stack)

### 1.1 UltraMemV2 / PEER (Parameter von FLOPs entkoppeln)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Das Herzstück von Modell B: ~2.4B der 3.0B Parameter liegen in **Product-Key Memory (PKM) Lookup-Tabellen** (Millionen winziger 1-Neuron-Experten), nur ~600M bilden den dichten Backbone. Pro Token werden nur Top-$k$ Keys geholt, wodurch ein 3B-Modell mit dem Compute eines winzigen Modells läuft und gleichzeitig den Memory-Access-Engpass klassischer MoEs vermeidet.
* **Spart:** **Compute / FLOPs (4–6×)**, **Trainingszeit**, **Memory-Bandwidth**.
* **Zahlt:** Custom PKM-Lookup-Kernel-Komplexität; ~2.4 GB VRAM für Tabellen (in INT8).
* **Qualitätsverlust:** **Positiv bis Minimal** (bei Fakten/Recall deutlich stärker als Dense bei gleichem FLOP-Budget).
* **Quellen:**
  * UltraMemV2: <https://arxiv.org/abs/2508.18756>
  * PEER (*Mixture of a Million Experts*): <https://arxiv.org/abs/2407.04153>

### 1.2 CoLA / CoLA-M (Low-Rank Activation Pretraining)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Ersetzt im 600M-Backbone alle dichten MLP- und Attention-Projektionen nativ von Anfang an durch Low-Rank-Autoencoder-Bottlenecks ($h = B\,\sigma(Ax)$). Dadurch sinkt der effektive Backbone-Compute von ~600M auf **~300M-Dense-Äquivalent**, während **CoLA-M** zusätzlich Aktivierungen im Backward spart.
* **Spart:** **Compute / FLOPs (~2×)**, **VRAM** (Gewichte + Aktivierungen), **Zeit (~1.86×)**.
* **Zahlt:** Nicht 1:1 kompatibel zu Standard-Llama-Weights (für From-Scratch egal).
* **Qualitätsverlust:** **Keiner bis Minimal** (im Paper praktisch identische Perplexity bei halben FLOPs).
* **Quellen:**
  * CoLA: <https://arxiv.org/abs/2502.10940>

### 1.3 Self-Growth via HyperCloning + GStack from Scratch *(Spezifisch für Modell B)*
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Da keine externen Checkpoints genutzt werden dürfen, startet Modell B als eigener **150M–300M-CoLA-Backbone** und verbringt dort die ersten ~30–50 % des Trainings extrem schnell. Danach wird exakt dieses eigene Modell per **HyperCloning** (Breite) und **GStack** (Tiefe) funktions-erhaltend auf den 600M-Backbone hochskaliert.
* **Spart:** **Compute / Zeit (2.5–4×)** gegenüber direktem Start in der Endgröße.
* **Zahlt:** Kurzer Übergangs-Loss-Spike und LR-Warmup-Reset beim Wachstumsschritt.
* **Qualitätsverlust:** **Keiner bis Minimal**.
* **Quellen:**
  * HyperCloning: <https://arxiv.org/abs/2409.12903>
  * GStack (*Stacking Your Transformers*): <https://arxiv.org/abs/2405.15319>

### 1.4 SuperBPE (Superword Tokenization)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Eigener from-scratch trainierter BPE-Tokenizer, der in Phase 2 über Leerzeichen hinweg ganze Phrasen (*Superwords*) zu einem Token verschmilzt. Reduziert die Token-Anzahl pro Korpus um bis zu **33 %** und passt perfekt zu Cut Cross-Entropy.
* **Spart:** **Compute / FLOPs (~1.4–1.5× pro Byte Text)**, **Zeit**, **effektive Kontextlänge**.
* **Zahlt:** Größeres Vokabular (durch CCE im VRAM neutralisiert); minimal schwächer bei reinem Buchstabier-/Zeichen-Level-Handling.
* **Qualitätsverlust:** **Positiv** (+4 % Schnitt / +8.2 % MMLU im Paper bei geringerem Compute).
* **Quellen:**
  * SuperBPE: <https://arxiv.org/abs/2503.13423>

### 1.5 Self-Bootstrapped JEST (Joint Example Selection) *(Spezifisch für Modell B)*
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Nutzt das eigene Stage-1-Modell (150M–300M aus Phase 1) als Proxy-Scorer, um aus Super-Batches genau die Sub-Batches mit maximaler gemeinsamer Lernbarkeit (*Joint Learnability*) für den 3B-Run auszuwählen — komplett ohne fremde Modelle.
* **Spart:** **Iterationen / Compute (bis 5–13×)**, **Trainingszeit**.
* **Zahlt:** Etwas Vorlauf-/Proxy-Compute auf dem eigenen Stage-1-Modell.
* **Qualitätsverlust:** **Positiv**.
* **Quellen:**
  * JEST: <https://arxiv.org/abs/2406.17711>

### 1.6 Hyper-Synthetic & Kuratiertes Daten-Curriculum + Optimal Mixing
* **Rank:** **S**
* **Kurz-Zusammenfassung:** From-Scratch-Training ausschließlich auf hochdichten, semantisch deduplizierten und kuratierten Korpora (z. B. FineWeb-Edu, DCLM, synthetische Lehrbücher), um jedes kostbare Token im 4-Tage-Budget maximal auszunutzen.
* **Spart:** **Tokens / Compute (2–8× weniger Tokens)**.
* **Zahlt:** Offline-CPU-Preprocessing; schmaleres Wissen bei obskuren Web-Nischen.
* **Qualitätsverlust:** **Stark Positiv** bei Sprache, Logik und Code.
* **Quellen:**
  * FineWeb-Edu / Hugging Face Data Curation & DoReMi / RegMix (Domain Mixing Benchmarks).

### 1.7 Multi-Head Latent Attention (MLA) & Multi-Token Prediction (MTP)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** **MLA** komprimiert K/V nativ in einen Low-Rank-Latent-Vektor (spart Aktivierungs-VRAM und Bandwidth). **MTP** trainiert 2–4 zukünftige Tokens gleichzeitig pro Position, was gerade beim From-Scratch-Training ein deutlich dichteres Lernsignal liefert.
* **Spart:** **VRAM & Bandwidth (MLA)**, **Tokens / Sample-Effizienz (MTP)**.
* **Zahlt:** ~5–10 % Extra-Compute für MTP-Heads; leichte RoPE-Entkopplungs-Komplexität bei MLA.
* **Qualitätsverlust:** **Positiv**.
* **Quellen:**
  * DeepSeek-V3 (MLA & MTP): <https://arxiv.org/abs/2412.19437>
  * Better & Faster LLMs via Multi-token Prediction: <https://arxiv.org/abs/2404.19737>

### 1.8 FlashAttention-2 (4060 Ti) $\to$ FlashAttention-3 (H100) & FlexAttention
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Tiling + Online-Softmax im GPU-SRAM ohne $O(N^2)$-Attention-Matrix. Läuft mit FA-2 auf der 4060 Ti und schaltet auf H100 automatisch auf FA-3 mit asynchroner TMA-Ausführung.
* **Spart:** **VRAM** ($O(N^2) \to O(N)$), **HBM-Bandwidth**, **Zeit (2–4×)**.
* **Zahlt:** Minimaler Setup-Aufwand.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * FlashAttention-3: <https://arxiv.org/abs/2407.08608>
  * FlashAttention-Repo: <https://github.com/Dao-AILab/flash-attention>

### 1.9 Liger-Kernel & `torch.compile` (Kernel Fusion)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Verschmilzt RMSNorm, RoPE, SwiGLU/$\text{ReLU}^2$ und Loss-Ops in Triton-SRAM-Kernels und schützt die 4060 Ti vor ihrem 288 GB/s-Bandwidth-Limit.
* **Spart:** **VRAM (40–60 %)**, **HBM-Bandwidth**, **Zeit (+15–30 %)**.
* **Zahlt:** Compile-Zeit beim Start.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * Liger-Kernel Paper: <https://arxiv.org/abs/2410.10989>
  * Liger-Kernel Repo: <https://github.com/linkedin/Liger-Kernel>

### 1.10 Cut Cross-Entropy (CCE)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Berechnet Cross-Entropy blockweise im SRAM, ohne die $(N \times V)$-Logit-Matrix im VRAM abzulegen — essenziell bei großem SuperBPE-Vokabular.
* **Spart:** **VRAM (z. B. 24 GB $\to$ ~1 MB)**, **HBM-Bandwidth**.
* **Zahlt:** Minimaler Recompute-Compute im Backward.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * Cut Cross-Entropy: <https://arxiv.org/abs/2411.09009>

### 1.11 Sequence Packing + Document Masking (EoS-Aligned)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Füllt Sequenzen ohne Padding und isoliert Dokumente per Block-Diagonal-Maske.
* **Spart:** **Compute (0 % Padding-Verschwendung)**, **Zeit**, **VRAM**.
* **Zahlt:** Masken-/Data-Loader-Logik.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * Modded-NanoGPT: <https://github.com/KellerJordan/modded-nanogpt>

### 1.12 Selective Activation Checkpointing + Fused Backward-Optimizer Hook (LOMO-Prinzip)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Recompute selektiver Aktivierungen + sofortiges Layer-Update im Backward-Pass via `register_post_accumulate_grad_hook`, sodass weder volle Aktivierungen noch globale Gradienten-Tensoren im VRAM liegen.
* **Spart:** **VRAM (50–75 % Aktivierungen + nahezu komplette Gradienten-Allokation)**.
* **Zahlt:** **Compute (+10–25 % Recomputation)**.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * LOMO (*LOw-Memory Optimization*): <https://arxiv.org/abs/2310.10195>

### 1.13 FP8 Training (Tile-/Block-Wise Scaling) & Pure-BF16 + Stochastic Rounding
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Pure-BF16 / `torchao` FP8 mit **Stochastic Rounding** (ohne FP32 Master Weights) auf `sm_89`; auf H100 (`sm_90`) nativer Wechsel auf DeepSeek-V3 Block-Wise FP8.
* **Spart:** **VRAM (2–3×)**, **Bandwidth**, **Compute/Zeit (auf H100 1.5–2×)**.
* **Zahlt:** Outlier-Sensitivität, Block-Scaling-Setup.
* **Qualitätsverlust:** **Keiner bis Minimal**.
* **Quellen:**
  * DeepSeek-V3 (FP8 Framework): <https://arxiv.org/abs/2412.19437>
  * TorchAO: <https://github.com/pytorch/ao>

### 1.14 Muon Optimizer (8-bit) + WSD-Schedule + muP + Checkpoint Averaging (EMA / LAWA)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** **8-bit Muon** für den 600M-CoLA-Backbone (1 Momentum-Buffer statt 2, bis ~1.5–2× schnellere Konvergenz) + Sparse Adam/SGD für die 2.4B UltraMemV2-Tabellen. **muP** erlaubt das Hyperparameter-Tuning am winzigen Stage-1-Modell auf der 4060 Ti mit 1:1-Übertragbarkeit auf 3B/H100. **WSD** + **EMA/LAWA** sichern den sauberen Cooldown.
* **Spart:** **Steps / Zeit (Muon)**, **Optimizer-VRAM**, **Tuning-Compute (muP)**.
* **Zahlt:** <5 % Newton-Schulz-Compute; etwas Disk/RAM für Checkpoints.
* **Qualitätsverlust:** **Positiv**.
* **Quellen:**
  * Muon: <https://github.com/KellerJordan/Muon>
  * Modded-NanoGPT: <https://github.com/KellerJordan/modded-nanogpt>
  * muP: <https://github.com/microsoft/mup>

---

## 2. A-Rank Methoden (Performance-, Architektur- & Lossy-Booster)

### 2.1 Self-Scored Token-Selective Sparse Backward (Lossy Rho-1++) *(Mit Zeitplan-Regel)*
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Da kein externes Referenzmodell erlaubt ist, bewertet das **eigene Stage-1-Modell** (oder ein Online-EMA-Loss-Vergleich) den Excess-Loss jedes Tokens. Im Forward laufen 100 % der Tokens, im Backward werden die einfachsten **50–60 % der Tokens** physisch abgeschnitten.
* **Aktivierungs-Regel:**
  * **0 % – 30/50 %:** **Deaktiviert** (100 % der Tokens im Backward — Stage 1 & früher Growth-Stabilisierung).
  * **30/50 % – 90 %:** **Aktiviert** mit **50–60 % Cutoff** (nur die informativsten 40–50 % Tokens gehen durch den vollen Backward-Pass).
  * **90 % – 100 % (Cooldown):** **Deaktiviert** (wieder 100 % der Tokens für maximale Endqualität).
* **Spart:** **Backward-Compute (1.4–1.8× in der aktiven Phase)**, **Trainingszeit**.
* **Zahlt:** Gather/Scatter-Overhead; leicht verrauschte Gradienten.
* **Qualitätsverlust:** **Minimal bis Moderat** (durch späte Aktivierung und volle Gradienten im Cooldown abgefangen).
* **Quellen:**
  * Rho-1 (*Not All Tokens Are What You Need*): <https://arxiv.org/abs/2404.07965>

### 2.2 Modded-NanoGPT Boosters (QK-Norm, $\text{ReLU}^2$, Value Embeddings, U-Net Skips, Logit Softcap)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Beim From-Scratch-Design von Modell B können alle Speedrun-Architektur-Tricks nativ eingebaut werden: QK-Norm gegen LR-Explosionen, $\text{ReLU}^2$ für billigere und schärfere MLPs, U-Net-Skip-Connections zwischen Block $i$ und $L-i$, Value Embeddings und Logit Softcap.
* **Spart:** **Tokens / Zeit (massiv schnellere Konvergenz auf Small/Mid-Scale)**.
* **Zahlt:** Leicht erhöhter VRAM durch Value Embeddings.
* **Qualitätsverlust:** **Positiv**.
* **Quellen:**
  * Modded-NanoGPT: <https://github.com/KellerJordan/modded-nanogpt>

### 2.3 Ada FP8/INT8 mit FP16-Accumulate + COAT (Compression of Optimizer & Activations)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Nutzt den maximalen `sm_89`-Durchsatz (FP8/INT8 mit FP16-Accumulate) zusammen mit **COAT** (Dynamic Range Expansion für Optimizer-States + Mixed-Granularity FP8-Aktivierungen).
* **Spart:** **Compute / Zeit (1.4–1.9× auf 4060 Ti)**, **VRAM (~1.54×)**, **HBM-Bandwidth (2×)**.
* **Zahlt:** Höhere Anforderung an Block-Scaling & Stochastic Rounding auf `sm_89`.
* **Qualitätsverlust:** **Minimal bis Moderat**.
* **Quellen:**
  * COAT: <https://arxiv.org/abs/2410.19313>
  * TorchAO: <https://github.com/pytorch/ao>

### 2.4 Sparse Upcycling (Eigener Stage-1-Backbone $\to$ 3B UltraMemV2) *(Spezifisch für Modell B)*
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Klinkt beim Übergang in Phase 2 die 2.4B UltraMemV2-Lookup-Tabellen (mit Zero-Init der Output-Projektion) nahtlos in den eigenen, frisch trainierten 600M-CoLA-Backbone ein, ohne den bereits gelernten Loss zu zerstören.
* **Spart:** **Compute / Zeit** (Lookup-Tabellen müssen nicht während Phase 1 mitgeschleppt werden).
* **Zahlt:** ~2.4 GB VRAM ab Phase 2.
* **Qualitätsverlust:** **Keiner**.
* **Quellen:**
  * UltraMemV2: <https://arxiv.org/abs/2508.18756>

### 2.5 APOLLO / APOLLO-Mini + 8-bit Adam (Optimizer-VRAM-Schutz)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Rank-1-Subspace-Scaling (**APOLLO-Mini**) reduziert AdamW-States auf <0.3 GB und dient als ultrakompakter Fallback, falls die Kombination aus 600M-Backbone, 2.4B-UltraMem-Tabellen und langen Sequenzen den 16-GB-VRAM sprengt.
* **Spart:** **VRAM (Optimizer-States um 50–90 % reduziert)**.
* **Zahlt:** Minimaler Subspace-Projektions-Overhead.
* **Qualitätsverlust:** **Keiner bis Minimal**.
* **Quellen:**
  * APOLLO: <https://arxiv.org/abs/2412.05270>
  * bitsandbytes (8-bit Optimizer): <https://github.com/bitsandbytes-foundation/bitsandbytes>

### 2.6 Hybrid Sliding-Window Attention & Cross-Layer Attention (CLA) / YOCO
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Kombiniert lokale Sliding-Window-Attention mit wenigen Full-Attention-Layern (z. B. 3:1) und teilt K/V-Projektionen über Layer-Blöcke (**CLA**) bzw. via globalem KV-Cache (**YOCO**).
* **Spart:** **Compute** ($O(N^2) \to O(N \cdot W)$), **HBM-Bandwidth**, **Aktivierungs-VRAM**.
* **Zahlt:** Leicht reduzierter Long-Range-Recall bei zu wenigen Global-Layern.
* **Qualitätsverlust:** **Minimal**.
* **Quellen:**
  * Cross-Layer Attention (CLA): <https://arxiv.org/abs/2405.14361>
  * YOCO (*You Only Cache Once*): <https://arxiv.org/abs/2405.05254>

### 2.7 Aggressives Mixture-of-Depths (MoD) & H-Net Dynamic Chunking
* **Rank:** **A**
* **Kurz-Zusammenfassung:** **MoD** skippt in jedem 2. Layer 50 % der Tokens per Residual; **H-Net** chunked vorhersagbare Byte-/Token-Sequenzen dynamisch in den mittleren Layern. Perfekt stapelbar mit CoLA für noch weniger Backbone-FLOPs.
* **Spart:** **Compute / FLOPs (1.5–2.5×)**, **Aktivierungs-VRAM**.
* **Zahlt:** Router- & Dynamic-Batching-Komplexität.
* **Qualitätsverlust:** **Minimal** (MoD 50 %) bis **Positiv** (H-Net bei Code/Multilingual).
* **Quellen:**
  * Mixture-of-Depths: <https://arxiv.org/abs/2404.02258>
  * H-Net (*Dynamic Chunking*): <https://arxiv.org/abs/2507.07955>

### 2.8 Sequence Length Warmup & ScheduleFree+
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Startet Phase 1 mit kurzen Sequenzen (512 $\to$ 2048 $\to$ 4096) zur Reduktion von $O(N^2)$-Attention in frühen Steps. **ScheduleFree+** bietet eine Alternative zu WSD ohne festen Training-Horizon.
* **Spart:** **Compute / Zeit (15–30 % in Phase 1)** bzw. Horizon-Planung.
* **Zahlt:** Leicht mehr VRAM bei ScheduleFree+ (Extra-Iterate-Buffer).
* **Qualitätsverlust:** **Keiner bis Positiv**.
* **Quellen:**
  * Schedule-Free Repo: <https://github.com/facebookresearch/schedule_free>
