# Run 2: Sparsamer Looped Transformer (Hauptplan, ohne CTM)

## Rahmenbedingungen & Zielsetzung
* **Ziel:** Ausbau des Run-1-Modells ([ModelA.md](file:///home/henry/Documents/Henry/minotaur/ModelA.md) oder [ModelB.md](file:///home/henry/Documents/Henry/minotaur/ModelB.md), **3B Parameter**) zu einem **Looped-Reasoning-Transformer** auf einer einzelnen **RTX 4060 Ti 16 GB (`sm_89`, Ada Lovelace, 288 GB/s)**.
* **Skalierbarkeit:** Der gesamte Code-Stack skaliert 1:1 per Flag auf eine **NVIDIA H100 80 GB (`sm_90`, Hopper, 3.35 TB/s)** (FlashAttention-3, natives Hopper-FP8, reduziertes/selektives Checkpointing).
* **Abgrenzung:** Dieser Hauptplan verzichtet bewusst auf experimentelle CTM-Module (Sakana AI). Die CTM-Light-Erweiterung ist isoliert in [Run2_Exp.md](file:///home/henry/Documents/Henry/minotaur/Run2_Exp.md) dokumentiert.

### Hardware-Realität auf der RTX 4060 Ti (Warum Loops hier funktionieren — oder explodieren)
* **Der 4060-Ti-Vorteil (32 MB L2-Cache vs. 288 GB/s GDDR6):** Wenn ein kompakter **CoLA-Core** mehrfach hintereinander geloopt wird, bleiben Gewichte und Zwischenstände deutlich heißer im 32 MB L2-Cache statt ständig neue Layer aus dem langsamen GDDR6-VRAM zu streamen.
* **Die 3 Fallen beim naiven Loopen:**
  1. **FLOP-Explosion:** $R$ Loops = $R\times$ Forward- und $R\times$ Backward-FLOPs.
  2. **BPTT-VRAM-Explosion:** Volles Backpropagation Through Time über $R$ Loops vervielfacht den Aktivierungs-VRAM um Faktor $R$.
  3. **Fakten-Mangel:** Gewicht-Sharing verbessert *Reasoning/Coding*, speichert aber keine zusätzlichen Weltwissens-Fakten.

---

## Zwei Einstiegspfade

### Pfad 1 (Empfohlen): Loop-Upcycling aus Run 1 (1–2 Tage Continual Pretraining)
1. **Checkpoint übernehmen:** Lädt den fertigen 3B-Checkpoint aus Run 1 (Model A oder Model B).
2. **Aufteilung:** Die äußeren Layer werden als **Prelude** & **Coda** fixiert; der mittlere Block wird als **Recurrent Core** definiert.
3. **Zero-Init Loop-Gate:** Das Loop-Residual-Gate startet bei `0` $\implies$ bei $R=1$ verhält sich das Modell zu Beginn **exakt wie Run 1** (kein initialer Qualitätsverlust).
4. **Curriculum:** Loop-Tiefe schrittweise hochfahren ($R = 1 \to 2 \to 4$), parallel zu **MoR** und **Truncated BPTT**.

### Pfad 2 (Optional): Looped From Scratch (4–7 Tage)
* Direktes From-Scratch-Training im [ModelB.md](file:///home/henry/Documents/Henry/minotaur/ModelB.md)-Setup (150M–300M CoLA-Backbone $\to$ HyperCloning/GStack auf 600M Looped Core + 2.4B UltraMemV2). Spart keinen Gesamt-Compute gegenüber Pfad 1, vermeidet aber jeglichen Umweg über ein ungelooctes 3B-Zwischenstadium.

---

## 1. Kern-Techniken für Run 2

### 1.1 Sandwich-Layout (Prelude $\to$ Core $\to$ Coda) + Input-Injection & Per-Token Loop-Gate
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Nur die mittleren Layer werden geloopt; **Prelude** (Eingangs-Encoding) und **Coda** (Readout) bleiben unshared. In jedem Loop wird der Prelude-Output leicht re-injiziert (*Input-Injection*, Huginn-Prinzip), damit der Core nicht „vergisst“, was die ursprüngliche Eingabe war. Ein leichtgewichtiges **Per-Token-Gate** (wie bei GLM-5.4 / Inkling 2 Mini) reguliert die Residual-Mischung und verhindert Hidden-State-Explosionen über viele Ticks.
* **Spart:** **VRAM** (weniger physische Gewichte für hohe logische Tiefe), **Stabilität** (keine Divergenz bei $R \ge 3$).
* **Zahlt:** Minimaler Gate-Compute (<1 %).
* **Qualitätsverlust:** **Positiv** (deutlich stabiler als naives Layer-Sharing).
* **Quellen:**
  * Huginn (*Scaling up Test-Time Compute with Latent Reasoning*): <https://arxiv.org/abs/2502.05171>

### 1.2 Randomized Loop Depth + Truncated BPTT ($k=1\text{--}2$ mit `torch.no_grad()`)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Pro Trainingsschritt wird eine zufällige Loop-Tiefe $R$ gesampelt (z. B. $R \in [1, 6]$). Die ersten $R - k$ Loops laufen komplett unter **`torch.no_grad()`** (keine Aktivierungen gespeichert!); Backpropagation läuft **nur durch die letzten $k = 1$ oder $k = 2$ Loops**.
* **Spart:** **Loop-Aktivierungs-VRAM (75–85 %!)**, **Backward-FLOPs (50–70 %!)**; ermöglicht zudem flexibles Test-Time-Scaling zur Inferenz.
* **Zahlt:** Frühere Loops erhalten nur indirekte Gradienten über den Zustand, was bei $k=1$ leicht verrauschter sein kann.
* **Qualitätsverlust:** **Minimal** (bei $k=2$ praktisch verlustfrei, bei $k=1$ leicht lossy).
* **Quellen:**
  * Huginn: <https://arxiv.org/abs/2502.05171>

### 1.3 Mixture-of-Recursions (MoR) + Selektives KV-Caching
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Ein winziger Router weist jedem Token dynamisch seine Loop-Tiefe zu: einfache Tokens (Stopwords, Satzzeichen) steigen bereits nach **Loop 1** aus; nur die komplexesten **25–40 %** der Tokens laufen durch Loop 2 bis 4. Attention und KV-Cache werden in tieferen Loops **selektiv nur für noch aktive Tokens** berechnet und gecacht.
* **Spart:** **Forward- & Backward-FLOPs (2–3×!)**, **KV-Cache- & Aktivierungs-VRAM**.
* **Zahlt:** Dynamisches Token-Routing (Gather/Scatter-Overhead) und Router-Stabilisierung.
* **Qualitätsverlust:** **Keiner bis Positiv** (bei gleichem FLOP-Budget deutlich besser als starre Loops).
* **Quellen:**
  * Mixture-of-Recursions (MoR): <https://arxiv.org/abs/2507.10524>

### 1.4 Loop-Specific Tiny Adapters (Step-Embeddings + Per-Loop RMSNorms / optional Rank-4/8 Loop-LoRA)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Jede Loop-Iteration $t$ erhält eigene **RMSNorm-Gewichte** und ein winziges **Step-Embedding** (optional einen Rank-4/8 Loop-LoRA), sodass derselbe Core weiß, ob er gerade im „Planen“- (Loop 1) oder „Verfeinern“-Schritt (Loop 4) rechnet.
* **Spart:** **Konvergenz-Zeit** (bricht die Starrheit von reinem Weight-Sharing auf).
* **Zahlt:** Winziger VRAM-Overhead (<1 % Extra-Parameter). *(Hinweis: Step-Norms + Step-Embeddings extrapolieren besser auf ungesehene Loop-Tiefen als feste Loop-LoRAs.)*
* **Qualitätsverlust:** **Positiv** (starker Qualitätsgewinn bei vernachlässigbaren Kosten).
* **Quellen:**
  * Huginn (Step-Conditioning-Prinzip): <https://arxiv.org/abs/2502.05171>

### 1.5 UltraMemV2 (Fakten außerhalb des Loops) + CoLA-Core (Reasoning im Loop)
* **Rank:** **S**
* **Kurz-Zusammenfassung:** Perfekte Arbeitsteilung mit Run 1: Die **2.4B UltraMemV2-Lookup-Parameter** bleiben im Prelude/Coda (oder werden **genau 1×** abgefragt, *nicht* in jedem Loop neu!) und liefern Weltwissen/Fakten. Der kompakte **CoLA-Core** (~300–600M Low-Rank-Backbone) übernimmt das iterative Denken im Loop und bleibt auf der 4060 Ti optimal im 32 MB L2-Cache.
* **Spart:** **Compute / FLOPs (4–6×)**, **HBM-Bandwidth**.
* **Zahlt:** PKM-Lookup-Struktur; ~2.4 GB VRAM für INT8-Tabellen.
* **Qualitätsverlust:** **Positiv**.
* **Quellen:**
  * UltraMemV2: <https://arxiv.org/abs/2508.18756>
  * CoLA: <https://arxiv.org/abs/2502.10940>

### 1.6 Doppelte Token-Sparsity: MoR (Forward) × Token-Selective Sparse Backward (Rho-1++)
* **Rank:** **A**
* **Kurz-Zusammenfassung:** Kombiniert adaptives Looping im Forward (**MoR**) mit **Token-Selective Sparse Backward** (Abschneiden der einfachsten **50–60 %** der Tokens im Backward-Pass):
  * **0 % – 30/50 % von Run 2:** Sparse Backward **Aus** (100 % der Tokens für stabile Loop-Anpassung).
  * **30/50 % – 90 % von Run 2:** Sparse Backward **Aktiv (50–60 % Cutoff)** + aktives MoR.
  * **90 % – 100 % (WSD-Cooldown):** Sparse Backward wieder **Aus** (100 % der Tokens für sauberes Polishing).
* **Spart:** **Backward-Compute (zusätzlich 1.4–1.8× in der aktiven Phase)**, **Trainingszeit**.
* **Zahlt:** Gather/Scatter-Logik; leicht höheres Gradienten-Rauschen in der Mittelphase.
* **Qualitätsverlust:** **Minimal bis Moderat** (durch Cooldown mit 100 % Tokens abgefedert).
* **Quellen:**
  * Rho-1: <https://arxiv.org/abs/2404.07965>
  * Mixture-of-Recursions: <https://arxiv.org/abs/2507.10524>

### 1.7 KV-Sharing über Loops (CLA / YOCO) + Loop-kompatibler Fused Grad-Hook (LOMO-Prinzip)
* **Rank:** **S / A**
* **Kurz-Zusammenfassung:**
  * **KV-Sharing (CLA/YOCO):** K/V-Projektionen für inaktive/kontextuelle Tokens werden aus Loop 1 wiederverwendet oder über einen globalen Cache geteilt, statt in jedem Loop komplett neu gerechnet zu werden.
  * **Loop-kompatibler Fused Grad-Hook:** Beim geloopten Core werden dieselben Gewichte im Backward $k$-mal durchlaufen. Der `register_post_accumulate_grad_hook` (LOMO-Prinzip) muss daher so gesteuert werden, dass er erst **nach dem letzten der $k$ Loop-Backwards** das Optimizer-Update ausführt und die Gradienten freigibt (bei $k=1$ sofort!).
* **Spart:** **HBM-Bandwidth & KV-VRAM (CLA/YOCO)**, **Gradienten-VRAM (LOMO-Hook)**.
* **Zahlt:** Etwas mehr Hook-/Counter-Logik im Backward; leicht weniger flexible Attention über tiefe Loops bei statischem KV-Reuse.
* **Qualitätsverlust:** **Keiner bis Minimal**.
* **Quellen:**
  * Cross-Layer Attention (CLA): <https://arxiv.org/abs/2405.14361>
  * YOCO: <https://arxiv.org/abs/2405.05254>
  * LOMO: <https://arxiv.org/abs/2310.10195>

### 1.8 Übernommener Run-1-Unterbau (Standard-Basis)
Run 2 erbt unverändert alle Systems-, Optimizer- und Präzisions-Bausteine aus Run 1:
* **Attention & Kernels:** FlashAttention-2 (4060 Ti) $\to$ FlashAttention-3 (H100) (<https://arxiv.org/abs/2407.08608>), Liger-Kernel (<https://arxiv.org/abs/2410.10989>), Cut Cross-Entropy (<https://arxiv.org/abs/2411.09009>).
* **Tokenizer & Data:** SuperBPE (<https://arxiv.org/abs/2503.13423>), Sequence Packing + Document Masking.
* **Optimizer & Präzision:** 8-bit Muon (<https://github.com/KellerJordan/Muon>) / APOLLO-Mini (<https://arxiv.org/abs/2412.05270>), Ada FP8/INT8 + COAT (<https://arxiv.org/abs/2410.19313>), WSD-Schedule + EMA/LAWA.

---

## 2. Zeitplan für Loop-Upcycling (1–2 Tage auf RTX 4060 Ti)

| Phase | Fortschritt | Loop-Tiefe & BPTT | Sparse Backward | Fokus |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Loop-Warmup** | **0 % – 30/50 %** | $R = 1 \to 2$, TBPTT $k=2$ | **Aus (100 % Tokens)** | Zero-Init Loop-Gate langsam öffnen; Step-Norms & MoR-Router einlernen. |
| **Phase 2: Deep Looped + Sparse** | **30/50 % – 90 %** | $R \in [1, 4\text{--}6]$, TBPTT $k=1\text{--}2$ (`no_grad` vorher), MoR aktiv | **Aktiv (50–60 % Cutoff)** | Maximaler Reasoning-Durchsatz auf kuratierten/synthetischen Reasoning- & Code-Daten. |
| **Phase 3: WSD-Cooldown** | **90 % – 100 %** | $R \in [1, 4]$, TBPTT $k=2$ | **Aus (100 % Tokens)** | Steiler LR-Decay mit vollen Gradienten + Checkpoint Averaging / EMA. |

---

## 3. Checkliste fürs Review: Wo können wir noch mehr sparen oder kürzen?
Diese Punkte sollten wir über kleine Einzel-Experimente (Mini-Ablationen) klären, bevor wir final bauen:
1. **TBPTT $k=1$ vs. $k=2$:** Spart $k=1$ genug VRAM/Compute auf der 4060 Ti, ohne dass die Loop-Qualität gegenüber $k=2$ sichtbar einbricht?
2. **MoR-Dynamic-Routing vs. Starres $R=2$:** Lohnt sich der Gather/Scatter-Overhead von MoR auf einer einzelnen 4060 Ti schon bei kurzen Sequenzen, oder ist ein starres $R=2$ mit `torch.compile` schneller?
3. **Loop-LoRA weglassen:** Reichen **Per-Loop RMSNorms + Step-Embeddings** völlig aus, um den Extra-VRAM und die feste Max-Loop-Bindung von Loop-LoRAs zu sparen?
4. **CLA-KV-Reuse vs. Selektives MoR-KV:** Reicht ein gemeinsamer KV-Cache aus Loop 1 (extrem schnell auf 288 GB/s) oder braucht Loop 3–4 eigene frische Keys/Values für aktive Tokens?
5. **UltraMemV2 im Loop vs. außerhalb:** Bestätigen, dass 1× Lookup im Prelude/Coda (statt in jedem Loop) praktisch ohne Qualitätsverlust bleibt.
