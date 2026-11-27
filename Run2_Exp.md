# Run 2 Exp: Experimentelle Erweiterungen (CTM-Light & LFM3-Conv-YOCO-Loop)

## Rahmenbedingungen & Abgrenzung
* **Basis:** Baut vollständig auf [Run2.md](file:///home/henry/Documents/Henry/minotaur/Run2.md) auf (3B Looped CoLA + UltraMemV2 auf RTX 4060 Ti 16 GB `sm_89`, skalierbar auf H100 `sm_90`).
* **Zweck:** Isolierter Experimental-Zweig für Konzepte aus Sakana AI's **Continuous Thought Machine (CTM)** (<https://arxiv.org/abs/2505.05522>) sowie für die experimentelle Verknüpfung von **LFM3 Gated Short Conv ($k=7$) + SWA + YOCO** innerhalb eines geloopten Cores. Wird **nur** dann in den Haupt-Run übernommen, wenn schnelle Mini-Experimente einen klaren Mehrwert bei geringem Overhead zeigen.

### Was aus dem originalen CTM bewusst gestrichen wird
Das Original-CTM ist für kleine bis mittlere Reasoning-Netze gedacht und skaliert naiv nicht auf einen 3B-Transformer auf 16 GB VRAM:
1. **Keine Per-Neuron-Mini-LSTMs (NLMs):** Ein eigenes kleines MLP/LSTM pro Neuron über die Zeit-Achse sprengt bei $D \ge 2048$ sowohl VRAM als auch Memory-Bandwidth auf der 4060 Ti.
2. **Keine volle $O(D^2)$-Synchronisationsmatrix:** Alle paarweisen Neuron-Korrelationen über alle Channels ($D \times D$) zu tracken ist auf 16 GB untragbar.

Stattdessen testen wir **3 leichtgewichtige CTM-Light-Module**, die exakt die Stärken von CTM (Neural Synchronization als Latent-State + Certainty-Kalibrierung) auf unseren Looped Transformer übertragen.

---

## 1. Die experimentellen Module (CTM-Light & LFM3-YOCO-Loop)

### Modul E1: Subsampled / Low-Rank Neural Synchronization
* **Rank:** **Experimentell (B)**
* **Kurz-Zusammenfassung:** Anstatt am Ende der Loops nur den letzten Hidden-State $h_T$ zu nutzen, verfolgt E1 die zeitliche Synchronisation (Skalarprodukt-Akkumulation über die Loop-Ticks $t = 1 \dots R$) für eine **kleine, gelernte/gesampelte Auswahl von $P = 256\text{--}512$ Neuron-Paaren** ($O(P)$ statt $O(D^2)$). Dieser kompakte Synchronisations-Vektor wird per kleiner Projektion in den Readout/Coda- oder Router-Input gemischt.
* **Spart:** **Inferenz- & Reasoning-Steps** (bessere Generalisierung auf algorithmische/mehrstufige Probleme bei gleicher Loop-Tiefe).
* **Zahlt:** Leichter Zusatz-VRAM für die $P$-Pair-Akkumulatoren (wenige MB) + kleiner Tensor-Gather/Dot-Product-Compute pro Loop-Tick.
* **Qualitätsverlust / Risiko:** Unklar auf großen Sprachmodellen (in CTM auf Vision/Mazes/RL stark, bei Open-Ended Text wenig erprobt).
* **Quelle:**
  * Continuous Thought Machines (CTM): <https://arxiv.org/abs/2505.05522>

### Modul E2: Certainty-Based Adaptive Halting Loss (CTM-Loss)
* **Rank:** **Experimentell (B)**
* **Kurz-Zusammenfassung:** Ersetzt den klassischen PonderNet-/Prior-Loss des MoR-Routers durch den **CTM-Certainty-Loss**: Für jedes Token wird über die Ticks $t = 1 \dots R$ die Ausgangs-Certainty ($1 - \text{normierte Entropie}$) getrackt. Der Loss optimiert gezielt **zwei Punkte**: (1) den Tick $t_{\min}$ mit dem niedrigsten Cross-Entropy-Loss und (2) den Tick $t_{\text{cert}}$ mit der höchsten Certainty. Dadurch lernt das Modell kalibriert, wann ein Token „fertig gedacht“ ist.
* **Spart:** **Test-Time- & Trainings-Compute** (saubereres Early-Stopping für einfache Tokens im MoR-Router).
* **Zahlt:** Benötigt Zwischen-Logits (bzw. einen leichten Auxiliary-Head + Cut Cross-Entropy) an den Loop-Exits während der Experimente $\to$ moderater Extra-Compute.
* **Qualitätsverlust / Risiko:** Wenn der Auxiliary-Head zu stark drückt, kann das Modell voreilig „sicher“ werden und zu früh aussteigen.
* **Quelle:**
  * Continuous Thought Machines (CTM): <https://arxiv.org/abs/2505.05522>
  * Mixture-of-Recursions (MoR): <https://arxiv.org/abs/2507.10524>

### Modul E3 (Optional): Lightweight Tick-History Buffer (1D-Conv / EMA statt Neuron-LSTM)
* **Rank:** **Experimentell (C)**
* **Kurz-Zusammenfassung:** Billiger Ersatz für die CTM-Neuron-History: Hält pro Channel einen winzigen Ringbuffer bzw. ein **EMA über die letzten $M = 4$ Loop-Ticks** und wendet eine leichte 1D-Depthwise-Conv (oder 2-Gewichts-Mischung) über die Tick-Achse an, bevor der nächste Loop startet.
* **Spart:** Vermeidet komplettes Vergessen früherer Tick-Zustände bei tiefen Loops ($R \ge 4$), ohne die Kosten von Per-Neuron-NLMs.
* **Zahlt:** Speichert $M$ History-Tensoren (oder 1 EMA-Tensor) im Core; +5–10 % Bandwidth-Last auf der 4060 Ti.
* **Qualitätsverlust / Risiko:** Auf 288 GB/s GDDR6 möglicherweise langsamer als der Qualitätsgewinn — muss im Mini-Benchmark gegen einfache Input-Injection (Huginn) bestehen.
* **Quelle:**
  * Continuous Thought Machines (CTM): <https://arxiv.org/abs/2505.05522>

### Modul E4: LFM3 Gated Short Conv ($k=7$) + SWA + YOCO Cross-Attention im Looped Core
* **Rank:** **Experimentell (S-Potenzial)**
* **Kurz-Zusammenfassung:** Verknüpft das LFM3-Tri-Hybrid-Design mit **YOCO** direkt im Looped-Sandwich-Layout:
  * **Prelude (1× pro Forward, außerhalb des Loops):** Gated Short Conv ($k=7$) + Sliding-Window Attention (SWA, z. B. $W=256$) + 1× Global GQA/MLA erzeugt **einmalig** den globalen YOCO-KV-Cache (+ 1× UltraMemV2-Fakten-Lookup).
  * **Looped Core ($R$ Ticks):** Double-Gated Short Conv ($k=7$) (effektives lokales Rezeptivfeld wächst über $R=4$ Loops auf 25 Tokens; bei Inferenz/Decode nur 7-Token-Ringbuffer) + **YOCO Cross-Attention** auf dem festen Prelude-KV-Cache (kein K/V-Neuberechnen pro Loop!) + CoLA-MLP + **Head-Wise Gating** für FP8/INT8-Stabilität.
  * **Wichtiges MoR-Detail:** Die super-billige `conv1d(k=7)` läuft auf der vollen Sequenz (inaktive MoR-Tokens behalten ihren letzten Zustand, damit keine Nachbarschaftslücken entstehen); **nur** die teure YOCO-Cross-Attention und das CoLA-MLP werden auf die aktiven 30–40 % MoR-Tokens gesliced.
* **Spart:** **Loop-KV-Compute (100 %!)**, **Loop-KV-VRAM**, **HBM-Bandwidth auf der 4060 Ti**.
* **Zahlt:** Der globale KV-Cache bleibt über die Loop-Ticks hinweg statisch (nur $Q$ und das lokale Conv-Fenster aktualisieren sich pro Loop).
* **Qualitätsverlust / Risiko:** **Minimal** (globaler Kontext bleibt voll lesbar, lokales Reasoning läuft über Conv + $Q$-Refinement).
* **Quellen:**
  * YOCO (*You Only Cache Once*): <https://arxiv.org/abs/2405.05254>
  * Liquid AI LFM2 / LFM3 Hybrid Architecture (Gated Short Convolutions + SWA + Head-Wise Gating).

---

## 2. Geplanter Mini-Versuchsaufbau (< 1 Stunde auf RTX 4060 Ti)

Um vor dem großen 3B-Run sauber zu entscheiden, testen wir auf einem **50M–150M Proxy-Modell** (CoLA-Backbone, identische Loop-Struktur):

1. **Varianten (Ablations-Matrix):**
   * **V0 (Baseline):** `Run2` (Huginn Sandwich + Input-Injection + Step-Norms + MoR, $R \in [1, 4]$, $k=2$ TBPTT)
   * **V1:** `V0 + E1` (Subsampled Sync $P=256$)
   * **V2:** `V0 + E1 + E2` (Sync + Certainty-Halting-Loss)
   * **V3:** `V0 + E1 + E2 + E3` (mit Tick-History-EMA)
   * **V4:** `V0 + E4` (LFM3 Conv $k=7$ + SWA im Prelude + YOCO-Loop-Core + Head-Wise Gating)
   * **V5:** `V2 + E4` (Kombination bei positivem E1/E2-Ergebnis)

2. **Test-Tasks (wo CTM glänzen müsste):**
   * Kurzer Sprach-/Code-Pretraining-Slice (Val-Loss / Perplexity)
   * Synthetische Reasoning-Probes (Multi-Step Arithmetic, Parity / State-Tracking, verschachtelte Klammern / Code-Indent)

3. **Harte Go / No-Go-Kriterien für die Übernahme in Run 2:**
   * **Durchsatz-Limit:** Tokens/s dürfen gegenüber V0 **maximal 10 %** langsamer sein.
   * **VRAM-Limit:** Peak-VRAM-Zuwachs **< 500 MB** (hochgerechnet auf 3B).
   * **Qualitäts-Hürde:** E1/E2/E3 müssen entweder den **Validation-Loss auf Reasoning-Probes signifikant schlagen** oder beim **MoR Early-Exit mehr FLOPs einsparen** als die Baseline. Alles, was das nicht schafft, fliegt vor dem großen Run raus.
