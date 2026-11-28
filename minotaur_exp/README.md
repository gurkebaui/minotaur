# `minotaur_exp` — Modulares Experiment-Harness

Dieses Verzeichnis enthält die vollständige, flag-gesteuerte Implementierung der 7 gebündelten Mini-Experimente aus [Experiments.md](../Experiments.md) inklusive übergeordneter YAML-Konfiguration und Master-Runner mit Datum/Uhrzeit-Logging.

## 1. Übergeordnete Config (`suite_config.yaml`)
In [suite_config.yaml](suite_config.yaml) steuerst du zentral:
- `selected_experiments`: z.B. `["all"]`, `["exp0", "exp1"]` oder einzelne IDs `["0b", "1a", "5b"]`
- `model_scale`: `"proxy_150m"` (Standard für echte 4060-Ti-Experimente), `"tiny_smoke"` (Sekunden-Test) oder `"target_3b"`
- `time_scale`: Skaliert alle `max_minutes` global (z.B. `0.5` für halbe Laufzeit)
- `data.dataset_preset`: `"fineweb_edu"` (Stream + Cache, automatischer Fallback auf `"tinystories"` falls offline), `"tinystories"` oder `"synthetic"`
- `experiment_overrides`: Beliebige Hyperparameter pro Experiment-ID überschreiben

## 2. Experimente starten & auswählen

```bash
# Alle verfügbaren Experimente anzeigen
python minotaur_exp/run_experiments.py --list

# Schneller End-to-End Smoke-Test über alle 15 Sub-Runs (~1-2 Min)
python minotaur_exp/run_experiments.py --smoke --exp all --tag smoke

# Nur bestimmte Gruppen oder Sub-Runs starten
python minotaur_exp/run_experiments.py --exp 0 1a 5

# Interaktive Auswahl im Terminal
python minotaur_exp/run_experiments.py --interactive
```

## 3. Logs mit Datum & Uhrzeit (überschreibsicher)
Jeder Start legt automatisch einen eigenen Zeitstempel-Ordner unter `minotaur_exp/logs/YYYY-MM-DD_HH-MM-SS[_tag]/` an:
- `run_<timestamp>.log`: Vollständiges Terminal-Log
- `suite_config_snapshot_<timestamp>.json`: Exakter Config- & Hardware-Snapshot
- `summary_<timestamp>.csv` & `summary_<timestamp>.md`: Live nach jedem Experiment aktualisierte Vergleichstabelle (BPB, Bytes/s, VRAM-Breakdown, Reasoning-Probes, Go/No-Go)
- `<exp_id>/metrics_<timestamp>.jsonl` & `<exp_id>/checkpoint.pt`: Step-Metriken & Checkpoints
