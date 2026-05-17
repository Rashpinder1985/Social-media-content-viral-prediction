# ML improvements (class imbalance & threshold tuning)

Experiments live here so the **original** pipeline in the project root stays unchanged.

## What this folder does

1. **Virality label tuning** — try different engagement-rate percentiles (80%, 85%, 90%, 95%).
2. **Class imbalance handling** — oversample, undersample, SMOTE on the training set only.
3. **Classification threshold tuning** — pick the best probability cutoff (not always 0.5) using a validation set.
4. **Better metrics** — balanced accuracy, ROC-AUC, F1 (not accuracy alone).

## Setup

```bash
cd ml_improvements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need the main project cleaned once:

```bash
cd ..
python data_cleaning.py
```

## Run experiments

```bash
python run_experiments.py
```

## Outputs

| File | Description |
|------|-------------|
| `results/virality_threshold_sweep.csv` | Class balance for each label percentile |
| `results/imbalance_strategy_comparison.csv` | All strategy × percentile metrics |
| `results/best_configuration.json` | Winning settings |
| `results/ML_IMPROVEMENTS_REPORT.md` | Short engineer summary |
| `results/figures/` | Plots |
