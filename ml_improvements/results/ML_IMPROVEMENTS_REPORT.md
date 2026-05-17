# ML Improvements Report

## Problem
- Original pipeline: ~90% not viral / ~10% viral (9:1 imbalance).
- Default K-NN uses threshold 0.5 and no resampling → strong majority-class bias.

## Approaches tested

### 1. Virality label threshold (engagement-rate percentile)

| virality_percentile | n_not_viral | n_viral | viral_pct | imbalance_ratio |
| --- | --- | --- | --- | --- |
| 0.8000 | 19793.0000 | 4949.0000 | 0.2000 | 3.9994 |
| 0.8500 | 21030.0000 | 3712.0000 | 0.1500 | 5.6654 |
| 0.9000 | 22264.0000 | 2478.0000 | 0.1002 | 8.9847 |
| 0.9500 | 23501.0000 | 1241.0000 | 0.0502 | 18.9371 |

### 2. Training-time imbalance strategies
- **baseline** — no change
- **oversample** — duplicate minority posts to balance
- **undersample** — reduce majority (50% minority:majority ratio)
- **smote** — synthetic minority examples

### 3. Decision threshold tuning
- After `predict_proba`, sweep 0.05–0.95 on **validation** set; maximize F1.

## Best configuration

```json
{
  "virality_percentile": 0.8,
  "virality_threshold": 0.1483337502705407,
  "imbalance_strategy": "undersample",
  "balance_n_not_viral": 19793,
  "balance_n_viral": 4949,
  "balance_viral_pct": 0.20002425026271117,
  "balance_imbalance_ratio": 3.999393816932714,
  "train_size_after_resample": 8907,
  "train_viral_pct_after_resample": 0.3333333333333333,
  "decision_threshold_default": 0.5,
  "decision_threshold_tuned": 0.45,
  "val_f1_default": 0.6723565670934092,
  "val_f1_tuned": 0.6756882874475035,
  "test_accuracy_default": 0.8571428571428571,
  "test_balanced_accuracy_default": 0.8107181948303444,
  "test_precision_default": 0.621043627031651,
  "test_recall_default": 0.7333333333333333,
  "test_f1_default": 0.6725335803612784,
  "test_roc_auc_default": 0.8789616294289193,
  "test_accuracy_tuned": 0.8545160638512831,
  "test_balanced_accuracy_tuned": 0.8128639259480381,
  "test_precision_tuned": 0.6123128119800333,
  "test_recall_tuned": 0.7434343434343434,
  "test_f1_tuned": 0.6715328467153284,
  "test_roc_auc_tuned": 0.8789616294289193
}
```

## Baseline vs best (test set, tuned threshold)

| Metric | Original-style (p90, baseline, t=0.5) | Best |
|--------|--------------------------------------|------|
| F1 | 0.5504 | 0.6715 |
| Recall (viral) | 0.4738 | 0.7434 |
| Precision | 0.6564 | 0.6123 |
| Balanced accuracy | 0.7231 | 0.8129 |

## Recommendation

- Use virality percentile **0.8** if you want `20.0%` viral class.
- Apply **`undersample`** on training data only.
- Use decision threshold **0.45** (not 0.5) when calling `predict_proba`.
