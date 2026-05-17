# K-NN Model Summary — Social Media Virality Project

## Setup
- **K values tested:** [1, 3, 5, 7, 11]
- **Features:** 378
- **Train / test:** stratified 80/20 split (scaled features)
- **Best K (by test accuracy):** 3

## Model comparison

|       k |   train_accuracy |   test_accuracy |   cv_accuracy_mean |   precision |   recall |     f1 |
|--------:|-----------------:|----------------:|-------------------:|------------:|---------:|-------:|
|  1.0000 |           1.0000 |          0.9297 |             0.9277 |      0.6451 |   0.6633 | 0.6541 |
|  3.0000 |           0.9629 |          0.9351 |             0.9307 |      0.6915 |   0.6371 | 0.6632 |
|  5.0000 |           0.9534 |          0.9305 |             0.9299 |      0.6801 |   0.5786 | 0.6253 |
|  7.0000 |           0.9473 |          0.9287 |             0.9266 |      0.6907 |   0.5222 | 0.5947 |
| 11.0000 |           0.9375 |          0.9264 |             0.9168 |      0.7481 |   0.4012 | 0.5223 |

## Best model metrics (test set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9351 |
| Precision | 0.6915 |
| Recall | 0.6371 |
| F1 Score | 0.6632 |

## Confusion matrix (counts)

| | Pred Not viral | Pred Viral |
|--|----------------|------------|
| **Actual Not viral** | 4312 | 141 |
| **Actual Viral** | 180 | 316 |

## Classification report

```
precision    recall  f1-score   support

   Not viral       0.96      0.97      0.96      4453
       Viral       0.69      0.64      0.66       496

    accuracy                           0.94      4949
   macro avg       0.83      0.80      0.81      4949
weighted avg       0.93      0.94      0.93      4949
```

## Discussion (project prompts)

- **Is predicting viral harder than non-viral?** Viral class recall = 63.7% — the model often misses viral posts (180 false negatives vs 141 false positives).
- **What if K is too small?** K=1 usually has highest train accuracy but can overfit; compare train vs test gap in `01_accuracy_vs_k.png`.
- **Class imbalance:** ~10% viral posts; high accuracy can still mean poor viral detection.

## Figures

- `01_accuracy_vs_k.png`
- `02_confusion_matrix.png`
- `03_metrics_best_k.png`