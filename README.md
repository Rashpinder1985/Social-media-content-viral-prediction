# Social Media Viral Prediction

Predict whether an Instagram post will go **viral** using **K-NN**, and find content patterns with **K-Means**.

School project: machine learning on Instagram-style engagement data.

---

## What you need

- Python 3.10+
- `instagram_dataset.csv` in this folder (included in the repo)

---

## Quick start

```bash
# 1. Go to project folder
cd Social-media-content-viral-prediction

# 2. Create virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the pipeline (in order)
python data_cleaning.py
python eda.py
python knn_model.py
python kmeans_clustering.py
```

Each script prints where it saves outputs.

---

## Project steps

| Step | Script | Output folder |
|------|--------|----------------|
| 1. Clean data | `data_cleaning.py` | `processed/` |
| 2. Explore (EDA) | `eda.py` | `processed/eda_figures/` |
| 3. Classify (K-NN) | `knn_model.py` | `processed/knn_results/` |
| 4. Cluster (K-Means) | `kmeans_clustering.py` | `processed/kmeans_results/` |

---

## Main files

```
data_cleaning.py      # Flatten posts, build features, label viral/not viral
eda.py                # Charts and EDA summary
knn_model.py          # K-NN classifier (K = 1, 3, 5, 7, 11)
kmeans_clustering.py  # K-Means + elbow plot + viral rate per cluster
instagram_dataset.csv # Raw data
requirements.txt
Social Media Virality Project.docx   # Project brief
```

---

## How “viral” is defined

A post is **viral (1)** if its engagement rate is in the **top 10%**:

`(likes + comments) / followers`

You can change this in `data_cleaning.py` → `VIRALITY_PERCENTILE`.

---

## Results (our run)

- **K-NN best K:** 3 — test accuracy ~93.5%, F1 ~0.66  
- **K-Means best K:** 3 clusters — different viral rates per cluster  

See `processed/knn_results/KNN_SUMMARY.md` and `processed/kmeans_results/KMEANS_SUMMARY.md` for details.

---

## Note on large files

`train_scaled.csv` and `test_scaled.csv` are **not** on GitHub (too large). They are created when you run `data_cleaning.py`.

---

## License

MIT — see [LICENSE](LICENSE).
