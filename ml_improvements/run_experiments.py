"""
ML engineer experiments: virality threshold tuning, imbalance handling,
and classification threshold optimization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Reuse feature engineering from main project
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from data_cleaning import build_feature_matrix  # noqa: E402

ML_DIR = Path(__file__).resolve().parent
POSTS_PATH = PROJECT_DIR / "processed" / "posts_cleaned.csv"
OUT_DIR = ML_DIR / "results"
FIG_DIR = OUT_DIR / "figures"

RANDOM_STATE = 42
K_NEIGHBORS = 3
VIRALITY_PERCENTILES = [0.80, 0.85, 0.90, 0.95]
STRATEGIES = ("baseline", "oversample", "undersample", "smote")
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


def relabel_viral(posts: pd.DataFrame, percentile: float) -> pd.DataFrame:
    out = posts.copy()
    threshold = out["engagement_rate"].quantile(percentile)
    out["viral"] = (out["engagement_rate"] >= threshold).astype(int)
    out["virality_percentile"] = percentile
    out["virality_threshold"] = threshold
    return out


def class_balance_stats(y: pd.Series) -> dict:
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    total = len(y)
    return {
        "n_not_viral": n0,
        "n_viral": n1,
        "viral_pct": n1 / total if total else 0,
        "imbalance_ratio": n0 / n1 if n1 else float("inf"),
    }


def split_and_scale(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, y_train.values, y_val.values, y_test.values, scaler


def apply_resampling(
    X: np.ndarray, y: np.ndarray, strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "baseline":
        return X, y
    if strategy == "oversample":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    elif strategy == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    else:
        raise ValueError(strategy)
    return sampler.fit_resample(X, y)


def tune_decision_threshold(y_true: np.ndarray, proba_viral: np.ndarray) -> tuple[float, float]:
    """Pick threshold on validation set that maximizes F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in THRESHOLD_GRID:
        pred = (proba_viral >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_at_threshold(
    y_true: np.ndarray, proba_viral: np.ndarray, threshold: float
) -> dict:
    pred = (proba_viral >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba_viral) if len(np.unique(y_true)) > 1 else np.nan,
    }


def run_single_config(
    posts: pd.DataFrame,
    percentile: float,
    strategy: str,
) -> dict | None:
    labeled = relabel_viral(posts, percentile)
    y = labeled["viral"]
    balance = class_balance_stats(y)

    if balance["n_viral"] < 30:
        return None

    X, _, _ = build_feature_matrix(labeled)
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    except ValueError:
        return None

    try:
        X_train_r, y_train_r = apply_resampling(X_train, y_train, strategy)
    except ValueError:
        return None

    model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights="distance", n_jobs=-1)
    model.fit(X_train_r, y_train_r)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    default_val = evaluate_at_threshold(y_val, val_proba, 0.5)
    default_test = evaluate_at_threshold(y_test, test_proba, 0.5)

    opt_threshold, val_f1_at_opt = tune_decision_threshold(y_val, val_proba)
    tuned_test = evaluate_at_threshold(y_test, test_proba, opt_threshold)

    train_balance_after = class_balance_stats(pd.Series(y_train_r))

    return {
        "virality_percentile": percentile,
        "virality_threshold": float(labeled["virality_threshold"].iloc[0]),
        "imbalance_strategy": strategy,
        **{f"balance_{k}": v for k, v in balance.items()},
        "train_size_after_resample": len(y_train_r),
        "train_viral_pct_after_resample": train_balance_after["viral_pct"],
        "decision_threshold_default": 0.5,
        "decision_threshold_tuned": opt_threshold,
        "val_f1_default": default_val["f1"],
        "val_f1_tuned": val_f1_at_opt,
        **{f"test_{k}_default": v for k, v in default_test.items()},
        **{f"test_{k}_tuned": v for k, v in tuned_test.items()},
    }


def plot_threshold_sweep(sweep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        sweep_df["virality_percentile"].astype(str),
        sweep_df["viral_pct"] * 100,
        color="#4C72B0",
    )
    ax.set_xlabel("Virality percentile (top X% = viral)")
    ax.set_ylabel("Viral class %")
    ax.set_title("Class balance vs virality label threshold")
    fig.savefig(FIG_DIR / "01_virality_percentile_balance.png")
    plt.close(fig)


def plot_strategy_comparison(results_df: pd.DataFrame) -> None:
    plot_df = results_df.copy()
    plot_df["label"] = plot_df.apply(
        lambda r: f"p{int(r['virality_percentile']*100)}-{r['imbalance_strategy']}", axis=1
    )
    plot_df = plot_df.sort_values("test_f1_tuned", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.35)))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.2, plot_df["test_f1_default"], height=0.35, label="F1 (threshold=0.5)")
    ax.barh(y_pos + 0.2, plot_df["test_f1_tuned"], height=0.35, label="F1 (tuned threshold)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("F1 score (test set)")
    ax.set_title("Imbalance strategy × virality percentile")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_strategy_f1_comparison.png")
    plt.close(fig)


def plot_best_roc(y_test: np.ndarray, proba: np.ndarray, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (viral recall)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(FIG_DIR / "03_best_model_roc.png")
    plt.close(fig)


def plot_threshold_curve(y_val: np.ndarray, proba: np.ndarray, chosen_t: float) -> None:
    thresholds = THRESHOLD_GRID
    f1s = [f1_score(y_val, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, f1s, marker="o", markersize=3)
    ax.axvline(chosen_t, color="#C44E52", linestyle="--", label=f"Best = {chosen_t:.2f}")
    ax.set_xlabel("Decision threshold (P(viral))")
    ax.set_ylabel("Validation F1")
    ax.set_title("Classification threshold tuning")
    ax.legend()
    fig.savefig(FIG_DIR / "04_threshold_tuning_curve.png")
    plt.close(fig)


def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    sweep_df: pd.DataFrame,
    results_df: pd.DataFrame,
    best: dict,
    baseline_row: pd.Series,
) -> None:
    lines = [
        "# ML Improvements Report",
        "",
        "## Problem",
        "- Original pipeline: ~90% not viral / ~10% viral (9:1 imbalance).",
        "- Default K-NN uses threshold 0.5 and no resampling → strong majority-class bias.",
        "",
        "## Approaches tested",
        "",
        "### 1. Virality label threshold (engagement-rate percentile)",
        "",
        _df_to_md_table(sweep_df),
        "",
        "### 2. Training-time imbalance strategies",
        "- **baseline** — no change",
        "- **oversample** — duplicate minority posts to balance",
        "- **undersample** — reduce majority (50% minority:majority ratio)",
        "- **smote** — synthetic minority examples",
        "",
        "### 3. Decision threshold tuning",
        "- After `predict_proba`, sweep 0.05–0.95 on **validation** set; maximize F1.",
        "",
        "## Best configuration",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
        "",
        "## Baseline vs best (test set, tuned threshold)",
        "",
        f"| Metric | Original-style (p90, baseline, t=0.5) | Best |",
        f"|--------|--------------------------------------|------|",
        f"| F1 | {baseline_row['test_f1_default']:.4f} | {best['test_f1_tuned']:.4f} |",
        f"| Recall (viral) | {baseline_row['test_recall_default']:.4f} | {best['test_recall_tuned']:.4f} |",
        f"| Precision | {baseline_row['test_precision_default']:.4f} | {best['test_precision_tuned']:.4f} |",
        f"| Balanced accuracy | {baseline_row['test_balanced_accuracy_default']:.4f} | {best['test_balanced_accuracy_tuned']:.4f} |",
        "",
        "## Recommendation",
        "",
        f"- Use virality percentile **{best['virality_percentile']}** if you want `{best['balance_viral_pct']*100:.1f}%` viral class.",
        f"- Apply **`{best['imbalance_strategy']}`** on training data only.",
        f"- Use decision threshold **{best['decision_threshold_tuned']:.2f}** (not 0.5) when calling `predict_proba`.",
        "",
    ]
    (OUT_DIR / "ML_IMPROVEMENTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def refit_best_and_plot(
    posts: pd.DataFrame, best: dict
) -> None:
    labeled = relabel_viral(posts, best["virality_percentile"])
    X, _, _ = build_feature_matrix(labeled)
    y = labeled["viral"]
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    X_train_r, y_train_r = apply_resampling(X_train, y_train, best["imbalance_strategy"])

    model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights="distance", n_jobs=-1)
    model.fit(X_train_r, y_train_r)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    t = best["decision_threshold_tuned"]

    plot_threshold_curve(y_val, val_proba, t)
    plot_best_roc(y_test, test_proba, f"ROC — best config (strategy={best['imbalance_strategy']})")

    cm = confusion_matrix(y_test, (test_proba >= t).astype(int))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not viral", "Viral"],
        yticklabels=["Not viral", "Viral"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (tuned t={t:.2f})")
    fig.savefig(FIG_DIR / "05_best_confusion_matrix.png")
    plt.close(fig)


def main() -> None:
    if not POSTS_PATH.exists():
        raise FileNotFoundError(f"Run ../data_cleaning.py first. Missing {POSTS_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Loading cleaned posts...")
    posts = pd.read_csv(POSTS_PATH)

    print("Phase 1: virality percentile sweep (class balance only)...")
    sweep_rows = []
    for p in VIRALITY_PERCENTILES:
        labeled = relabel_viral(posts, p)
        stats = class_balance_stats(labeled["viral"])
        sweep_rows.append({"virality_percentile": p, **stats})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_DIR / "virality_threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep_df)

    print("Phase 2: grid search (percentile × imbalance strategy)...")
    results = []
    for p in VIRALITY_PERCENTILES:
        for strategy in STRATEGIES:
            print(f"  p={p:.2f}, strategy={strategy}...", flush=True)
            row = run_single_config(posts, p, strategy)
            if row:
                results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "imbalance_strategy_comparison.csv", index=False)
    plot_strategy_comparison(results_df)

    best_idx = results_df["test_f1_tuned"].idxmax()
    best = results_df.loc[best_idx].to_dict()

    baseline_mask = (results_df["virality_percentile"] == 0.90) & (
        results_df["imbalance_strategy"] == "baseline"
    )
    baseline_row = results_df[baseline_mask].iloc[0]

    with open(OUT_DIR / "best_configuration.json", "w") as f:
        json.dump(best, f, indent=2)

    write_report(sweep_df, results_df, best, baseline_row)
    refit_best_and_plot(posts, best)

    print("\n=== Best configuration ===")
    print(f"  Virality percentile: {best['virality_percentile']}")
    print(f"  Strategy:            {best['imbalance_strategy']}")
    print(f"  Decision threshold:  {best['decision_threshold_tuned']:.2f}")
    print(f"  Test F1 (tuned):     {best['test_f1_tuned']:.4f}  (baseline p90: {baseline_row['test_f1_default']:.4f})")
    print(f"  Test recall (tuned): {best['test_recall_tuned']:.4f}  (baseline: {baseline_row['test_recall_default']:.4f})")
    print(f"\nResults: {OUT_DIR}/")


if __name__ == "__main__":
    main()
