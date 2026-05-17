"""
Social Media Virality Project — K-NN Classification (Day 4)
Predict viral (1) vs not viral (0) using scaled train/test splits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "processed" / "train_scaled.csv"
TEST_PATH = PROJECT_DIR / "processed" / "test_scaled.csv"
OUT_DIR = PROJECT_DIR / "processed" / "knn_results"

K_VALUES = [1, 3, 5, 7, 11]
CV_FOLDS = 5
RANDOM_STATE = 42
TARGET_COL = "viral"


def load_xy(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def evaluate_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int,
) -> dict:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
    )

    return {
        "k": k,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "model": model,
        "y_pred_test": y_pred_test,
    }


def plot_accuracy_vs_k(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results_df["k"], results_df["train_accuracy"], marker="o", label="Train accuracy")
    ax.plot(results_df["k"], results_df["test_accuracy"], marker="s", label="Test accuracy")
    ax.plot(
        results_df["k"],
        results_df["cv_accuracy_mean"],
        marker="^",
        linestyle="--",
        label=f"{CV_FOLDS}-fold CV (train)",
    )
    ax.fill_between(
        results_df["k"],
        results_df["cv_accuracy_mean"] - results_df["cv_accuracy_std"],
        results_df["cv_accuracy_mean"] + results_df["cv_accuracy_std"],
        alpha=0.15,
    )
    best_k = int(results_df.loc[results_df["test_accuracy"].idxmax(), "k"])
    best_acc = results_df["test_accuracy"].max()
    ax.axvline(best_k, color="#C44E52", linestyle=":", linewidth=1.5, label=f"Best K={best_k}")
    ax.scatter([best_k], [best_acc], color="#C44E52", s=80, zorder=5)

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("K (number of neighbors)")
    ax.set_ylabel("Accuracy")
    ax.set_title("K-NN: Accuracy vs K")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.savefig(OUT_DIR / "01_accuracy_vs_k.png")
    plt.close(fig)


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, k: int) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
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
    ax.set_title(f"Confusion matrix (K={k})")
    fig.savefig(OUT_DIR / "02_confusion_matrix.png")
    plt.close(fig)


def plot_metrics_bar(results_df: pd.DataFrame, best_k: int) -> None:
    row = results_df[results_df["k"] == best_k].iloc[0]
    metrics = ["precision", "recall", "f1", "test_accuracy"]
    labels = ["Precision", "Recall", "F1", "Accuracy"]
    values = [row[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B3"]
    ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Classification metrics at best K={best_k}")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.savefig(OUT_DIR / "03_metrics_best_k.png")
    plt.close(fig)


def write_summary(
    results_df: pd.DataFrame,
    best_k: int,
    y_test: pd.Series,
    y_pred: np.ndarray,
    report: str,
) -> None:
    best = results_df[results_df["k"] == best_k].iloc[0]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    lines = [
        "# K-NN Model Summary — Social Media Virality Project",
        "",
        "## Setup",
        f"- **K values tested:** {K_VALUES}",
        f"- **Features:** {results_df.attrs.get('n_features', 'N/A')}",
        f"- **Train / test:** stratified 80/20 split (scaled features)",
        f"- **Best K (by test accuracy):** {best_k}",
        "",
        "## Model comparison",
        "",
        results_df[
            ["k", "train_accuracy", "test_accuracy", "cv_accuracy_mean", "precision", "recall", "f1"]
        ]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best model metrics (test set)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {best['test_accuracy']:.4f} |",
        f"| Precision | {best['precision']:.4f} |",
        f"| Recall | {best['recall']:.4f} |",
        f"| F1 Score | {best['f1']:.4f} |",
        "",
        "## Confusion matrix (counts)",
        "",
        f"| | Pred Not viral | Pred Viral |",
        f"|--|----------------|------------|",
        f"| **Actual Not viral** | {tn} | {fp} |",
        f"| **Actual Viral** | {fn} | {tp} |",
        "",
        "## Classification report",
        "",
        "```",
        report.strip(),
        "```",
        "",
        "## Discussion (project prompts)",
        "",
        "- **Is predicting viral harder than non-viral?** "
        f"Viral class recall = {best['recall']:.1%} — the model often misses viral posts "
        f"({fn} false negatives vs {fp} false positives).",
        "- **What if K is too small?** K=1 usually has highest train accuracy but can overfit; "
        "compare train vs test gap in `01_accuracy_vs_k.png`.",
        "- **Class imbalance:** ~10% viral posts; high accuracy can still mean poor viral detection.",
        "",
        "## Figures",
        "",
        "- `01_accuracy_vs_k.png`",
        "- `02_confusion_matrix.png`",
        "- `03_metrics_best_k.png`",
    ]
    (OUT_DIR / "KNN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Run data_cleaning.py first to create train/test CSV files.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Loading scaled train/test data...")
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {X_train.shape[1]}")

    print(f"Evaluating K = {K_VALUES} ...")
    rows = []
    best_result = None

    for k in K_VALUES:
        print(f"  K={k}...", end=" ", flush=True)
        result = evaluate_k(X_train, y_train, X_test, y_test, k)
        rows.append({key: result[key] for key in result if key not in ("model", "y_pred_test")})
        print(
            f"test acc={result['test_accuracy']:.4f} | "
            f"F1={result['f1']:.4f} | recall={result['recall']:.4f}"
        )

    results_df = pd.DataFrame(rows)
    results_df.attrs["n_features"] = X_train.shape[1]
    results_df.to_csv(OUT_DIR / "knn_comparison_table.csv", index=False)

    best_idx = results_df["test_accuracy"].idxmax()
    best_k = int(results_df.loc[best_idx, "k"])

    # Re-fit best model for final plots
    best_result = evaluate_k(X_train, y_train, X_test, y_test, best_k)
    report = classification_report(
        y_test,
        best_result["y_pred_test"],
        target_names=["Not viral", "Viral"],
        zero_division=0,
    )

    plot_accuracy_vs_k(results_df)
    plot_confusion_matrix(y_test, best_result["y_pred_test"], best_k)
    plot_metrics_bar(results_df, best_k)
    write_summary(results_df, best_k, y_test, best_result["y_pred_test"], report)

    print(f"\nBest K = {best_k}")
    print(f"  Test accuracy: {results_df.loc[best_idx, 'test_accuracy']:.4f}")
    print(f"  Precision:     {results_df.loc[best_idx, 'precision']:.4f}")
    print(f"  Recall:        {results_df.loc[best_idx, 'recall']:.4f}")
    print(f"  F1:            {results_df.loc[best_idx, 'f1']:.4f}")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
