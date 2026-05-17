"""
Social Media Virality Project — K-Means Clustering (Day 5)
Discover content/behavior segments and compare viral rates per cluster.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "processed" / "posts_cleaned.csv"
OUT_DIR = PROJECT_DIR / "processed" / "kmeans_results"

# Behavioral / content features (document: interpretable clusters)
CLUSTER_FEATURES = [
    "followers",
    "likes",
    "comments",
    "caption_length",
    "hashtag_count",
    "post_hour",
    "post_day_of_week",
    "engagement_rate",
    "has_video_views",
    "is_verified",
]
K_RANGE = range(2, 11)
RANDOM_STATE = 42
PCA_SAMPLE = 5000  # subsample for scatter (faster plot)


def load_and_prepare() -> tuple[pd.DataFrame, np.ndarray, StandardScaler, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    work = df.copy()
    work["is_video"] = (work["media_type"] == "video").astype(int)

    feature_cols = CLUSTER_FEATURES + ["is_video"]
    X_raw = work[feature_cols].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return work, X_scaled, scaler, X_raw


def elbow_analysis(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels, sample_size=min(8000, len(X)), random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": inertia, "silhouette": sil})
    return pd.DataFrame(rows)


def choose_k(elbow_df: pd.DataFrame) -> int:
    """Pick K with best silhouette (tie-break: elbow bend heuristic)."""
    best_sil_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
    # Elbow: max second derivative drop in inertia
    inertia = elbow_df["inertia"].values
    drops = np.diff(inertia)
    drops2 = np.diff(drops)
    if len(drops2) > 0:
        elbow_k = int(elbow_df.iloc[np.argmin(drops2) + 2]["k"])
    else:
        elbow_k = best_sil_k
    # Prefer silhouette if close; document expects elbow discussion
    return best_sil_k if best_sil_k else elbow_k


def run_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    tagged = df.copy()
    tagged["cluster"] = labels
    profiles = (
        tagged.groupby("cluster")
        .agg(
            posts=("viral", "count"),
            viral_rate=("viral", "mean"),
            avg_engagement=("engagement_rate", "mean"),
            avg_likes=("likes", "mean"),
            avg_comments=("comments", "mean"),
            avg_hashtags=("hashtag_count", "mean"),
            avg_caption_len=("caption_length", "mean"),
            avg_followers=("followers", "mean"),
            pct_video=("is_video", "mean"),
            pct_verified=("is_verified", "mean"),
            avg_hour=("post_hour", "mean"),
        )
        .reset_index()
    )
    # Human-readable cluster names by dominant traits
    names = []
    for _, row in profiles.iterrows():
        parts = []
        if row["viral_rate"] >= profiles["viral_rate"].quantile(0.75):
            parts.append("High-viral")
        elif row["viral_rate"] <= profiles["viral_rate"].quantile(0.25):
            parts.append("Low-viral")
        if row["pct_video"] >= 0.4:
            parts.append("video-heavy")
        elif row["pct_video"] <= 0.15:
            parts.append("image-heavy")
        if row["avg_followers"] >= profiles["avg_followers"].median():
            parts.append("large-account")
        else:
            parts.append("smaller-account")
        names.append(" / ".join(parts) if parts else f"Cluster {int(row['cluster'])}")
    profiles["cluster_label"] = names
    return profiles


def plot_elbow(elbow_df: pd.DataFrame, chosen_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="#4C72B0", linewidth=2)
    axes[0].axvline(chosen_k, color="#C44E52", linestyle="--", label=f"Selected K={chosen_k}")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
    axes[0].set_title("Elbow method")
    axes[0].legend()

    axes[1].plot(elbow_df["k"], elbow_df["silhouette"], marker="s", color="#55A868", linewidth=2)
    axes[1].axvline(chosen_k, color="#C44E52", linestyle="--", label=f"Selected K={chosen_k}")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Cluster separation (higher = better)")
    axes[1].legend()

    fig.suptitle("Fig 1 — Choosing optimal K", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_elbow_plot.png")
    plt.close(fig)


def plot_pca_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    viral: np.ndarray,
    k: int,
) -> None:
    n = min(PCA_SAMPLE, len(X))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=n, replace=False)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X[idx])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    scatter0 = axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels[idx],
        cmap="tab10",
        alpha=0.45,
        s=12,
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].set_title(f"K-Means clusters (K={k})")
    plt.colorbar(scatter0, ax=axes[0], label="Cluster")

    viral_colors = np.where(viral[idx] == 1, "#C44E52", "#4C72B0")
    axes[1].scatter(coords[:, 0], coords[:, 1], c=viral_colors, alpha=0.4, s=12)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[1].set_title("Same space — viral vs not viral")
    from matplotlib.lines import Line2D

    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=8, label="Viral"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="Not viral"),
        ],
        loc="upper right",
    )

    fig.suptitle("Fig 2 — Cluster visualization (PCA 2D)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_cluster_pca.png")
    plt.close(fig)


def plot_viral_rate_by_cluster(profiles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = profiles.sort_values("viral_rate", ascending=True)
    colors = plt.cm.RdYlGn(order["viral_rate"] / max(order["viral_rate"].max(), 0.01))
    ax.barh(
        [f"C{int(c)}: {lbl[:28]}" for c, lbl in zip(order["cluster"], order["cluster_label"])],
        order["viral_rate"],
        color=colors,
    )
    overall = profiles["posts"].sum()
    baseline = (profiles["viral_rate"] * profiles["posts"]).sum() / overall
    ax.axvline(baseline, color="#333", linestyle="--", linewidth=1.5, label=f"Overall viral rate ({baseline:.1%})")
    ax.set_xlabel("Viral rate")
    ax.set_title("Fig 3 — Viral rate within each cluster")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_viral_rate_by_cluster.png")
    plt.close(fig)


def plot_centroid_heatmap(
    profiles: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    cols = [
        "avg_engagement",
        "avg_likes",
        "avg_comments",
        "avg_hashtags",
        "avg_caption_len",
        "pct_video",
        "avg_hour",
    ]
    mat = profiles.set_index("cluster")[cols].copy()
    # Normalize columns 0-1 for comparison across clusters
    mat_norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, max(4, len(profiles) * 0.5)))
    sns.heatmap(
        mat_norm.T,
        annot=mat.T.round(2),
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Relative intensity (per feature)"},
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature (centroid mean)")
    ax.set_title("Fig 4 — Cluster centroids (interpretation)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_centroid_heatmap.png")
    plt.close(fig)


def write_summary(
    elbow_df: pd.DataFrame,
    chosen_k: int,
    profiles: pd.DataFrame,
    overall_viral: float,
) -> None:
    best = profiles.loc[profiles["viral_rate"].idxmax()]
    worst = profiles.loc[profiles["viral_rate"].idxmin()]

    lines = [
        "# K-Means Clustering Summary — Social Media Virality Project",
        "",
        "## Setup",
        f"- **Features:** {', '.join(CLUSTER_FEATURES + ['is_video'])}",
        f"- **K tested:** {list(K_RANGE)}",
        f"- **Selected K:** {chosen_k} (best silhouette score)",
        f"- **Overall viral rate:** {overall_viral:.1%}",
        "",
        "## Elbow & silhouette",
        "",
        elbow_df.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Cluster profiles",
        "",
        profiles.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Key insights",
        "",
        f"- **Highest viral cluster:** C{int(best['cluster'])} ({best['cluster_label']}) — "
        f"**{best['viral_rate']:.1%}** viral rate ({int(best['posts']):,} posts)",
        f"- **Lowest viral cluster:** C{int(worst['cluster'])} ({worst['cluster_label']}) — "
        f"**{worst['viral_rate']:.1%}** viral rate ({int(worst['posts']):,} posts)",
        f"- **Spread:** {(best['viral_rate'] - worst['viral_rate']) * 100:.1f} percentage points between best and worst cluster.",
        "",
        "## Integrated insight (Day 5 — link to K-NN)",
        "",
        "- Certain content/behavior clusters naturally produce more viral posts.",
        "- Creators could shift toward high-viral clusters (e.g. similar engagement + format patterns).",
        "- Platforms promoting only high-engagement clusters may amplify bias and echo chambers.",
        "",
        "## Figures",
        "",
        "- `01_elbow_plot.png`",
        "- `02_cluster_pca.png`",
        "- `03_viral_rate_by_cluster.png`",
        "- `04_centroid_heatmap.png`",
    ]
    (OUT_DIR / "KMEANS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run data_cleaning.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Loading post-level data...")
    df, X, scaler, X_raw = load_and_prepare()
    feature_cols = CLUSTER_FEATURES + ["is_video"]
    overall_viral = df["viral"].mean()
    print(f"  Posts: {len(df):,} | Features: {X.shape[1]} | Viral rate: {overall_viral:.1%}")

    print("Elbow analysis...")
    elbow_df = elbow_analysis(X)
    chosen_k = choose_k(elbow_df)
    print(f"  Selected K = {chosen_k}")
    elbow_df.to_csv(OUT_DIR / "elbow_metrics.csv", index=False)

    print("Running K-Means...")
    model, labels = run_kmeans(X, chosen_k)
    df["cluster"] = labels

    profiles = cluster_profiles(df, labels, feature_cols)
    profiles.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)
    df[["account", "post_id", "viral", "cluster"]].to_csv(OUT_DIR / "posts_with_clusters.csv", index=False)

    plot_elbow(elbow_df, chosen_k)
    plot_pca_clusters(X, labels, df["viral"].values, chosen_k)
    plot_viral_rate_by_cluster(profiles)
    plot_centroid_heatmap(profiles, feature_cols)
    write_summary(elbow_df, chosen_k, profiles, overall_viral)

    print("\nCluster viral rates:")
    for _, row in profiles.sort_values("cluster").iterrows():
        print(
            f"  C{int(row['cluster'])}: {row['viral_rate']:.1%} viral "
            f"({int(row['posts']):,} posts) — {row['cluster_label']}"
        )
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
