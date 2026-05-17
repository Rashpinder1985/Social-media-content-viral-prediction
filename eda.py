"""
Social Media Virality Project — Exploratory Data Analysis (Day 3)
Generates 8–10 visualizations from processed/posts_cleaned.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "processed" / "posts_cleaned.csv"
FIG_DIR = PROJECT_DIR / "processed" / "eda_figures"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NUMERIC_FOR_CORR = [
    "followers",
    "likes",
    "comments",
    "caption_length",
    "hashtag_count",
    "engagement_rate",
    "post_hour",
    "has_video_views",
    "is_verified",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["viral_label"] = df["viral"].map({0: "Not viral", 1: "Viral"})
    df["total_engagement"] = df["likes"] + df["comments"]
    df["log_followers"] = np.log1p(df["followers"])
    df["log_engagement"] = np.log1p(df["total_engagement"])
    return df


def setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="husl", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def fig01_engagement_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df["engagement_rate"], bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(
        df["virality_threshold"].iloc[0],
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"Viral threshold ({df['virality_threshold'].iloc[0]:.3f})",
    )
    axes[0].set_xlabel("Engagement rate (likes + comments) / followers")
    axes[0].set_ylabel("Number of posts")
    axes[0].set_title("Distribution of engagement rate")
    axes[0].legend()

    axes[1].hist(
        np.log1p(df["engagement_rate"]),
        bins=60,
        color="#55A868",
        edgecolor="white",
        alpha=0.85,
    )
    axes[1].set_xlabel("log(1 + engagement rate)")
    axes[1].set_ylabel("Number of posts")
    axes[1].set_title("Log-scaled engagement (easier to read tail)")

    fig.suptitle("Fig 1 — Engagement distribution", fontsize=13, y=1.02)
    fig.savefig(FIG_DIR / "01_engagement_distribution.png")
    plt.close(fig)


def fig02_viral_vs_nonviral(df: pd.DataFrame) -> None:
    metrics = ["likes", "comments", "caption_length", "hashtag_count", "engagement_rate"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, metrics):
        sns.boxplot(data=df, x="viral_label", y=col, hue="viral_label", ax=ax, legend=False)
        ax.set_xlabel("")
        ax.set_title(col.replace("_", " ").title())

    axes[-1].axis("off")
    fig.suptitle("Fig 2 — Viral vs non-viral posts (key metrics)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_viral_vs_nonviral_boxplots.png")
    plt.close(fig)


def fig03_posting_time(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Hour: viral rate by hour
    hour_stats = (
        df.groupby("post_hour", observed=True)
        .agg(posts=("viral", "count"), viral_rate=("viral", "mean"))
        .reset_index()
    )
    ax = axes[0]
    ax.bar(hour_stats["post_hour"], hour_stats["viral_rate"], color="#8172B3", alpha=0.85)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Viral rate")
    ax.set_title("Viral rate by posting hour")
    ax.set_xticks(range(0, 24, 2))

    # Day of week heatmap: post count + viral rate
    pivot_count = df.pivot_table(
        index="post_day_of_week",
        columns="post_hour",
        values="viral",
        aggfunc="count",
        fill_value=0,
    )
    pivot_viral = df.pivot_table(
        index="post_day_of_week",
        columns="post_hour",
        values="viral",
        aggfunc="mean",
        fill_value=np.nan,
    )

    sns.heatmap(
        pivot_viral,
        ax=axes[1],
        cmap="YlOrRd",
        linewidths=0.2,
        cbar_kws={"label": "Viral rate"},
        yticklabels=DAY_NAMES,
    )
    axes[1].set_xlabel("Hour of day (UTC)")
    axes[1].set_ylabel("Day of week")
    axes[1].set_title("Viral rate heatmap (day × hour)")

    fig.suptitle("Fig 3 — Posting time patterns", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_posting_time_patterns.png")
    plt.close(fig)


def fig04_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[NUMERIC_FOR_CORR + ["viral"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Fig 4 — Feature correlation matrix")
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png")
    plt.close(fig)


def fig05_hashtag_impact(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hashtag_bins = pd.cut(
        df["hashtag_count"],
        bins=[-1, 0, 5, 10, 20, 50, 200],
        labels=["0", "1-5", "6-10", "11-20", "21-50", "50+"],
    )
    hashtag_stats = (
        df.assign(hashtag_bin=hashtag_bins)
        .groupby("hashtag_bin", observed=True)
        .agg(posts=("viral", "count"), viral_rate=("viral", "mean"), avg_engagement=("engagement_rate", "mean"))
        .reset_index()
    )

    x = range(len(hashtag_stats))
    axes[0].bar(x, hashtag_stats["viral_rate"], color="#DD8452", alpha=0.9)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(hashtag_stats["hashtag_bin"], rotation=0)
    axes[0].set_xlabel("Hashtag count (binned)")
    axes[0].set_ylabel("Viral rate")
    axes[0].set_title("Viral rate by hashtag count")

    sns.scatterplot(
        data=df.sample(min(3000, len(df)), random_state=42),
        x="hashtag_count",
        y="engagement_rate",
        hue="viral_label",
        alpha=0.35,
        ax=axes[1],
    )
    axes[1].set_xlabel("Hashtag count")
    axes[1].set_ylabel("Engagement rate")
    axes[1].set_title("Hashtags vs engagement (sample)")
    axes[1].legend(title="", loc="upper right", fontsize=8)

    fig.suptitle("Fig 5 — Hashtag impact on virality", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_hashtag_impact.png")
    plt.close(fig)


def fig06_media_type(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    media_stats = (
        df.groupby("media_type")
        .agg(posts=("viral", "count"), viral_rate=("viral", "mean"), avg_likes=("likes", "mean"))
        .reset_index()
    )
    sns.barplot(data=media_stats, x="media_type", y="viral_rate", hue="media_type", ax=axes[0], legend=False)
    axes[0].set_ylabel("Viral rate")
    axes[0].set_title("Viral rate by media type")

    sns.boxplot(
        data=df,
        x="media_type",
        y="likes",
        hue="viral_label",
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Likes (log scale)")
    axes[1].set_title("Likes by media type and virality")
    axes[1].legend(title="", loc="upper right", fontsize=8)

    fig.suptitle("Fig 6 — Image vs video performance", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_media_type_comparison.png")
    plt.close(fig)


def fig07_followers_engagement(df: pd.DataFrame) -> None:
    sample = df.sample(min(4000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=sample,
        x="log_followers",
        y="log_engagement",
        hue="viral_label",
        alpha=0.4,
        ax=ax,
    )
    ax.set_xlabel("log(1 + followers)")
    ax.set_ylabel("log(1 + likes + comments)")
    ax.set_title("Fig 7 — Account size vs post engagement")
    ax.legend(title="", loc="lower right")
    fig.savefig(FIG_DIR / "07_followers_vs_engagement.png")
    plt.close(fig)


def fig08_top_categories(df: pd.DataFrame, top_n: int = 12) -> None:
    top_cats = df["category_name"].value_counts().head(top_n).index
    subset = df[df["category_name"].isin(top_cats)]
    cat_stats = (
        subset.groupby("category_name")
        .agg(posts=("viral", "count"), viral_rate=("viral", "mean"))
        .reset_index()
        .sort_values("viral_rate", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(cat_stats["category_name"], cat_stats["viral_rate"], color="#64B5CD")
    ax.set_xlabel("Viral rate")
    ax.set_title(f"Fig 8 — Viral rate by top {top_n} content categories")
    fig.savefig(FIG_DIR / "08_category_viral_rate.png")
    plt.close(fig)


def fig09_verified_accounts(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    verify_stats = df.groupby("is_verified").agg(viral_rate=("viral", "mean"), posts=("viral", "count"))
    labels = ["Not verified", "Verified"]
    ax.bar(labels, verify_stats["viral_rate"].values, color=["#8C8C8C", "#4C72B0"])
    ax.set_ylabel("Viral rate")
    ax.set_title("Fig 9 — Viral rate: verified vs non-verified accounts")
    for i, (_, row) in enumerate(verify_stats.iterrows()):
        ax.text(i, row["viral_rate"] + 0.002, f"n={int(row['posts']):,}", ha="center", fontsize=9)
    fig.savefig(FIG_DIR / "09_verified_comparison.png")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    threshold = df["virality_threshold"].iloc[0]
    viral_pct = df["viral"].mean() * 100

    viral = df[df["viral"] == 1]
    nonviral = df[df["viral"] == 0]

    lines = [
        "# EDA Summary — Social Media Virality Project",
        "",
        f"- **Posts analyzed:** {len(df):,}",
        f"- **Viral threshold (engagement rate):** {threshold:.4f}",
        f"- **Viral posts:** {viral_pct:.1f}%",
        "",
        "## Key comparisons (viral vs non-viral median)",
        "",
        f"| Metric | Viral | Non-viral |",
        f"|--------|-------|-----------|",
    ]
    for col in ["likes", "comments", "hashtag_count", "caption_length", "engagement_rate"]:
        lines.append(
            f"| {col} | {viral[col].median():.2f} | {nonviral[col].median():.2f} |"
        )

    media = df.groupby("media_type")["viral"].mean()
    lines.extend(["", "## Viral rate by media type", ""])
    for m, rate in media.items():
        lines.append(f"- **{m}:** {rate:.1%}")

    hour_best = df.groupby("post_hour")["viral"].mean().idxmax()
    hour_rate = df.groupby("post_hour")["viral"].mean().max()
    lines.extend([
        "",
        "## Posting time",
        f"- Highest viral rate hour (UTC): **{int(hour_best)}:00** ({hour_rate:.1%})",
        "",
        "## Hashtags",
    ])
    zero_rate = df[df["hashtag_count"] == 0]["viral"].mean()
    high_rate = df[df["hashtag_count"] >= 11]["viral"].mean()
    lines.append(f"- Viral rate with 0 hashtags: **{zero_rate:.1%}**")
    lines.append(f"- Viral rate with 11+ hashtags: **{high_rate:.1%}**")

    lines.extend([
        "",
        "## Expected observations (from project doc)",
        "- Higher engagement rate correlates with virality — check Fig 1–2.",
        "- Video content may outperform images — check Fig 6.",
        "- Extreme hashtag use may not always help — check Fig 5.",
        "",
        "## Figures",
        "",
    ])
    for p in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- `{p.name}`")

    (FIG_DIR / "EDA_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run: python3 data_cleaning.py"
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = load_data()

    print(f"Loaded {len(df):,} posts | Viral: {df['viral'].mean():.1%}")
    print("Generating figures...")

    fig01_engagement_distribution(df)
    fig02_viral_vs_nonviral(df)
    fig03_posting_time(df)
    fig04_correlation_heatmap(df)
    fig05_hashtag_impact(df)
    fig06_media_type(df)
    fig07_followers_engagement(df)
    fig08_top_categories(df)
    fig09_verified_accounts(df)
    write_summary(df)

    n_figs = len(list(FIG_DIR.glob("*.png")))
    print(f"Saved {n_figs} figures to {FIG_DIR}/")
    print(f"Summary: {FIG_DIR / 'EDA_SUMMARY.md'}")


if __name__ == "__main__":
    main()
