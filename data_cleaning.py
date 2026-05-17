"""
Social Media Virality Project — Data Cleaning & Feature Engineering
Maps instagram_dataset.csv (account + nested posts) to post-level ML-ready data.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
RAW_CSV = PROJECT_DIR / "instagram_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "processed"
VIRALITY_PERCENTILE = 0.90  # top 10% engagement rate = viral (document: student-defined threshold)
RANDOM_STATE = 42


def parse_posts(raw_value) -> list[dict]:
    """Parse nested posts column (Python/JSON-like list of dicts)."""
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if text in ("", "[]", "nan"):
        return []
    normalized = (
        text.replace("null", "None")
        .replace("true", "True")
        .replace("false", "False")
    )
    try:
        parsed = ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def count_hashtags(caption: str | None) -> int:
    if not caption or not isinstance(caption, str):
        return 0
    return len(re.findall(r"#\w+", caption))


def caption_length(caption: str | None) -> int:
    if not caption or not isinstance(caption, str):
        return 0
    return len(caption.strip())


def media_type(post: dict) -> str:
    if post.get("video_url"):
        return "video"
    if post.get("image_url"):
        return "image"
    return "unknown"


def flatten_to_posts(accounts: pd.DataFrame) -> pd.DataFrame:
    """One row per post with account-level context merged in."""
    rows: list[dict] = []

    for _, account in accounts.iterrows():
        posts = parse_posts(account.get("posts"))
        followers = safe_float(account.get("followers"), 1.0)
        following = safe_float(account.get("following"))
        is_verified = str(account.get("is_verified", "no")).lower() in ("yes", "true", "1")

        for post in posts:
            if not isinstance(post, dict):
                continue

            likes = safe_float(post.get("likes"))
            comments = safe_float(post.get("comments"))
            video_views = safe_float(post.get("video_view_count"))
            caption = post.get("caption") or ""
            dt = post.get("datetime")

            engagement_rate = (likes + comments) / max(followers, 1.0)

            rows.append(
                {
                    "account": account.get("account"),
                    "post_id": post.get("id"),
                    "followers": followers,
                    "following": following,
                    "is_verified": int(is_verified),
                    "likes": likes,
                    "comments": comments,
                    "video_view_count": video_views,
                    "caption": caption,
                    "caption_length": caption_length(caption),
                    "hashtag_count": count_hashtags(caption),
                    "datetime_unix": safe_float(dt) if dt is not None else np.nan,
                    "media_type": media_type(post),
                    "engagement_rate": engagement_rate,
                    "account_avg_engagement": safe_float(account.get("avg_engagement")),
                    "posts_count": safe_float(account.get("posts_count")),
                    "category_name": account.get("category_name") or account.get("business_category_name"),
                }
            )

    return pd.DataFrame(rows)


def add_time_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Posting hour and day-of-week from unix timestamp."""
    out = posts.copy()
    ts = pd.to_datetime(out["datetime_unix"], unit="s", errors="coerce", utc=True)
    out["post_hour"] = ts.dt.hour
    out["post_day_of_week"] = ts.dt.dayofweek
    out["post_month"] = ts.dt.month
    return out


def define_viral_target(posts: pd.DataFrame, percentile: float = VIRALITY_PERCENTILE) -> pd.DataFrame:
    """Binary target: 1 if engagement_rate >= threshold (document requirement)."""
    out = posts.copy()
    threshold = out["engagement_rate"].quantile(percentile)
    out["viral"] = (out["engagement_rate"] >= threshold).astype(int)
    out["virality_threshold"] = threshold
    return out


def clean_posts(posts: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, outliers, and invalid rows."""
    df = posts.copy()

    # Drop rows without usable engagement signal
    df = df.dropna(subset=["likes", "comments", "followers"])
    df = df[df["followers"] > 0]

    # Cap extreme engagement (data errors / bot spikes) at 99.5th percentile
    cap = df["engagement_rate"].quantile(0.995)
    df["engagement_rate"] = df["engagement_rate"].clip(upper=cap)

    # Fill missing time features with median
    for col in ("post_hour", "post_day_of_week", "post_month"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["category_name"] = df["category_name"].fillna("Unknown")
    df["media_type"] = df["media_type"].replace("unknown", np.nan)
    df["media_type"] = df["media_type"].fillna(df["media_type"].mode().iloc[0] if len(df) else "image")

    # Shares not in dataset — document lists them; we use video_view_count as optional proxy for video posts
    df["has_video_views"] = (df["video_view_count"] > 0).astype(int)

    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Numeric features for K-NN / K-Means + encoded categoricals.
    Returns (feature_df, numeric_cols, categorical_cols).
    """
    numeric_cols = [
        "followers",
        "following",
        "is_verified",
        "likes",
        "comments",
        "caption_length",
        "hashtag_count",
        "post_hour",
        "post_day_of_week",
        "posts_count",
        "account_avg_engagement",
        "video_view_count",
        "has_video_views",
    ]
    categorical_cols = ["media_type", "category_name"]

    features = df[numeric_cols + categorical_cols].copy()
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    return features, numeric_cols, categorical_cols


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading raw Instagram dataset...")
    accounts = pd.read_csv(RAW_CSV, encoding="latin-1", low_memory=False)
    accounts = accounts.loc[:, ~accounts.columns.str.match(r"^Unnamed")]

    print(f"  Accounts loaded: {len(accounts):,}")

    print("Flattening nested posts to post-level rows...")
    posts = flatten_to_posts(accounts)
    print(f"  Raw posts extracted: {len(posts):,}")

    posts = add_time_features(posts)
    posts = define_viral_target(posts)
    posts = clean_posts(posts)

    threshold = posts["virality_threshold"].iloc[0]
    viral_rate = posts["viral"].mean()
    print(f"  Virality threshold (p{VIRALITY_PERCENTILE:.0%}): {threshold:.6f}")
    print(f"  Viral posts: {posts['viral'].sum():,} ({viral_rate:.1%})")

    # Save human-readable cleaned data (before scaling)
    posts.to_csv(OUTPUT_DIR / "posts_cleaned.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'posts_cleaned.csv'}")

    X, _, _ = build_feature_matrix(posts)
    y = posts["viral"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        posts.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    train_out = X_train_scaled.copy()
    train_out["viral"] = y_train.values
    test_out = X_test_scaled.copy()
    test_out["viral"] = y_test.values

    train_out.to_csv(OUTPUT_DIR / "train_scaled.csv")
    test_out.to_csv(OUTPUT_DIR / "test_scaled.csv")

    # Metadata for reproducibility
    meta = pd.DataFrame(
        {
            "item": [
                "virality_percentile",
                "virality_threshold",
                "train_rows",
                "test_rows",
                "feature_count",
                "viral_rate_overall",
            ],
            "value": [
                VIRALITY_PERCENTILE,
                threshold,
                len(train_out),
                len(test_out),
                X.shape[1],
                viral_rate,
            ],
        }
    )
    meta.to_csv(OUTPUT_DIR / "cleaning_metadata.csv", index=False)

    print(f"  Train: {len(train_out):,} | Test: {len(test_out):,} | Features: {X.shape[1]}")
    print(f"  Saved scaled splits to {OUTPUT_DIR}/")
    print("Done. Next: EDA (Day 3) → K-NN (Day 4) → K-Means (Day 5).")


if __name__ == "__main__":
    main()
