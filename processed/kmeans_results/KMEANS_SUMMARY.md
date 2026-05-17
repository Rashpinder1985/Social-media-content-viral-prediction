# K-Means Clustering Summary — Social Media Virality Project

## Setup
- **Features:** followers, likes, comments, caption_length, hashtag_count, post_hour, post_day_of_week, engagement_rate, has_video_views, is_verified, is_video
- **K tested:** [2, 3, 4, 5, 6, 7, 8, 9, 10]
- **Selected K:** 3 (best silhouette score)
- **Overall viral rate:** 10.0%

## Elbow & silhouette

|     k |   inertia |   silhouette |
|------:|----------:|-------------:|
|  2.00 | 222514.81 |         0.40 |
|  3.00 | 197721.27 |         0.41 |
|  4.00 | 174181.93 |         0.22 |
|  5.00 | 163498.81 |         0.23 |
|  6.00 | 138757.85 |         0.24 |
|  7.00 | 130434.60 |         0.20 |
|  8.00 | 119089.90 |         0.21 |
|  9.00 | 113752.99 |         0.21 |
| 10.00 | 104010.81 |         0.22 |

## Cluster profiles

|   cluster |   posts |   viral_rate |   avg_engagement |   avg_likes |   avg_comments |   avg_hashtags |   avg_caption_len |   avg_followers |   pct_video |   pct_verified |   avg_hour | cluster_label                              |
|----------:|--------:|-------------:|-----------------:|------------:|---------------:|---------------:|------------------:|----------------:|------------:|---------------:|-----------:|:-------------------------------------------|
|         0 |   21311 |        0.103 |            0.099 |     128.371 |          4.288 |         14.956 |           442.561 |        3663.219 |       0.000 |          0.000 |     12.676 | High-viral / image-heavy / smaller-account |
|         1 |    3407 |        0.085 |            0.086 |     218.193 |          4.863 |         13.472 |           356.398 |        5188.962 |       1.000 |          0.000 |     12.652 | video-heavy / large-account                |
|         2 |      24 |        0.000 |            0.029 |     141.875 |          5.292 |         14.542 |          1010.500 |        5358.500 |       0.000 |          1.000 |     12.917 | Low-viral / image-heavy / large-account    |

## Key insights

- **Highest viral cluster:** C0 (High-viral / image-heavy / smaller-account) — **10.3%** viral rate (21,311 posts)
- **Lowest viral cluster:** C2 (Low-viral / image-heavy / large-account) — **0.0%** viral rate (24 posts)
- **Spread:** 10.3 percentage points between best and worst cluster.

## Integrated insight (Day 5 — link to K-NN)

- Certain content/behavior clusters naturally produce more viral posts.
- Creators could shift toward high-viral clusters (e.g. similar engagement + format patterns).
- Platforms promoting only high-engagement clusters may amplify bias and echo chambers.

## Figures

- `01_elbow_plot.png`
- `02_cluster_pca.png`
- `03_viral_rate_by_cluster.png`
- `04_centroid_heatmap.png`