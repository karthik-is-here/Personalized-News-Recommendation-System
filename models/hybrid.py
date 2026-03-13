# models/hybrid.py
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
import pickle
import pandas as pd
import numpy as np
import os

# Import our two recommenders
from models.content_based  import get_recommendations as cb_recommend
from models.content_based  import load_artifacts      as cb_load
from models.collaborative  import (get_recommendations as cf_recommend,
                                   load_model,
                                   build_ratings_df)


# ── Adaptive weight thresholds ─────────────────────────────────────────────────
# How many interactions a user needs before we trust collaborative more.
# These are reasonable defaults — you could tune them later.
COLD_START_THRESHOLD  = 5    # fewer than this → user is "new"
WARM_USER_THRESHOLD   = 20   # more than this  → user is "experienced"


def _get_user_interaction_count(user_id: str,
                                interactions: pd.DataFrame) -> int:
    return len(interactions[interactions["user_id"] == user_id])


def _adaptive_weights(interaction_count: int) -> tuple:
    """
    Returns (content_weight, collab_weight) that sum to 1.0.

    Transition is LINEAR between the two thresholds — not a hard jump.
    This avoids a jarring change in recommendations as a user crosses
    an arbitrary threshold.

                  content_w
    1.0 |████████\
        |         \
    0.5 |          ●----------
        |         /
    0.0 |________/____________
              5       20     interactions
    """
    if interaction_count <= COLD_START_THRESHOLD:
        return 0.85, 0.15   # mostly content-based

    if interaction_count >= WARM_USER_THRESHOLD:
        return 0.30, 0.70   # mostly collaborative

    # Linear interpolation between the two extremes
    t = (interaction_count - COLD_START_THRESHOLD) / \
        (WARM_USER_THRESHOLD - COLD_START_THRESHOLD)   # 0.0 → 1.0
    content_w = 0.85 - (0.55 * t)   # 0.85 → 0.30
    collab_w  = 1.0 - content_w
    return round(content_w, 3), round(collab_w, 3)


def _normalize_scores(df: pd.DataFrame,
                      score_col: str) -> pd.DataFrame:
    """
    Min-max normalize a score column to [0, 1].

    Formula: (x - min) / (max - min)

    Edge case: if all scores are identical (max == min),
    normalization would divide by zero → return 0.5 for all.
    """
    df = df.copy()
    min_s = df[score_col].min()
    max_s = df[score_col].max()

    if max_s == min_s:
        df[score_col] = 0.5
    else:
        df[score_col] = (df[score_col] - min_s) / (max_s - min_s)

    return df


def get_recommendations(user_id:   str,
                        top_n:     int  = 10,
                        processed_dir: str = "data/processed",
                        raw_dir:   str  = "data/raw",
                        model_dir: str  = "saved_models") -> pd.DataFrame:
    """
    Hybrid recommendations: blend content-based + collaborative scores.

    Returns DataFrame with columns:
        article_id, title, category, tags, publish_date,
        cb_score, cf_score, hybrid_score, content_weight, collab_weight
    """
    articles     = pd.read_csv(f"{raw_dir}/articles.csv")
    interactions = pd.read_csv(f"{raw_dir}/interactions.csv")

    n_interactions = _get_user_interaction_count(user_id, interactions)
    content_w, collab_w = _adaptive_weights(n_interactions)

    print(f"  User {user_id}: {n_interactions} interactions → "
          f"weights (content={content_w}, collab={collab_w})")

    # ── Content-based scores ───────────────────────────────────────────────
    # Fetch more than top_n so the merge has overlap to work with
    cb_recs = cb_recommend(
        user_id, top_n=len(articles),
        processed_dir=processed_dir, raw_dir=raw_dir
    )[["article_id", "similarity_score"]].rename(
        columns={"similarity_score": "cb_score"}
    )

    # ── Collaborative scores ───────────────────────────────────────────────
    model, trainset = load_model(model_dir)
    cf_recs = cf_recommend(
        user_id, model, trainset, articles, interactions,
        top_n=len(articles)
    )[["article_id", "predicted_rating"]].rename(
        columns={"predicted_rating": "cf_score"}
    )

    # ── Normalize both to [0, 1] BEFORE blending ───────────────────────────
    cb_recs = _normalize_scores(cb_recs, "cb_score")
    cf_recs = _normalize_scores(cf_recs, "cf_score")

    # ── Merge on article_id ────────────────────────────────────────────────
    # outer join: keep articles scored by either model
    # articles missing from one model get NaN → fill with 0.0
    merged = (cb_recs
              .merge(cf_recs, on="article_id", how="outer")
              .fillna(0.0))

    # ── Weighted blend ─────────────────────────────────────────────────────
    merged["hybrid_score"] = (
        content_w * merged["cb_score"] +
        collab_w  * merged["cf_score"]
    ).round(4)

    # Store weights in the DataFrame — useful for UI explainability
    merged["content_weight"] = content_w
    merged["collab_weight"]  = collab_w

    # ── Filter already-read, rank, enrich ─────────────────────────────────
    already_read = set(
        interactions[interactions["user_id"] == user_id]["article_id"]
    )
    merged = merged[~merged["article_id"].isin(already_read)]

    results = (merged
               .sort_values("hybrid_score", ascending=False)
               .head(top_n)
               .merge(articles[["article_id", "title",
                                 "category", "tags", "publish_date"]],
                      on="article_id")
               .reset_index(drop=True))

    return results


def explain_recommendation(row: pd.Series) -> str:
    """
    Generate a human-readable explanation for a recommendation.
    Explainability builds user trust — always worth adding.
    """
    cb  = row["cb_score"]
    cf  = row["cf_score"]
    cw  = row["content_weight"]
    cow = row["collab_weight"]

    if cw >= 0.7:
        driver = "content match"
        reason = f"matches your reading interests closely (score: {cb:.2f})"
    elif cow >= 0.7:
        driver = "readers like you"
        reason = f"highly rated by users with similar tastes (score: {cf:.2f})"
    else:
        driver = "mixed signals"
        reason = (f"both content match ({cb:.2f}) and "
                  f"reader similarity ({cf:.2f}) suggest this")

    return f"Recommended because of {driver}: {reason}"


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for test_user in ["U001", "U010", "U025"]:
        print("\n" + "═" * 60)
        print(f"  Hybrid Recommendations — {test_user}")
        print("═" * 60)

        recs = get_recommendations(test_user, top_n=5)

        for _, row in recs.iterrows():
            print(f"\n  [{row['category']}] {row['title']}")
            print(f"  Hybrid: {row['hybrid_score']:.4f}  "
                  f"(CB: {row['cb_score']:.3f} × {row['content_weight']} | "
                  f"CF: {row['cf_score']:.3f} × {row['collab_weight']})")
            print(f"  💡 {explain_recommendation(row)}")