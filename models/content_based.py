# models/content_based.py

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_artifacts(processed_dir: str = "data/processed",
                   raw_dir:       str = "data/raw"):
    """
    Load everything we saved in preprocessing.
    Keeping this in one place means we only change it here if paths change.
    """
    with open(f"{processed_dir}/tfidf_matrix.pkl", "rb") as f:
        tfidf_data = pickle.load(f)

    with open(f"{processed_dir}/user_profiles.pkl", "rb") as f:
        user_profiles = pickle.load(f)

    articles = pd.read_csv(f"{raw_dir}/articles.csv")
    interactions = pd.read_csv(f"{raw_dir}/interactions.csv")

    return (tfidf_data["matrix"],
            tfidf_data["article_ids"],
            user_profiles,
            articles,
            interactions)


def get_recommendations(user_id:       str,
                        top_n:         int  = 10,
                        processed_dir: str  = "data/processed",
                        raw_dir:       str  = "data/raw") -> pd.DataFrame:
    """
    Content-based recommendations for a single user.

    Args:
        user_id  : e.g. "U001"
        top_n    : how many articles to return

    Returns:
        DataFrame with columns:
            article_id, title, category, tags, similarity_score
        sorted by similarity_score descending
    """
    tfidf_matrix, article_ids, user_profiles, articles, interactions = \
        load_artifacts(processed_dir, raw_dir)

    # ── Guard: cold start ──────────────────────────────────────────────────
    profile = user_profiles.get(user_id)
    if profile is None:
        print(f"  No profile found for {user_id} (cold start).")
        # Fallback: return the most recently published articles
        return (articles
                .sort_values("publish_date", ascending=False)
                .head(top_n)[["article_id", "title", "category", "tags"]]
                .assign(similarity_score=None))

    # ── Cosine similarity ──────────────────────────────────────────────────
    # profile shape   : (1, n_features)  — one row vector
    # tfidf_matrix    : (n_articles, n_features)
    # scores shape    : (1, n_articles)  — one similarity per article
    scores = cosine_similarity(
        profile.reshape(1, -1),   # user taste vector
        tfidf_matrix              # all article vectors at once (fast!)
    ).flatten()                   # → 1-D array of length n_articles

    # ── Build a results DataFrame ──────────────────────────────────────────
    results = pd.DataFrame({
        "article_id":       article_ids,
        "similarity_score": scores,
    })

    # ── Filter already-read articles ──────────────────────────────────────
    # Recommending something they've already seen is a bad experience.
    already_read = set(
        interactions[interactions["user_id"] == user_id]["article_id"]
    )
    results = results[~results["article_id"].isin(already_read)]

    # ── Rank and enrich ────────────────────────────────────────────────────
    results = (results
               .sort_values("similarity_score", ascending=False)
               .head(top_n)
               .merge(articles[["article_id", "title",
                                 "category", "tags",
                                 "publish_date"]],
                      on="article_id")
               .round({"similarity_score": 4}))

    return results.reset_index(drop=True)


def get_similar_articles(article_id:    str,
                         top_n:         int = 5,
                         processed_dir: str = "data/processed",
                         raw_dir:       str = "data/raw") -> pd.DataFrame:
    """
    Bonus function: given an article, find the most similar ones.
    This powers "You might also like..." sections in the UI.

    Same cosine similarity logic — but we compare one article
    vector against all others, instead of a user profile.
    """
    tfidf_matrix, article_ids, _, articles, _ = \
        load_artifacts(processed_dir, raw_dir)

    if article_id not in article_ids:
        return pd.DataFrame()

    idx    = article_ids.index(article_id)
    # Extract this article's row vector and compare against all others
    scores = cosine_similarity(
        tfidf_matrix[idx],   # shape: (1, n_features)
        tfidf_matrix         # shape: (n_articles, n_features)
    ).flatten()

    results = pd.DataFrame({
        "article_id":       article_ids,
        "similarity_score": scores,
    })

    # Exclude the article itself (it will always score 1.0 against itself)
    results = (results[results["article_id"] != article_id]
               .sort_values("similarity_score", ascending=False)
               .head(top_n)
               .merge(articles[["article_id", "title", "category", "tags"]],
                      on="article_id")
               .round({"similarity_score": 4}))

    return results.reset_index(drop=True)


# ── Quick demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 55)
    print("  Content-Based Recommendations for U001")
    print("═" * 55)
    recs = get_recommendations("U001", top_n=5)
    print(recs[["title", "category", "similarity_score"]].to_string())

    print("\n" + "═" * 55)
    print("  Articles similar to A0001")
    print("═" * 55)
    similars = get_similar_articles("A0001", top_n=3)
    print(similars[["title", "category", "similarity_score"]].to_string())