# utils/preprocessing.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ── Action Weights ─────────────────────────────────────────────────────────────
# These numbers are a design choice, not a law. You could tune them later.
ACTION_WEIGHTS = {
    "share": 3.0,
    "like":  2.0,
    "view":  1.0,
}


def load_raw_data(data_dir: str = "data/raw"):
    """Load the three CSVs we generated in Step 1."""
    articles     = pd.read_csv(f"{data_dir}/articles.csv")
    users        = pd.read_csv(f"{data_dir}/users.csv")
    interactions = pd.read_csv(f"{data_dir}/interactions.csv")
    return articles, users, interactions


def build_article_features(articles: pd.DataFrame):
    """
    Combine each article's category, tags, and content into one text blob,
    then fit a TF-IDF vectorizer across all articles.

    Why combine fields?
    - Content alone is noisy (lots of filler words)
    - Category + tags are clean signals
    - Combining gives the vectorizer more to work with

    Returns:
        tfidf_matrix : sparse matrix, shape (n_articles, n_features)
        vectorizer   : fitted TfidfVectorizer (we'll need this later
                       to transform NEW articles at serving time)
        article_ids  : list of article_ids in the same row order
                       as tfidf_matrix — critical for index alignment
    """
    # Build one text string per article that blends all signals.
    # We repeat category and tags to give them more weight than body text.
    articles["text_blob"] = (
        articles["category"] + " " +
        articles["category"] + " " +    # repeated → higher TF weight
        articles["tags"]     + " " +
        articles["tags"]     + " " +    # repeated
        articles["content"]
    )

    vectorizer = TfidfVectorizer(
        max_features=500,       # keep top 500 most informative words
        stop_words="english",   # drop "the", "is", "and", etc.
        ngram_range=(1, 2),     # capture both single words AND two-word phrases
                                # e.g. "machine learning" as one feature
        min_df=2,               # ignore words appearing in fewer than 2 articles
    )

    tfidf_matrix = vectorizer.fit_transform(articles["text_blob"])
    # Shape: (200 articles × up to 500 features)
    # It's a *sparse* matrix — mostly zeros, stored efficiently

    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Sample features: {vectorizer.get_feature_names_out()[:10]}")

    return tfidf_matrix, vectorizer, articles["article_id"].tolist()


def build_user_profiles(users:        pd.DataFrame,
                        interactions: pd.DataFrame,
                        tfidf_matrix: object,
                        article_ids:  list) -> dict:
    """
    For each user, compute a weighted average of the TF-IDF vectors
    of articles they've interacted with.

    The result is a single vector per user that represents their taste.
    Users with similar vectors have similar tastes → collaborative signal.

    Returns:
        profiles: dict mapping user_id → numpy array (taste vector)
    """
    # Build a lookup: article_id → row index in tfidf_matrix
    # This is just for fast access — don't skip this, it matters for speed
    article_index = {aid: idx for idx, aid in enumerate(article_ids)}

    profiles = {}

    for _, user in users.iterrows():
        user_id = user["user_id"]

        # Get all interactions for this user
        user_actions = interactions[interactions["user_id"] == user_id]

        if user_actions.empty:
            # Cold start problem: new user, no history → can't build profile
            # We'll handle this in the UI with an onboarding step later
            profiles[user_id] = None
            continue

        weighted_sum = np.zeros(tfidf_matrix.shape[1])
        total_weight = 0.0

        for _, row in user_actions.iterrows():
            article_id = row["article_id"]
            action     = row["action"]

            if article_id not in article_index:
                continue   # safety check — article might have been removed

            weight        = ACTION_WEIGHTS.get(action, 1.0)
            article_vec   = tfidf_matrix[article_index[article_id]].toarray()[0]
            weighted_sum += weight * article_vec
            total_weight += weight

        if total_weight > 0:
            profile_vector = weighted_sum / total_weight
            # Normalize to unit length so cosine similarity works correctly
            profiles[user_id] = normalize(
                profile_vector.reshape(1, -1)
            ).flatten()
        else:
            profiles[user_id] = None

    active_profiles = sum(1 for v in profiles.values() if v is not None)
    print(f"  Built profiles for {active_profiles}/{len(users)} users")

    return profiles


def save_processed_data(tfidf_matrix, vectorizer,
                        article_ids, profiles,
                        out_dir: str = "data/processed"):
    """
    Persist everything to disk.
    .pkl (pickle) files store Python objects exactly — including the
    fitted vectorizer, which we need at serving time to transform new text.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save TF-IDF matrix + article ID order together so they never get out of sync
    with open(f"{out_dir}/tfidf_matrix.pkl", "wb") as f:
        pickle.dump({"matrix": tfidf_matrix, "article_ids": article_ids}, f)

    with open(f"{out_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"{out_dir}/user_profiles.pkl", "wb") as f:
        pickle.dump(profiles, f)

    print(f"  ✓ Saved tfidf_matrix.pkl, vectorizer.pkl, user_profiles.pkl")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading raw data...")
    articles, users, interactions = load_raw_data()

    print("\nBuilding article TF-IDF features...")
    tfidf_matrix, vectorizer, article_ids = build_article_features(articles)

    print("\nBuilding user taste profiles...")
    profiles = build_user_profiles(users, interactions,
                                   tfidf_matrix, article_ids)

    print("\nSaving processed data...")
    save_processed_data(tfidf_matrix, vectorizer, article_ids, profiles)

    print("\n── Sample profile for U001 ───────────────────────────────")
    if profiles.get("U001") is not None:
        vec = profiles["U001"]
        # Show the top 5 dimensions (words) with highest weight
        feature_names = vectorizer.get_feature_names_out()
        top5_idx      = vec.argsort()[-5:][::-1]
        print("  Top 5 taste signals:")
        for idx in top5_idx:
            print(f"    '{feature_names[idx]}': {vec[idx]:.4f}")
    else:
        print("  U001 has no interaction history yet (cold start)")