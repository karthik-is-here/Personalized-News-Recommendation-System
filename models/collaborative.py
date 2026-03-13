# models/collaborative.py
# Pure numpy/scikit-learn SVD — no scikit-surprise dependency

import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os

ACTION_TO_RATING = {
    "share": 5.0,
    "like":  3.5,
    "view":  2.0,
}


def build_ratings_df(interactions: pd.DataFrame) -> pd.DataFrame:
    """Convert interactions to one rating per (user, article) pair."""
    interactions = interactions.copy()
    interactions["rating"] = interactions["action"].map(ACTION_TO_RATING)
    ratings = (interactions
               .groupby(["user_id", "article_id"])["rating"]
               .max()
               .reset_index())
    print(f"  Ratings: {ratings['user_id'].nunique()} users × "
          f"{ratings['article_id'].nunique()} articles "
          f"({len(ratings)} observed)")
    return ratings


def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Build a dense user-item matrix from the ratings DataFrame.

    Rows = users, Columns = articles, Values = ratings (0 if unrated).
    Also returns the ordered lists of user_ids and article_ids so we
    can map back from matrix indices to real IDs later.
    """
    user_ids    = sorted(ratings["user_id"].unique())
    article_ids = sorted(ratings["article_id"].unique())

    user_idx    = {u: i for i, u in enumerate(user_ids)}
    article_idx = {a: i for i, a in enumerate(article_ids)}

    matrix = np.zeros((len(user_ids), len(article_ids)), dtype=np.float32)

    for _, row in ratings.iterrows():
        u = user_idx[row["user_id"]]
        a = article_idx[row["article_id"]]
        matrix[u][a] = row["rating"]

    print(f"  User-item matrix shape: {matrix.shape}")
    print(f"  Sparsity: {100 * (matrix == 0).sum() / matrix.size:.1f}% zeros")
    return matrix, user_ids, article_ids


def train_collaborative_model(ratings:   pd.DataFrame,
                              n_factors: int  = 20,
                              run_cv:    bool = True):
    """
    Train a matrix factorization model using TruncatedSVD.

    TruncatedSVD decomposes the user-item matrix into:
        U  (users   × factors)
        Σ  (factors diagonal — importance of each factor)
        Vt (factors × articles)

    To predict a rating: U[user] · Σ · Vt[:, article]
    We reconstruct the full matrix and read off predictions.
    """
    matrix, user_ids, article_ids = build_user_item_matrix(ratings)

    # n_factors = number of latent dimensions to keep
    # More factors = more expressive, but slower and risks overfitting
    n_factors = min(n_factors, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    svd.fit(matrix)

    # Reconstruct the full matrix — this fills in the zeros with predictions
    # Shape: (n_users × n_articles) — every cell is now a predicted rating
    reconstructed = svd.inverse_transform(svd.transform(matrix))

    if run_cv:
        # Simple held-out evaluation: mask 10% of known ratings, measure error
        known_mask  = matrix > 0
        n_known     = known_mask.sum()
        sample_size = max(1, int(n_known * 0.10))

        known_positions = list(zip(*np.where(known_mask)))
        sample_pos = [known_positions[i]
                      for i in np.random.choice(
                          len(known_positions), sample_size, replace=False)]

        actual    = [matrix[r, c]    for r, c in sample_pos]
        predicted = [reconstructed[r, c] for r, c in sample_pos]

        rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))
        mae  = np.mean(np.abs(np.array(actual) - np.array(predicted)))
        print(f"  Held-out RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
        print(f"  (evaluated on {sample_size} held-out ratings)")

    print(f"  ✓ SVD model trained  "
          f"(explained variance: "
          f"{svd.explained_variance_ratio_.sum()*100:.1f}%)")

    return {
        "svd":           svd,
        "reconstructed": reconstructed,
        "user_ids":      user_ids,
        "article_ids":   article_ids,
        "user_index":    {u: i for i, u in enumerate(user_ids)},
        "article_index": {a: i for i, a in enumerate(article_ids)},
    }


def get_recommendations(user_id:   str,
                        model:     dict,
                        trainset,              # unused — kept for API compat
                        articles:  pd.DataFrame,
                        interactions: pd.DataFrame,
                        top_n:     int = 10) -> pd.DataFrame:
    """
    Look up predicted ratings for every unseen article and return top-N.
    """
    user_index    = model["user_index"]
    article_ids   = model["article_ids"]
    reconstructed = model["reconstructed"]

    if user_id not in user_index:
        # User has no interactions at all — return empty
        return pd.DataFrame()

    u_idx = user_index[user_id]

    already_seen = set(
        interactions[interactions["user_id"] == user_id]["article_id"]
    )

    predictions = []
    for article_id in article_ids:
        if article_id in already_seen:
            continue
        a_idx = model["article_index"][article_id]
        predictions.append({
            "article_id":       article_id,
            "predicted_rating": round(float(reconstructed[u_idx, a_idx]), 4),
        })

    if not predictions:
        return pd.DataFrame()

    results = (pd.DataFrame(predictions)
               .sort_values("predicted_rating", ascending=False)
               .head(top_n)
               .merge(articles[["article_id", "title",
                                 "category", "tags", "publish_date"]],
                      on="article_id")
               .reset_index(drop=True))
    return results


def save_model(model, trainset=None, out_dir: str = "saved_models"):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/collaborative_model.pkl", "wb") as f:
        pickle.dump({"model": model, "trainset": None}, f)
    print(f"  ✓ Model saved to {out_dir}/collaborative_model.pkl")


def load_model(model_dir: str = "saved_models"):
    with open(f"{model_dir}/collaborative_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["trainset"]


if __name__ == "__main__":
    interactions = pd.read_csv("data/raw/interactions.csv")
    articles     = pd.read_csv("data/raw/articles.csv")

    print("Building ratings matrix...")
    ratings = build_ratings_df(interactions)

    print("\nTraining SVD model...")
    model = train_collaborative_model(ratings, run_cv=True)

    print("\nSample recommendations for U001:")
    print("═" * 55)
    recs = get_recommendations("U001", model, None,
                               articles, interactions, top_n=5)
    print(recs[["title", "category", "predicted_rating"]].to_string())

    print("\nSaving model...")
    save_model(model)