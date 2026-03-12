# models/collaborative.py

import pickle
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import os


# ── Action → synthetic rating ──────────────────────────────────────────────────
# surprise expects ratings in a fixed scale. We map our actions to 1–5
# so the library is happy, while preserving relative signal strength.
ACTION_TO_RATING = {
    "share": 5.0,
    "like":  3.5,
    "view":  2.0,
}
RATING_SCALE = (1, 5)


def build_ratings_df(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw interactions into a ratings DataFrame.

    A user may have multiple interactions with the same article
    (e.g. viewed then liked). We take the MAX rating per pair —
    the strongest signal wins.

    Returns DataFrame with columns: user_id, article_id, rating
    """
    interactions = interactions.copy()
    interactions["rating"] = interactions["action"].map(ACTION_TO_RATING)

    # Aggregate: one row per (user, article) pair
    ratings = (interactions
               .groupby(["user_id", "article_id"])["rating"]
               .max()
               .reset_index())

    print(f"  Ratings matrix: {ratings['user_id'].nunique()} users × "
          f"{ratings['article_id'].nunique()} articles "
          f"({len(ratings)} observed ratings)")
    return ratings


def train_collaborative_model(ratings: pd.DataFrame,
                              n_factors:  int = 20,
                              n_epochs:   int = 30,
                              run_cv:     bool = True):
    """
    Train an SVD model (matrix factorization) using the surprise library.

    Args:
        ratings    : DataFrame from build_ratings_df()
        n_factors  : number of latent factors — the hidden "concepts"
                     the model discovers. More = expressive but slower.
                     20 is a good starting point for small datasets.
        n_epochs   : training iterations. More = better fit, but
                     watch for overfitting on small datasets.
        run_cv     : if True, print cross-validation RMSE so you can
                     gauge model quality before serving

    Returns:
        trained SVD model, fitted on the full dataset
    """
    # surprise needs a special Dataset object
    reader  = Reader(rating_scale=RATING_SCALE)
    dataset = Dataset.load_from_df(
        ratings[["user_id", "article_id", "rating"]],
        reader
    )

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        biased=True,     # learns global average + per-user/item bias
                         # e.g. some articles are universally popular
        random_state=42,
    )

    if run_cv:
        print("  Running 3-fold cross-validation...")
        cv_results = cross_validate(
            model, dataset, measures=["RMSE", "MAE"],
            cv=3, verbose=False
        )
        mean_rmse = cv_results["test_rmse"].mean()
        mean_mae  = cv_results["test_mae"].mean()
        print(f"  CV RMSE: {mean_rmse:.4f}  |  MAE: {mean_mae:.4f}")
        # RMSE is "average prediction error" in rating units (1–5 scale).
        # RMSE < 1.0 on this scale is generally good for a small dataset.

    # Train final model on ALL data (no held-out set)
    # For production you'd keep a test split — we'll discuss this in eval
    trainset = dataset.build_full_trainset()
    model.fit(trainset)
    print("  ✓ Model trained on full dataset")

    return model, trainset


def get_recommendations(user_id:   str,
                        model,
                        trainset,
                        articles:  pd.DataFrame,
                        interactions: pd.DataFrame,
                        top_n:     int = 10) -> pd.DataFrame:
    """
    Generate collaborative filtering recommendations for one user.

    For every article the user HASN'T rated, ask the model:
    "What rating would this user give this article?"
    Then return the top-N predicted ratings.
    """
    # Articles this user has already interacted with
    already_seen = set(
        interactions[interactions["user_id"] == user_id]["article_id"]
    )

    # All article IDs in our dataset
    all_article_ids = articles["article_id"].tolist()

    # Predict rating for every unseen article
    predictions = []
    for article_id in all_article_ids:
        if article_id in already_seen:
            continue

        # surprise's predict() takes raw user/item IDs
        pred = model.predict(uid=user_id, iid=article_id)
        predictions.append({
            "article_id":       article_id,
            "predicted_rating": round(pred.est, 4),
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


def save_model(model, trainset,
               out_dir: str = "saved_models"):
    """Persist the trained model for use in the Streamlit app."""
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/collaborative_model.pkl", "wb") as f:
        pickle.dump({"model": model, "trainset": trainset}, f)
    print(f"  ✓ Model saved to {out_dir}/collaborative_model.pkl")


def load_model(model_dir: str = "saved_models"):
    with open(f"{model_dir}/collaborative_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["trainset"]


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    interactions = pd.read_csv("data/raw/interactions.csv")
    articles     = pd.read_csv("data/raw/articles.csv")

    print("\nBuilding ratings matrix...")
    ratings = build_ratings_df(interactions)

    print("\nTraining SVD model...")
    model, trainset = train_collaborative_model(ratings, run_cv=True)

    print("\nSample recommendations for U001:")
    print("═" * 55)
    recs = get_recommendations("U001", model, trainset,
                               articles, interactions, top_n=5)
    print(recs[["title", "category", "predicted_rating"]].to_string())

    print("\nSaving model...")
    save_model(model, trainset)