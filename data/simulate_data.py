# data/simulate_data.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
# Setting a seed means every time you run this, you get the SAME fake data.
# This is critical for debugging — you don't want your data changing under you.
random.seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_ARTICLES = 200
NUM_USERS    = 50
NUM_INTERACTIONS = 2000

CATEGORIES = ["Technology", "Sports", "Politics", "Health", "Science",
              "Entertainment", "Business", "World"]

# Tags give the content-based model richer signal than category alone.
TAGS_BY_CATEGORY = {
    "Technology":    ["AI", "machine learning", "startups", "cybersecurity",
                      "smartphones", "cloud computing", "robotics"],
    "Sports":        ["football", "basketball", "tennis", "Olympics",
                      "soccer", "cricket", "athletics"],
    "Politics":      ["elections", "policy", "government", "diplomacy",
                      "legislation", "democracy", "geopolitics"],
    "Health":        ["mental health", "nutrition", "fitness", "medicine",
                      "wellness", "vaccines", "research"],
    "Science":       ["space", "climate", "biology", "physics",
                      "environment", "discoveries", "research"],
    "Entertainment": ["movies", "music", "celebrities", "streaming",
                      "gaming", "awards", "culture"],
    "Business":      ["stocks", "economy", "startups", "finance",
                      "markets", "entrepreneurship", "trade"],
    "World":         ["Asia", "Europe", "Africa", "conflict",
                      "international", "diplomacy", "migration"],
}

# Weighted action types — share is rarer than view (realistic!)
ACTIONS        = ["view", "like", "share"]
ACTION_WEIGHTS = [0.70, 0.20, 0.10]

# ── Article Generator ─────────────────────────────────────────────────────────
def generate_articles(n: int) -> pd.DataFrame:
    """
    Creates n fake articles. Each gets a category, 2-4 relevant tags,
    a fake title, a short fake 'content' snippet, and a publish date
    within the last 90 days.
    """
    articles = []
    base_date = datetime.now()

    for i in range(n):
        category = random.choice(CATEGORIES)
        tags     = random.sample(TAGS_BY_CATEGORY[category],
                                 k=random.randint(2, 4))

        article = {
            "article_id":    f"A{i+1:04d}",           # e.g. A0001
            "title":         f"{category} Update #{i+1}: "
                             f"{' & '.join(tags[:2])}",
            "category":      category,
            "tags":          ", ".join(tags),          # stored as a string
            "content":       (                         # fake "body text"
                f"This article covers {', '.join(tags)} in the context "
                f"of {category.lower()}. Latest developments show that "
                f"{tags[0]} continues to shape the {category.lower()} "
                f"landscape significantly."
            ),
            "publish_date":  (
                base_date - timedelta(days=random.randint(0, 90))
            ).strftime("%Y-%m-%d"),
        }
        articles.append(article)

    return pd.DataFrame(articles)


# ── User Generator ─────────────────────────────────────────────────────────────
def generate_users(n: int) -> pd.DataFrame:
    """
    Creates n users. Each user has 1-2 preferred categories.
    This preference is what will create learnable signal in the interactions.
    """
    users = []
    for i in range(n):
        preferred = random.sample(CATEGORIES, k=random.randint(1, 2))
        users.append({
            "user_id":             f"U{i+1:03d}",     # e.g. U001
            "preferred_categories": ", ".join(preferred),
        })
    return pd.DataFrame(users)


# ── Interaction Simulator ──────────────────────────────────────────────────────
def simulate_interactions(users_df:    pd.DataFrame,
                           articles_df: pd.DataFrame,
                           n:           int) -> pd.DataFrame:
    """
    Simulates n interactions.

    KEY DESIGN DECISION:
    Users are 80% likely to interact with articles matching their preferences,
    and 20% likely to interact with random articles (exploration noise).
    Without that 80/20 split, a recommendation model has nothing to learn.
    With it, the model can discover: "U001 keeps reading Tech → show them Tech."
    """
    interactions = []
    base_time    = datetime.now()

    for _ in range(n):
        user    = users_df.sample(1).iloc[0]
        user_id = user["user_id"]
        prefs   = user["preferred_categories"].split(", ")

        # 80% chance: pick an article the user would actually like
        if random.random() < 0.80:
            preferred_articles = articles_df[
                articles_df["category"].isin(prefs)
            ]
            # Fallback: if no preferred articles found, go random
            article = (preferred_articles if len(preferred_articles) > 0
                       else articles_df).sample(1).iloc[0]
        else:
            # 20% exploration — keeps data realistic, avoids filter bubbles
            article = articles_df.sample(1).iloc[0]

        action = random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0]

        interactions.append({
            "user_id":              user_id,
            "article_id":           article["article_id"],
            "action":               action,
            "timestamp":            (
                base_time - timedelta(
                    minutes=random.randint(0, 60 * 24 * 30)  # last 30 days
                )
            ).strftime("%Y-%m-%d %H:%M:%S"),
            # Shares → longer reading time (they read carefully before sharing)
            "reading_time_seconds": (
                random.randint(120, 300) if action == "share"
                else random.randint(10, 180)
            ),
        })

    return pd.DataFrame(interactions)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    print("Generating articles...")
    articles = generate_articles(NUM_ARTICLES)
    articles.to_csv("data/raw/articles.csv", index=False)
    print(f"  ✓ {len(articles)} articles saved to data/raw/articles.csv")

    print("Generating users...")
    users = generate_users(NUM_USERS)
    users.to_csv("data/raw/users.csv", index=False)
    print(f"  ✓ {len(users)} users saved to data/raw/users.csv")

    print("Simulating interactions...")
    interactions = simulate_interactions(users, articles, NUM_INTERACTIONS)
    interactions.to_csv("data/raw/interactions.csv", index=False)
    print(f"  ✓ {len(interactions)} interactions saved to data/raw/interactions.csv")

    # ── Quick Sanity Check ─────────────────────────────────────────────────
    print("\n── Sanity Check ──────────────────────────────────────────")
    print("Articles sample:")
    print(articles[["article_id","title","category"]].head(3).to_string())
    print("\nInteractions distribution:")
    print(interactions["action"].value_counts().to_string())
    print("\nTop 5 most-read articles:")
    print(interactions["article_id"].value_counts().head().to_string())