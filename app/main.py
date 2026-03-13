# app/main.py

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

import streamlit as st
import pandas as pd
import numpy as np

from models.hybrid import get_recommendations, explain_recommendation
from models.collaborative import load_model, build_ratings_df

# ── Page config — must be the FIRST streamlit call ────────────────────────────
st.set_page_config(
    page_title="NewsFeed AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Off-white theme via CSS injection ─────────────────────────────────────────
# Streamlit's theming config covers basics; for a polished off-white feel
# we inject a small CSS block. This runs on every rerender (cheap, fine).
st.markdown("""
<style>
  /* ── Base surfaces ── */
  .stApp {
      background-color: #F8F7F4 !important;
  }
  [data-testid="stSidebar"] {
      background-color: #F0EDE8 !important;
  }
  [data-testid="stSidebar"] * {
      color: #1F2937 !important;
  }

  /* ── Force all white blocks to off-white ── */
  [data-testid="stVerticalBlock"],
  [data-testid="stHorizontalBlock"],
  [data-testid="block-container"],
  div.element-container,
  div.stMarkdown {
      background-color: transparent !important;
  }

  /* ── Metric boxes ── */
  [data-testid="stMetric"] {
      background-color: #EEEAE3 !important;
      border-radius: 8px;
      padding: 12px !important;
  }
  [data-testid="stMetricValue"],
  [data-testid="stMetricLabel"] {
      color: #1F2937 !important;
  }

  /* ── Selectbox + slider ── */
  [data-testid="stSelectbox"] > div,
  [data-testid="stSlider"] > div {
      background-color: #F0EDE8 !important;
      border-radius: 8px;
  }

  /* ── Buttons ── */
  .stButton > button {
      background-color: #4338CA !important;
      color: #FFFFFF !important;
      border: none !important;
      border-radius: 6px !important;
      font-size: 0.78rem !important;
      padding: 4px 12px !important;
  }
  .stButton > button:hover {
      background-color: #3730A3 !important;
  }

  /* ── Info / success / warning banners ── */
  [data-testid="stInfo"],
  [data-testid="stSuccess"],
  [data-testid="stWarning"] {
      background-color: #EEEAE3 !important;
      color: #1F2937 !important;
      border-radius: 8px !important;
  }

  /* ── Spinner ── */
  [data-testid="stSpinner"] {
      background-color: transparent !important;
  }

  /* ── Article cards ── */
  .article-card {
      background: #FFFFFF;
      border: 1px solid #E8E4DC;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .article-card:hover {
      box-shadow: 0 4px 12px rgba(0,0,0,0.10);
  }

  /* ── Typography ── */
  h1, h2, h3, h4, p, li, label, span {
      color: #1F2937 !important;
  }
  .explanation { color: #6B7280 !important; font-size: 0.82rem; }
  .badge {
      display: inline-block;
      background: #EEF2FF;
      color: #4338CA !important;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 0.75rem;
      font-weight: 600;
  }
  .score-pill {
      display: inline-block;
      background: #F0FDF4;
      color: #16A34A !important;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 0.75rem;
      font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_articles():
    return pd.read_csv("data/raw/articles.csv")

@st.cache_data
def load_users():
    return pd.read_csv("data/raw/users.csv")

@st.cache_data
def load_interactions():
    return pd.read_csv("data/raw/interactions.csv")

@st.cache_resource          # cache_resource for non-serializable objects (models)
def load_cf_model():
    return load_model("saved_models")


# ── Session state initialisation ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "current_user":    None,
        "viewed_articles": set(),   # articles clicked this session
        "recs_cache":      {},      # user_id → recommendations DataFrame
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar: user selection ────────────────────────────────────────────────────
def render_sidebar(users: pd.DataFrame) -> str:
    """
    Returns the selected user_id.
    Putting selection in the sidebar keeps the main area clean.
    """
    with st.sidebar:
        st.markdown("## 📰 NewsFeed AI")
        st.markdown("---")
        st.markdown("### 👤 Switch User")

        user_ids = users["user_id"].tolist()
        selected = st.selectbox(
            "Select a user profile",
            options=user_ids,
            index=0,
            help="Each user has different reading history and preferences"
        )

        # Show this user's known preferences (from our simulated data)
        user_row = users[users["user_id"] == selected].iloc[0]
        prefs    = user_row["preferred_categories"]

        st.markdown(f"**Preferred topics:**")
        for cat in prefs.split(", "):
            st.markdown(f"- {cat}")

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        top_n = st.slider(
            "Number of recommendations",
            min_value=3, max_value=20, value=10, step=1
        )
        st.session_state["top_n"] = top_n

        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit + scikit-learn + surprise")

    return selected


# ── User profile panel ─────────────────────────────────────────────────────────
def render_profile_panel(user_id: str,
                         recs:    pd.DataFrame,
                         interactions: pd.DataFrame):
    """
    Right-hand panel: shows blend weights and inferred interests.
    Explainability helps users trust the system.
    """
    st.markdown("### 📊 Your Profile")

    # Interaction count and blend weights
    n = len(interactions[interactions["user_id"] == user_id])
    st.metric("Total interactions", n)

    if not recs.empty and "content_weight" in recs.columns:
        cw  = recs.iloc[0]["content_weight"]
        cow = recs.iloc[0]["collab_weight"]

        st.markdown("**Recommendation blend:**")
        col1, col2 = st.columns(2)
        col1.metric("Content-based", f"{int(cw*100)}%")
        col2.metric("Collaborative", f"{int(cow*100)}%")

        if cw >= 0.7:
            st.info("🌱 New user mode — recommendations lean on article content")
        elif cow >= 0.7:
            st.success("🔥 Power user — collaborative filtering driving picks")
        else:
            st.warning("⚖️ Balanced blend of both models")

    # Inferred interests from interactions
    st.markdown("**Inferred interests:**")
    user_history = interactions[interactions["user_id"] == user_id]
    if not user_history.empty:
        cat_counts = (user_history
                      .merge(pd.read_csv("data/raw/articles.csv")
                             [["article_id", "category"]],
                             on="article_id")
                      ["category"].value_counts(normalize=True)
                      .head(5))
        for cat, pct in cat_counts.items():
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            st.markdown(
                f"`{cat:<15}` {bar} {int(pct*100)}%"
            )


# ── Article card renderer ──────────────────────────────────────────────────────
def render_article_card(row: pd.Series, rank: int):
    """
    Renders one article as a styled HTML card.
    We use st.markdown with unsafe_allow_html for custom card styling —
    pure Streamlit widgets can't achieve this level of polish.
    """
    explanation = explain_recommendation(row)
    tags_html   = " ".join(
        f'<span style="background:#F3F4F6;padding:2px 7px;'
        f'border-radius:4px;font-size:0.75rem;margin-right:4px;">'
        f'{t.strip()}</span>'
        for t in str(row.get("tags", "")).split(",")[:3]
    )

    card_html = f"""
    <div class="article-card">
        <span class="badge">{row['category']}</span>
        <span class="score-pill" style="float:right">
            ★ {row['hybrid_score']:.3f}
        </span>
        <div style="clear:both"></div>
        <h4 style="margin:6px 0 8px;color:#111827;font-size:1rem;">
            #{rank} &nbsp; {row['title']}
        </h4>
        <div style="margin-bottom:8px">{tags_html}</div>
        <div class="explanation">💡 {explanation}</div>
        <div style="color:#9CA3AF;font-size:0.75rem;margin-top:8px;">
            📅 {row.get('publish_date','N/A')}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # Simulate "reading" the article
    if st.button(f"Mark as read", key=f"read_{row['article_id']}_{rank}"):
        st.session_state["viewed_articles"].add(row["article_id"])
        # Bust the rec cache so next render picks up the new interaction
        st.session_state["recs_cache"].pop(
            st.session_state["current_user"], None
        )
        st.rerun()


# ── Main recommendations feed ──────────────────────────────────────────────────
def render_recommendations(user_id: str, top_n: int):
    """
    Core feed: loads (or retrieves cached) hybrid recommendations
    and renders each as an article card.
    """
    cache = st.session_state["recs_cache"]

    if user_id not in cache:
        with st.spinner("✨ Generating personalised recommendations..."):
            recs = get_recommendations(user_id, top_n=top_n)
            cache[user_id] = recs
            st.session_state["recs_cache"] = cache
    else:
        recs = cache[user_id]

    return recs


# ── App entry point ────────────────────────────────────────────────────────────
def main():
    init_session_state()

    articles     = load_articles()
    users        = load_users()
    interactions = load_interactions()

    # Sidebar returns the currently selected user
    user_id = render_sidebar(users)

    # Bust cache when user switches
    if st.session_state["current_user"] != user_id:
        st.session_state["current_user"] = user_id
        st.session_state["recs_cache"]   = {}

    top_n = st.session_state.get("top_n", 10)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"## 👋 Welcome back, **{user_id}**")
    st.markdown("*Your personalised news feed, powered by hybrid ML*")
    st.markdown("---")

    # ── Two-column layout ──────────────────────────────────────────────────
    feed_col, profile_col = st.columns([2, 1], gap="large")

    with feed_col:
        st.markdown("### 🔥 For You")
        recs = render_recommendations(user_id, top_n)

        if recs.empty:
            st.warning("No recommendations available yet — "
                       "try a user with more interactions.")
        else:
            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                render_article_card(row, rank)

    with profile_col:
        recs = st.session_state["recs_cache"].get(user_id, pd.DataFrame())
        render_profile_panel(user_id, recs, interactions)


if __name__ == "__main__":
    main()