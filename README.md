# 📰 Personalized News Recommendation System

A full-stack machine learning project that recommends news articles based on user reading behavior. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 How It Works

The system uses a **hybrid recommendation engine** combining two ML approaches:

- **Content-Based Filtering** — Represents each article as a TF-IDF vector and matches it against a user's taste profile built from their reading history
- **Collaborative Filtering** — Uses matrix factorization (TruncatedSVD) to find users with similar behavior and surface articles they enjoyed
- **Hybrid Blending** — Adaptively weights both models based on how much history a user has. New users lean on content-based; power users lean on collaborative

```
Raw Data → TF-IDF Features → Content Model ──┐
                                               ├──► Hybrid Score → Top-N Recs
User Interactions → Matrix Factorization ──────┘
```

---

## 🗂️ Project Structure

```
news-recommender/
│
├── data/
│   ├── raw/                    # Simulated articles, users, interactions (CSV)
│   ├── processed/              # TF-IDF matrix, vectorizer, user profiles (pkl)
│   └── simulate_data.py        # Generates realistic fake data
│
├── models/
│   ├── content_based.py        # TF-IDF + cosine similarity recommender
│   ├── collaborative.py        # SVD matrix factorization recommender
│   └── hybrid.py               # Adaptive weighted blend of both models
│
├── notebooks/
│   └── train_model.ipynb       # Google Colab training notebook
│
├── app/
│   └── main.py                 # Streamlit dashboard
│
├── utils/
│   └── preprocessing.py        # Feature engineering pipeline
│
├── saved_models/               # Serialized trained models (.pkl)
├── .streamlit/
│   └── config.toml             # Off-white UI theme
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/news-recommender.git
cd news-recommender
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate simulated data

```bash
python data/simulate_data.py
```

### 5. Build features

```bash
python utils/preprocessing.py
```

### 6. Train the collaborative model

```bash
python models/collaborative.py
```

### 7. Launch the dashboard

```bash
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`

---

## ✨ Features

| Feature                   | Description                                         |
| ------------------------- | --------------------------------------------------- |
| 🎯 Hybrid recommendations | Blends content + collaborative signals              |
| ⚖️ Adaptive weighting     | Blend ratio adjusts based on user history depth     |
| 💡 Explainable results    | Each recommendation shows why it was suggested      |
| 📊 User profile panel     | Visualises inferred interests and model weights     |
| 🔄 Mark as read           | Updates session state and refreshes recommendations |
| ❄️ Cold start handling    | Falls back to recency-ranked articles for new users |

---

## 🛠️ Tech Stack

| Layer         | Tools                                          |
| ------------- | ---------------------------------------------- |
| Data          | pandas, numpy                                  |
| Features      | scikit-learn (TfidfVectorizer)                 |
| Models        | scikit-learn (TruncatedSVD, cosine_similarity) |
| Dashboard     | Streamlit                                      |
| Serialization | pickle                                         |

---

## 📦 requirements.txt

```
numpy
pandas
scikit-learn
streamlit
```

---

## 📖 ML Pipeline

### Content-Based

1. Each article's category, tags, and content are combined into a text blob
2. TF-IDF vectorizer converts all articles into a 500-feature sparse matrix
3. User taste profile = weighted average of TF-IDF vectors of articles they've interacted with (share=3×, like=2×, view=1×)
4. Recommendations = articles with highest cosine similarity to the user's profile

### Collaborative Filtering

1. Interactions are converted to implicit ratings (share=5, like=3.5, view=2)
2. A user-item matrix is built (users × articles)
3. TruncatedSVD decomposes it into 20 latent factors
4. The reconstructed matrix predicts ratings for unseen articles

### Hybrid Blending

- Both models' scores are min-max normalized to [0, 1]
- A weighted blend is computed: `hybrid = content_w × cb_score + collab_w × cf_score`
- Weights adapt linearly: new users (≤5 interactions) → 85/15, power users (≥20) → 30/70

---

## 🙏 Acknowledgements

Built as a learning project to explore recommender systems, feature engineering, and ML-powered UIs from scratch.
