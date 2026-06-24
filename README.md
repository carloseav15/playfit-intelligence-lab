# Playfit Intelligence Lab 🎮

**A Data Science Portfolio Project — Catalog Quality Observatory & Hybrid Recommendation Engine**

> Built with real production data from [Playfit](https://github.com/carancibia/games-library) (63,682 games, 164K+ game-tags, 42K+ scores, 35K+ external match candidates).

## 🎯 What This Project Demonstrates

| Skill | Evidence |
|---|---|
| **Data Quality** | Coverage analysis, confidence scoring, duplicate detection, external matching quality |
| **Feature Engineering** | One-hot encoding of tags/platforms, feature selection, dimensionality reduction (SVD) |
| **Classical ML** | Content-based similarity, hybrid ranking, hyperparameter optimization |
| **Recommendation Systems** | Cold-start handling, popularity priors, MMR re-ranking, LambdaRank |
| **MLOps** | MLflow experiment tracking, grid search, A/B test simulation |
| **LLM/GenAI** | LLM explainability with GPT-4o-mini + MLflow genai evaluation |
| **Data Product** | Streamlit app with 3 tabs (Catalog Health, Recommender, Evaluation) |

## 🏗️ Architecture

```
playfit-intelligence-lab/
├── notebooks/                  # 6 portfolio notebooks
│   ├── 01_catalog_quality_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_recommender_model.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_mlflow_abtest_llm.ipynb
│   └── 06_lambdarank_duckdb.ipynb
├── src/                        # Reusable Python modules
│   ├── data/
│   │   ├── extract.py          # Supabase → Parquet export
│   │   └── analytics_duckdb.py # DuckDB OLAP queries
│   ├── features/
│   │   ├── catalog_features.py # Data quality analysis
│   │   └── game_features.py    # Feature matrix construction
│   ├── models/
│   │   ├── content_based.py    # Content-based similarity (SVD + NN)
│   │   ├── hybrid.py           # Hybrid recommender + MMR re-ranking
│   │   └── lambdarank.py       # LightGBM LambdaRank ranking model
│   ├── training/
│   │   ├── train_pipeline.py   # MLflow-wrapped training pipeline
│   │   └── experiment_grid.py  # Grid search over α/β/γ
│   └── evaluation/
│       ├── metrics.py          # precision@k, NDCG, coverage, novelty, diversity
│       ├── explainability.py   # Rule-based explanation generation
│       ├── llm_explainability.py # LLM-based narrative explanations
│       └── ab_test.py          # A/B test simulator with bootstrap CI
├── app/
│   └── streamlit_app.py        # 3-tab portfolio app (with MMR + LLM options)
├── docs/
│   ├── model_card.md           # Google-style Model Card
│   └── datasheet.md            # Datasheet for Dataset
├── Dockerfile                  # Multi-stage container image
├── render.yaml                 # Render.com deploy config
├── data/                       # Data (Parquet, gitignored)
│   ├── raw/
│   └── processed/
├── models/artifacts/
├── reports/figures/
├── scripts/
│   ├── generate_notebooks.py
│   └── generate_notebooks_extra.py
├── .streamlit/config.toml
├── .env
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Export data from Supabase (requires local Supabase instance)
python3 src/data/extract.py

# 3. Run notebooks
jupyter notebook notebooks/

# 4. Launch the app
streamlit run app/streamlit_app.py

# 5. MLflow UI (after running experiments)
mlflow ui
# Opens http://localhost:5000

# 6. Docker
docker build -t playfit-intelligence-lab .
docker run -p 8501:8501 playfit-intelligence-lab
```

## 🧪 Additional Models & Techniques

### LambdaRank (LightGBM)
Gradient-boosted ranking model with `objective="lambdarank"`. Trained on popularity-derived relevance labels with per-platform query groups. Alternative to the hybrid model, evaluated with NDCG.

### MMR Re-Ranking
`MMR = λ × relevance - (1-λ) × max_similarity_to_selected`

Post-processing step that trades relevance for diversity. Available as toggle in the Streamlit app.

### A/B Test Simulator
Bootstrap-based statistical test comparing two model variants. Reports: lift, 95% CI, p-value, win distribution. Runs on synthetic user profiles.

### LLM Explainability
Narrative game recommendations via GPT-4o-mini (OpenRouter). Evaluated against rule-based baseline with MLflow genai scorers (Correctness, Fluency).

## 📊 The Recommendation Model

`final_score = α × content_similarity + β × popularity_prior - γ × confidence_penalty`

Where:
- **Content similarity** (α): Cosine similarity on SVD-100 embedding of tags + platforms
- **Popularity prior** (β): Composite score from critic/user scores, sales, and sentiment
- **Confidence penalty** (γ): Penalization for low `data_confidence_score`
- **Cold-start**: Falls back to global popularity ranking when no user history exists

## 📈 Key Findings

- **63,682** games in catalog
- **60.6%** without genre → 28,842 games need genre assignment
- **51%** without cover → 23,366 games missing cover art
- **~3.5%** without release year
- **914** duplicate groups detected (1,162 candidate games)
- **35,253** external match candidates (Metacritic + VGSales)
- **High confidence** data: ~55% of catalog (score ≥ 70/100)

## 🛠️ Stack (2026)

| Category | Tools |
|---|---|
| **Core** | Python 3.12, SQL, Git, Docker |
| **Data Wrangling** | Polars 1.41, Pandas 3.0, DuckDB 1.5 |
| **ML** | scikit-learn 1.9, LightGBM 4.6 (LambdaRank), NearestNeighbors, SVD |
| **MLOps** | MLflow 3.14 (tracking + genai evaluation) |
| **LLM/GenAI** | OpenAI API, OpenRouter, GPT-4o-mini |
| **Visualization** | Matplotlib, Seaborn, Streamlit 1.58 |
| **Database** | Supabase Postgres (local, 27 tables) |
| **Deploy** | Docker, Render.com |

## 📝 License

MIT
