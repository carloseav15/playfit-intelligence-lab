"""
demo_loader.py — Carga el recomendador en modo demo (sin DB) para deploy en Render.

Uso:
    from src.models.demo_loader import load_demo_recommender
    rec = load_demo_recommender()
    rec.recommend(["zelda"], k=5)
"""

from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEMO_DIR = Path("data/demo")


def load_demo_recommender():
    """Load a HybridRecommender-like object from pre-computed demo data."""
    from src.models.hybrid import HybridRecommender

    reduced = np.load(str(DEMO_DIR / "reduced.npy"))
    game_ids = np.load(str(DEMO_DIR / "game_ids.npy")).tolist()
    titles = np.load(str(DEMO_DIR / "titles.npy"), allow_pickle=True).tolist()
    pop_scores = np.load(str(DEMO_DIR / "popularity_scores.npy"))
    conf_scores = np.load(str(DEMO_DIR / "confidence_scores.npy"))

    rec = HybridRecommender(alpha=0.7, beta=0.2, gamma=0.1)

    # Build minimal feature matrix
    import polars as pl
    fm = pl.DataFrame({
        "game_id": game_ids,
        "title": [str(t) if not isinstance(t, str) else t for t in titles],
        "popularity_score": pop_scores.tolist(),
        "data_confidence_score": conf_scores.tolist(),
        "richness_score": [1.0] * len(game_ids),
        "release_year": [0] * len(game_ids),
        "tags": [[] for _ in game_ids],
        "platforms": [[] for _ in game_ids],
        "genre_id": [""] * len(game_ids),
        "cover_url": [""] * len(game_ids),
    })

    rec.feature_matrix = fm
    rec.game_ids = game_ids
    rec.confidence_scores = conf_scores
    rec.popularity_scores = pop_scores
    rec.content_model = {"reduced": reduced, "game_ids": game_ids}
    rec.game_id_to_idx = {gid: i for i, gid in enumerate(game_ids)}

    return rec


def search_games_in_demo(query: str, rec, top_n: int = 10) -> list[dict]:
    """Simple title search within demo dataset."""
    q = query.lower()
    results = []
    for gid in rec.game_ids:
        idx = rec.game_id_to_idx[gid]
        title = str(rec.feature_matrix["title"][int(idx)])
        if q in title.lower():
            results.append({
                "game_id": gid,
                "title": title,
                "popularity_score": float(rec.popularity_scores[int(idx)]),
                "data_confidence_score": int(rec.confidence_scores[int(idx)]),
            })
    return sorted(results, key=lambda x: x["popularity_score"], reverse=True)[:top_n]
