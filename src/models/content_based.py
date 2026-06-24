from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data/processed")

METADATA_COLS = {
    "game_id", "title", "release_year", "genre_id",
    "best_critic_score", "best_user_score",
    "critic_review_count", "user_review_count",
    "critic_positive_ratio", "user_positive_ratio",
    "max_global_sales_millions", "total_global_sales_millions",
    "data_confidence_score",
    "has_external_id", "has_company", "has_age_rating",
    "has_summary", "has_sales", "has_review_sentiment",
    "log_sales", "total_review_count",
    "popularity_score", "richness_score",
}


def build_content_model(feature_matrix: pl.DataFrame | None = None,
                        n_components: int = 100) -> dict:
    if feature_matrix is None:
        feature_matrix = pl.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")

    content_cols = [c for c in feature_matrix.columns if c not in METADATA_COLS]
    content_matrix = feature_matrix.select(content_cols).to_numpy()
    content_matrix = np.nan_to_num(content_matrix, nan=0.0)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(content_matrix)

    nn = NearestNeighbors(n_neighbors=21, metric="cosine")
    nn.fit(reduced)

    return {
        "svd": svd,
        "nn": nn,
        "reduced": reduced,
        "game_ids": feature_matrix["game_id"].to_list(),
        "titles": feature_matrix["title"].to_list(),
    }


def content_recommend(game_idx: int, model: dict, k: int = 20) -> list[dict]:
    distances, indices = model["nn"].kneighbors(
        model["reduced"][game_idx].reshape(1, -1),
        n_neighbors=k + 1,
    )
    similarities = 1 - distances.flatten()
    top_indices = indices.flatten()[1:k+1]
    top_similarities = similarities[1:k+1]

    return [
        {
            "game_id": model["game_ids"][i],
            "title": model["titles"][i],
            "content_score": float(top_similarities[idx]),
        }
        for idx, i in enumerate(top_indices)
    ]


def content_similarity_between(model: dict, idx_a: int, idx_b: int) -> float:
    vec_a = model["reduced"][idx_a].reshape(1, -1)
    vec_b = model["reduced"][idx_b].reshape(1, -1)
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(vec_a, vec_b)[0, 0]
    return float(sim)
