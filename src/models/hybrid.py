import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

from src.models.content_based import METADATA_COLS


class HybridRecommender:
    def __init__(self, alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.feature_matrix: pl.DataFrame | None = None
        self.popularity_scores: np.ndarray | None = None
        self.confidence_scores: np.ndarray | None = None
        self.content_model: dict | None = None

    def fit(self, feature_matrix: pl.DataFrame, content_model: dict):
        self.feature_matrix = feature_matrix
        self.content_model = content_model
        self.popularity_scores = feature_matrix["popularity_score"].to_numpy()
        self.confidence_scores = feature_matrix["data_confidence_score"].to_numpy()
        self.game_ids = feature_matrix["game_id"].to_list()
        self.game_id_to_idx = {gid: i for i, gid in enumerate(self.game_ids)}

    def recommend(self, user_liked_games: list[str],
                  excluded_games: list[str] | None = None,
                  k: int = 20) -> list[dict]:
        if self.feature_matrix is None or self.content_model is None:
            raise ValueError("Model not fitted")

        excluded = set(excluded_games or [])
        liked_indices = [
            self.game_id_to_idx[gid]
            for gid in user_liked_games
            if gid in self.game_id_to_idx
        ]

        if not liked_indices:
            return self._cold_start_recommend(excluded, k)

        content_scores = self._content_scores_for_all(liked_indices)
        pop_scores = self.popularity_scores
        conf_penalty = (100 - self.confidence_scores) / 100 * self.gamma

        final_scores = (
            self.alpha * content_scores
            + self.beta * pop_scores
            - conf_penalty
        )

        candidate_indices = [
            i for i, gid in enumerate(self.game_ids)
            if gid not in excluded and gid not in user_liked_games
        ]

        top_indices = sorted(
            candidate_indices,
            key=lambda i: final_scores[i],
            reverse=True
        )[:k]

        return [
            {
                "game_id": self.game_ids[i],
                "title": str(self.feature_matrix["title"][int(i)]),
                "final_score": float(final_scores[i]),
                "content_score": float(content_scores[i]),
                "popularity_score": float(pop_scores[i]),
                "confidence_penalty": float(conf_penalty[i]),
                "data_confidence": int(self.confidence_scores[i]),
            }
            for i in top_indices
        ]

    def _content_scores_for_all(self, liked_indices: list[int]) -> np.ndarray:
        reduced = self.content_model["reduced"]
        query = reduced[liked_indices].mean(axis=0, keepdims=True)
        sims = cosine_similarity(query, reduced)[0]
        return np.nan_to_num(sims, nan=0.0)

    def _cold_start_recommend(self, excluded: set, k: int) -> list[dict]:
        candidate_indices = [
            i for i, gid in enumerate(self.game_ids)
            if gid not in excluded
        ]
        top_indices = sorted(
            candidate_indices,
            key=lambda i: self.popularity_scores[i],
            reverse=True
        )[:k]

        return [
            {
                "game_id": self.game_ids[i],
                "title": str(self.feature_matrix["title"][int(i)]),
                "final_score": float(self.popularity_scores[i]),
                "content_score": 0.0,
                "popularity_score": float(self.popularity_scores[i]),
                "confidence_penalty": 0.0,
                "data_confidence": int(self.confidence_scores[i]),
            }
            for i in top_indices
        ]

    def recommend_for_profile(self, profile_tags: list[str],
                              k: int = 20) -> list[dict]:
        tag_cols = [c for c in self.feature_matrix.columns
                    if c in profile_tags and c not in METADATA_COLS]
        if not tag_cols:
            return self._cold_start_recommend(set(), k)

        profile_vector = np.zeros(len(self.feature_matrix))
        for col in tag_cols:
            profile_vector += self.feature_matrix[col].to_numpy()
        profile_scores = profile_vector / max(len(tag_cols), 1)

        pop_scores = self.popularity_scores
        conf_penalty = (100 - self.confidence_scores) / 100 * self.gamma

        final_scores = (
            self.alpha * profile_scores
            + self.beta * pop_scores
            - conf_penalty
        )

        top_indices = np.argsort(final_scores)[::-1][:k]
        return [
            {
                "game_id": self.game_ids[i],
                "title": str(self.feature_matrix["title"][int(i)]),
                "final_score": float(final_scores[i]),
                "content_score": float(profile_scores[i]),
                "popularity_score": float(pop_scores[i]),
                "data_confidence": int(self.confidence_scores[i]),
            }
            for i in top_indices
        ]

    def get_similarity_between(self, idx_a: int, idx_b: int) -> float:
        reduced = self.content_model["reduced"]
        sim = cosine_similarity(
            reduced[int(idx_a)].reshape(1, -1),
            reduced[int(idx_b)].reshape(1, -1),
        )[0, 0]
        return float(sim)

    def rerank_with_mmr(self, recommendations: list[dict],
                        lambda_param: float = 0.5, k: int = 20) -> list[dict]:
        if not recommendations:
            return recommendations

        selected = []
        candidates = list(recommendations)
        k = min(k, len(candidates))

        for _ in range(k):
            best_idx = -1
            best_score = -float("inf")

            for i, cand in enumerate(candidates):
                relevance = cand["final_score"]
                if not selected:
                    mmr_score = relevance
                else:
                    max_sim = max(
                        self.get_similarity_between(
                            self.game_id_to_idx[cand["game_id"]],
                            self.game_id_to_idx[sel["game_id"]],
                        )
                        for sel in selected
                        if cand["game_id"] in self.game_id_to_idx
                        and sel["game_id"] in self.game_id_to_idx
                    )
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))

        return selected
