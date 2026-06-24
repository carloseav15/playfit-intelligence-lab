from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

from src.features.game_features import (
    build_feature_matrix, compute_popularity_score, compute_richness_score,
)
from src.models.content_based import build_content_model, METADATA_COLS

PROCESSED_DIR = Path("data/processed")


class LambdaRankRecommender:
    def __init__(self, **kwargs):
        default_params = {
            "objective": "lambdarank",
            "boosting_type": "gbdt",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        self.params = default_params
        self.model: LGBMRanker | None = None
        self.feature_names: list[str] | None = None
        self.game_ids: list[str] | None = None
        self.game_id_to_idx: dict[str, int] | None = None
        self.feature_matrix: pl.DataFrame | None = None

    def _prepare_data(self, feature_matrix: pl.DataFrame) -> tuple:
        keep_cols = METADATA_COLS - {"popularity_score", "richness_score"}
        content_cols = [c for c in feature_matrix.columns if c not in keep_cols]
        non_feature = {"game_id", "title", "release_year", "genre_id"}
        feature_cols = [c for c in content_cols if c not in non_feature]

        X = feature_matrix.select(feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        popularity = feature_matrix["popularity_score"].to_numpy()
        pop_min, pop_max = popularity.min(), popularity.max()
        if pop_max > pop_min:
            y = np.floor(4 * (popularity - pop_min) / (pop_max - pop_min)).astype(int)
        else:
            y = np.zeros(len(popularity), dtype=int)

        platform_prefixes = (
            "ps", "nintendo", "xbox", "pc", "sega", "atari", "arcade",
            "switch", "gamecube", "dreamcast", "saturn", "genesis",
            "snes", "nes", "gb", "gbc", "gba", "n64", "ds", "3ds",
            "psp", "ps_vita", "wii", "wii_u", "neo_geo", "game_gear",
            "macos", "linux", "android", "ios",
        )
        platform_cols = [c for c in feature_cols
                         if c.startswith(platform_prefixes)]
        platform_ids = feature_matrix.select(platform_cols).to_numpy()
        query_groups = np.where(platform_ids.sum(axis=1) > 0, platform_ids.argmax(axis=1), 0)
        unique_groups = np.unique(query_groups)
        group_sizes = []
        for g in unique_groups:
            mask = query_groups == g
            group_sizes.append(mask.sum())
        group = np.array(group_sizes, dtype=int)
        sort_idx = np.argsort(query_groups)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        return X_sorted, y_sorted, group, feature_cols, sort_idx

    def fit(self, feature_matrix: pl.DataFrame | None = None):
        if feature_matrix is None:
            feature_matrix = pl.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
        self.feature_matrix = feature_matrix
        self.game_ids = feature_matrix["game_id"].to_list()
        self.game_id_to_idx = {gid: i for i, gid in enumerate(self.game_ids)}

        X, y, group, feature_cols, sort_idx = self._prepare_data(feature_matrix)
        self.feature_names = feature_cols
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        cumsum = np.cumsum(group)
        split_point = np.searchsorted(cumsum, split, side="right")
        group_train = group[:split_point]
        group_val = group[split_point:]

        self.model = LGBMRanker(**self.params)
        self.model.fit(
            X=X_train, y=y_train, group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_at=[1, 3, 5, 10],
            eval_metric="ndcg",
            callbacks=[
                __import__("lightgbm").early_stopping(stopping_rounds=20, verbose=False),
                __import__("lightgbm").log_evaluation(period=50),
            ],
        )
        return self

    def recommend(self, liked_game_ids: list[str] | None = None,
                  k: int = 20) -> list[dict]:
        if self.model is None or self.feature_matrix is None:
            raise ValueError("Model not fitted")

        X, _, _, feature_cols, _ = self._prepare_data(self.feature_matrix)
        scores = self.model.predict(X)
        game_ids = self.feature_matrix["game_id"].to_list()
        titles = self.feature_matrix["title"].to_list()
        confidence = self.feature_matrix["data_confidence_score"].to_numpy()
        popularity = self.feature_matrix["popularity_score"].to_numpy()

        excluded = set(liked_game_ids or [])
        candidates = [
            (scores[i], game_ids[i], titles[i], confidence[i], popularity[i])
            for i in range(len(game_ids))
            if game_ids[i] not in excluded
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:k]

        return [
            {
                "game_id": gid,
                "title": title,
                "final_score": float(score),
                "lambdarank_score": float(score),
                "popularity_score": float(pop),
                "data_confidence": int(conf),
            }
            for score, gid, title, conf, pop in top
        ]

    def evaluate_ndcg(self, feature_matrix: pl.DataFrame | None = None) -> dict:
        if feature_matrix is None:
            feature_matrix = self.feature_matrix
        X, y, group, _, _ = self._prepare_data(feature_matrix)
        scores = self.model.predict(X)
        results = {}
        for k in [1, 3, 5, 10, 20]:
            ndcg_scores = []
            idx = 0
            for g_size in group:
                if idx + g_size <= len(scores):
                    y_q = y[idx:idx + g_size].reshape(1, -1)
                    s_q = scores[idx:idx + g_size].reshape(1, -1)
                    ndcg_scores.append(ndcg_score(y_q, s_q, k=k))
                idx += g_size
            results[f"ndcg@{k}"] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        return results
