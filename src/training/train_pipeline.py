from pathlib import Path

import mlflow
import numpy as np

from src.features.game_features import (
    build_feature_matrix, compute_popularity_score, compute_richness_score,
)
from src.models.content_based import build_content_model
from src.models.hybrid import HybridRecommender
from src.models.lambdarank import LambdaRankRecommender
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k,
    map_at_k, hit_rate_at_k, coverage, diversity,
)

PROCESSED_DIR = Path("data/processed")

TEST_PROFILES = [
    {"name": "Casual", "tags": ["casual", "puzzle", "family", "party", "rhythm"]},
    {"name": "Hardcore", "tags": ["action", "challenging", "souls_like", "rpg", "shooter"]},
    {"name": "Story", "tags": ["story_rich", "narrative", "adventure", "atmospheric", "single_player"]},
    {"name": "Strategy", "tags": ["strategy", "tactical", "simulation", "management", "turn_based"]},
    {"name": "Retro", "tags": ["retro", "arcade", "platformer", "pixel_art", "2d_flat"]},
]


def _evaluate_profile(rec, tags: list[str], k: int = 20) -> dict:
    recs = rec.recommend_for_profile(tags, k=k)
    game_ids = [r["game_id"] for r in recs]
    return game_ids


def train_hybrid(alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1,
                 log_mlflow: bool = True, experiment_name: str = "playfit-hybrid") -> dict:
    if log_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=f"alpha={alpha}_beta={beta}_gamma={gamma}")
        mlflow.log_params({"alpha": alpha, "beta": beta, "gamma": gamma})

    fm = compute_richness_score(compute_popularity_score(build_feature_matrix()))
    cm = build_content_model(fm, n_components=100)
    rec = HybridRecommender(alpha=alpha, beta=beta, gamma=gamma)
    rec.fit(fm, cm)

    import polars as pl

    all_recs, all_rel = [], []
    for prof in TEST_PROFILES:
        recs = _evaluate_profile(rec, prof["tags"])
        all_recs.append(recs)
        tag_cols = [c for c in prof["tags"] if c in fm.columns]
        relevant = set()
        for row in fm.to_dicts():
            if any(row.get(t, 0) == 1 for t in tag_cols):
                relevant.add(row["game_id"])
        all_rel.append(relevant)

    metrics = {}
    for k in [1, 3, 5, 10, 20]:
        precs = [precision_at_k(r, rel, k) for r, rel in zip(all_recs, all_rel)]
        recs_m = [recall_at_k(r, rel, k) for r, rel in zip(all_recs, all_rel)]
        ndcgs = [ndcg_at_k(r, rel, k) for r, rel in zip(all_recs, all_rel)]
        metrics[f"precision_at_{k}"] = float(np.mean(precs))
        metrics[f"recall_at_{k}"] = float(np.mean(recs_m))
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcgs))

    metrics["map_at_20"] = float(map_at_k(all_recs, all_rel, 20))
    metrics["hit_rate_at_20"] = float(hit_rate_at_k(all_recs, all_rel, 20))
    metrics["coverage"] = float(coverage(all_recs, len(fm)))

    popularity_map = dict(zip(fm["game_id"].to_list(), fm["popularity_score"].to_list()))
    import math
    nov_scores = []
    for recs in all_recs:
        for g in recs:
            pop = popularity_map.get(g, 0.0)
            if pop > 0:
                nov_scores.append(-math.log2(pop))
    metrics["novelty"] = float(np.mean(nov_scores)) if nov_scores else 0.0

    if log_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_components", 100)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ks = [1, 3, 5, 10, 20]
        for label in ["precision", "recall", "ndcg"]:
            vals = [metrics[f"{label}_at_{k}"] for k in ks]
            ax.plot(ks, vals, marker="o", label=label)
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        ax.set_title(f"Evaluation Metrics (α={alpha}, β={beta}, γ={gamma})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "metrics_vs_k.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ["Prec@5", "Recall@5", "NDCG@5", "MAP@20", "Coverage"]
        vals = [metrics["precision_at_5"], metrics["recall_at_5"],
                metrics["ndcg_at_5"], metrics["map_at_20"], metrics["coverage"]]
        ax.bar(bars, vals, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"])
        ax.set_ylim(0, 1)
        ax.set_title("Key Metrics")
        mlflow.log_figure(fig, "key_metrics.png")
        plt.close()

        mlflow.end_run()

    return {"metrics": metrics, "model": rec}


def train_lambdarank(log_mlflow: bool = True,
                     experiment_name: str = "playfit-lambdarank") -> dict:
    if log_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name="lambdarank_default")

    lr = LambdaRankRecommender()
    lr.fit()
    ndcg_results = lr.evaluate_ndcg()

    if log_mlflow:
        mlflow.log_params(lr.params)
        mlflow.log_metrics({f"val_{k}": v for k, v in ndcg_results.items()})
        mlflow.end_run()

    return {"ndcg": ndcg_results, "model": lr}
