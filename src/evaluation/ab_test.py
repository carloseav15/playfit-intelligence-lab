import numpy as np

from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import ndcg_at_k

TEST_PROFILES = [
    {"name": "Casual", "tags": ["casual", "puzzle", "family", "party", "rhythm"]},
    {"name": "Hardcore", "tags": ["action", "challenging", "souls_like", "rpg", "shooter"]},
    {"name": "Story", "tags": ["story_rich", "narrative", "adventure", "atmospheric", "single_player"]},
    {"name": "Strategy", "tags": ["strategy", "tactical", "simulation", "management", "turn_based"]},
    {"name": "Retro", "tags": ["retro", "arcade", "platformer", "pixel_art", "2d_flat"]},
]


class ABTestSimulator:
    def __init__(self, recommender_a: HybridRecommender, recommender_b: HybridRecommender,
                 n_users: int = 100):
        self.rec_a = recommender_a
        self.rec_b = recommender_b
        self.n_users = n_users

    def simulate(self, k: int = 10) -> dict:
        np.random.seed(42)
        results_a, results_b = [], []

        for _ in range(self.n_users):
            profile = np.random.choice(TEST_PROFILES)
            recs_a = self.rec_a.recommend_for_profile(profile["tags"], k=k)
            recs_b = self.rec_b.recommend_for_profile(profile["tags"], k=k)

            tag_cols = [c for c in self.rec_a.feature_matrix.columns
                        if c in profile["tags"] and c in self.rec_a.feature_matrix.columns]
            relevant = set()
            for row in self.rec_a.feature_matrix.to_dicts():
                if any(row.get(t, 0) == 1 for t in tag_cols):
                    relevant.add(row["game_id"])

            results_a.append(ndcg_at_k([r["game_id"] for r in recs_a], relevant, k))
            results_b.append(ndcg_at_k([r["game_id"] for r in recs_b], relevant, k))

        arr_a = np.array(results_a)
        arr_b = np.array(results_b)
        lifts = arr_b - arr_a
        n_wins_a = int((arr_a > arr_b).sum())
        n_wins_b = int((arr_b > arr_a).sum())
        n_ties = int((arr_a == arr_b).sum())

        boot_diffs = []
        for _ in range(10000):
            idx = np.random.randint(0, len(lifts), len(lifts))
            boot_diffs.append(lifts[idx].mean())
        boot_diffs = np.sort(boot_diffs)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

        mean_diff = lifts.mean()
        se = lifts.std() / np.sqrt(len(lifts))
        z = mean_diff / se if se > 0 else 0
        p_value = 2 * (1 - _normal_cdf(abs(z)))

        return {
            "model_a": {"mean_ndcg": float(arr_a.mean()), "std_ndcg": float(arr_a.std())},
            "model_b": {"mean_ndcg": float(arr_b.mean()), "std_ndcg": float(arr_b.std())},
            "lift": float(mean_diff),
            "lift_pct": float(mean_diff / arr_a.mean() * 100) if arr_a.mean() > 0 else 0,
            "ci_95": [float(ci_lo), float(ci_hi)],
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_wins_a": n_wins_a,
            "n_wins_b": n_wins_b,
            "n_ties": n_ties,
            "n_users": self.n_users,
        }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
