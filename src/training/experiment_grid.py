import mlflow
import pandas as pd

from src.training.train_pipeline import train_hybrid

PARAM_GRID = [
    (0.6, 0.3, 0.1),
    (0.5, 0.4, 0.1),
    (0.4, 0.5, 0.1),
    (0.7, 0.2, 0.1),
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.3, 0.6, 0.1),
    (0.6, 0.2, 0.2),
    (0.5, 0.35, 0.15),
    (0.45, 0.45, 0.1),
]


def run_grid_search(experiment_name: str = "playfit-hybrid-grid") -> pd.DataFrame:
    results = []
    for alpha, beta, gamma in PARAM_GRID:
        print(f"Training α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}...")
        out = train_hybrid(alpha, beta, gamma, log_mlflow=True, experiment_name=experiment_name)
        row = {"alpha": alpha, "beta": beta, "gamma": gamma}
        row.update(out["metrics"])
        results.append(row)
        print(f"  Precision_at_5: {out['metrics']['precision_at_5']:.4f}, NDCG_at_5: {out['metrics']['ndcg_at_5']:.4f}")

    df = pd.DataFrame(results)
    df = df.sort_values("ndcg_at_5", ascending=False).reset_index(drop=True)
    print("\n=== Grid Search Results (sorted by NDCG_at_5) ===")
    print(df[["alpha", "beta", "gamma", "precision_at_5", "ndcg_at_5", "coverage"]].to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest params: α={best['alpha']}, β={best['beta']}, γ={best['gamma']}")
    print(f"Best NDCG_at_5: {best['ndcg_at_5']:.4f}")
    return df
