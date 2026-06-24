import nbformat as nbf

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


NB_META = {
    "kernelspec": {
        "display_name": "Python 3.12",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.0"},
}


def make_notebook(generator_fn):
    global cells
    cells = []
    generator_fn()
    return nbf.v4.new_notebook(cells=cells, metadata=NB_META)


# ─────────────────────────────────────────────
# Notebook 05: MLflow + A/B Test + LLM
# ─────────────────────────────────────────────
def nb05():
    global cells
    cells = []

    md("# Playfit Intelligence Lab — 05: MLflow Tracking, A/B Testing & LLM Explainability")
    md("""Este notebook demuestra tres capacidades avanzadas de MLOps y evaluación:

1. **MLflow Experiment Tracking**: Logging de parámetros, métricas y artefactos
2. **A/B Test Simulator**: Confrontación estadística entre variantes del modelo
3. **LLM Explainability**: Explicaciones narrativas con GPT-4o-mini""")

    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl

from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
from src.models.content_based import build_content_model
from src.models.hybrid import HybridRecommender

print("Librerías cargadas.")
print(f"MLflow version: {__import__('mlflow').__version__}")""")

    md("## 1. MLflow Experiment Tracking\n\nCreamos un experimento y corremos grid search sobre α, β, γ.")

    code("""import mlflow

mlflow.set_experiment("playfit-hybrid-demo")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print("To view: run 'mlflow ui' in terminal, open http://localhost:5000")""")

    code("""from src.training.experiment_grid import run_grid_search

# This runs 10 experiments and logs params/metrics/figures to MLflow
df_results = run_grid_search(experiment_name="playfit-hybrid-demo")
print(f"\\nBest config: α={df_results.iloc[0]['alpha']}, β={df_results.iloc[0]['beta']}, γ={df_results.iloc[0]['gamma']}")""")

    md("## 2. A/B Test Simulator\n\nComparamos dos variantes del modelo con significancia estadística.")

    code("""from src.evaluation.ab_test import ABTestSimulator

fm = compute_richness_score(compute_popularity_score(build_feature_matrix()))
cm = build_content_model(fm, n_components=100)

rec_a = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
rec_a.fit(fm, cm)

rec_b = HybridRecommender(alpha=0.7, beta=0.2, gamma=0.1)
rec_b.fit(fm, cm)

ab = ABTestSimulator(rec_a, rec_b, n_users=100)
result = ab.simulate(k=10)

print("=== A/B Test: Variant A (α=0.5) vs Variant B (α=0.7) ===")
print(f"Users: {result['n_users']}")
print(f"Model A mean NDCG@10: {result['model_a']['mean_ndcg']:.4f}")
print(f"Model B mean NDCG@10: {result['model_b']['mean_ndcg']:.4f}")
print(f"Lift: {result['lift_pct']:.2f}% ({result['lift']:.4f})")
print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
print(f"p-value: {result['p_value']:.4f}")
print(f"Statistically significant: {result['significant']}")
print(f"Wins A: {result['n_wins_a']}, Wins B: {result['n_wins_b']}, Ties: {result['n_ties']}")""")

    code("""import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

means = [result['model_a']['mean_ndcg'], result['model_b']['mean_ndcg']]
stds = [result['model_a']['std_ndcg'], result['model_b']['std_ndcg']]
bars = ax1.bar(['Variant A (α=0.5)', 'Variant B (α=0.7)'], means, yerr=stds,
               capsize=10, color=['#3498db', '#e74c3c'])
ax1.set_ylabel('Mean NDCG@10')
ax1.set_title('A/B Test: Mean Performance')
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
             f'{mean:.4f}', ha='center', fontsize=11)

wins = [result['n_wins_a'], result['n_wins_b'], result['n_ties']]
ax2.bar(['Variant A Wins', 'Variant B Wins', 'Ties'], wins,
        color=['#3498db', '#e74c3c', '#95a5a6'])
ax2.set_ylabel('Users')
ax2.set_title(f'Win Distribution (p={result[\"p_value\"]:.3f})')

plt.tight_layout()
plt.savefig('../reports/figures/ab_test_results.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("## 3. LLM Explainability\n\nGeneramos explicaciones narrativas usando un LLM y comparamos con el baseline rule-based.")

    code("""from src.evaluation.explainability import make_explanation, get_game_details
from src.evaluation.llm_explainability import LLMExplainer, compare_explanations

fm = compute_richness_score(compute_popularity_score(build_feature_matrix()))
cm = build_content_model(fm, n_components=100)
rec = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
rec.fit(fm, cm)

results = rec.recommend(['the_legend_of_zelda_breath_of_the_wild'], k=5)

explainer = LLMExplainer(model="gpt-4o-mini")
print("Generating LLM explanations... (requires OPENROUTER_API_KEY or OPENAI_API_KEY)")

for r in results:
    details = get_game_details(r['game_id'])
    rule_expl = make_explanation(r)
    print(f"\\n--- {details.get('title', r['game_id'])} ---")
    print(f"Rule-based: {rule_expl}")
    llm_expl = explainer.explain(r)
    print(f"LLM: {llm_expl}")
    comp = compare_explanations(rule_expl, llm_expl)
    print(f"  (LLM longer: {comp['llm_is_longer']}, chars: {comp['rule_based_chars']} vs {comp['llm_based_chars']})")""")

    md("""## 4. Resumen de Capacidades Añadidas

| Capacidad | Herramienta | Estado |
|-----------|------------|--------|
| Experiment tracking | MLflow 3.14 | ✅ Logs a `mlruns/` |
| Grid search | 10 configuraciones α/β/γ | ✅ Auto-logueado |
| A/B Testing | Bootstrap + CI + p-value | ✅ |
| LLM Explainability | GPT-4o-mini via OpenRouter | ✅ (requiere API key) |
| MLflow UI | `mlflow ui` | ✅ http://localhost:5000 |""")

    return nbf.v4.new_notebook(cells=cells, metadata=NB_META)


# ─────────────────────────────────────────────
# Notebook 06: LambdaRank + DuckDB
# ─────────────────────────────────────────────
def nb06():
    global cells
    cells = []

    md("# Playfit Intelligence Lab — 06: LambdaRank & DuckDB Analytics")
    md("""Dos temas avanzados:

1. **LightGBM LambdaRank**: Modelo de ranking basado en gradient boosting comparado contra el híbrido
2. **DuckDB Analytics**: Motor OLAP embebido para consultas analíticas sobre Parquet""")

    md("## 1. LightGBM LambdaRank\n\nEntrenamos un `LGBMRanker` con `objective='lambdarank'` y comparamos NDCG contra el modelo híbrido.")

    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import lightgbm
print(f"LightGBM version: {lightgbm.__version__}")

from src.models.lambdarank import LambdaRankRecommender
from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
from src.models.content_based import build_content_model
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import ndcg_at_k

fm = compute_richness_score(compute_popularity_score(build_feature_matrix()))
cm = build_content_model(fm, n_components=100)

print("Training LambdaRank (this may take a few minutes)...")
lr = LambdaRankRecommender(n_estimators=100, learning_rate=0.1)
lr.fit(fm)
ndcg = lr.evaluate_ndcg(fm)

print("\\nLambdaRank NDCG results:")
for k, v in ndcg.items():
    print(f"  {k}: {v:.4f}")""")

    code("""# Compare with hybrid model
rec = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
rec.fit(fm, cm)

test_profiles = [
    {"name": "Casual", "tags": ["casual", "puzzle", "family"]},
    {"name": "Story", "tags": ["story_rich", "narrative", "adventure"]},
    {"name": "Hardcore", "tags": ["challenging", "action", "rpg"]},
]

hybrid_ndcgs = []
for prof in test_profiles:
    recs = rec.recommend_for_profile(prof["tags"], k=20)
    tag_cols = [c for c in prof["tags"] if c in fm.columns]
    relevant = set()
    for row in fm.to_dicts():
        if any(row.get(t, 0) == 1 for t in tag_cols):
            relevant.add(row["game_id"])
    ndcg_val = ndcg_at_k([r["game_id"] for r in recs], relevant, 10)
    hybrid_ndcgs.append(ndcg_val)

print("\\nModel Comparison (NDCG@10):")
print(f"  LambdaRank avg: {np.mean(list(ndcg.values())):.4f}")
print(f"  Hybrid avg:     {np.mean(hybrid_ndcgs):.4f}")""")

    code("""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
models = ['LambdaRank', 'Hybrid (α=0.5)']
scores = [np.mean(list(ndcg.values())), np.mean(hybrid_ndcgs)]
ax.bar(models, scores, color=['#9b59b6', '#3498db'], width=0.4)
ax.set_ylabel('Mean NDCG@10')
ax.set_title('LambdaRank vs Hybrid Recommender')
for i, s in enumerate(scores):
    ax.text(i, s + 0.005, f'{s:.4f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('../reports/figures/lambdarank_vs_hybrid.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("""## 2. DuckDB Analytics\n\nDuckDB permite consultas SQL directamente sobre archivos Parquet, ideal para análisis exploratorio sin cargar datos en memoria.""")

    code("""from src.data.analytics_duckdb import DuckDBAnalytics
da = DuckDBAnalytics()

print("Games by year (sample):")
print(da.games_by_year().sort("release_year", descending=True).head(10))

print("\\nTop platforms:")
print(da.top_platforms(10))

print("\\nAverage confidence by genre:")
print(da.avg_confidence_by_genre().head(10))

print("\\nCatalog coverage (single query):")
print(da.coverage_pivot())

print("\\nDuplicate groups by status:")
print(da.duplicate_summary())""")

    md("### Window Functions y Approximate Counts")

    code("""print("\\nTop games per platform (window function):")
print(da.ranked_games_by_platform().head(20))

print("\\nApproximate tag diversity (approx vs exact):")
print(da.approximate_tag_diversity().head(10))

da.close()""")

    md("### Performance: DuckDB vs Polars vs Pandas")

    perf_code = r'''
import time
import polars as pl
import pandas as pd

_ = da.query("SELECT count(*) FROM read_parquet('data/raw/game_tags.parquet')")

start = time.time()
for _ in range(5):
    q = "SELECT tag_id, count(*) as cnt FROM read_parquet('data/raw/game_tags.parquet') GROUP BY tag_id ORDER BY cnt DESC"
    r = da.query(q)
duck_time = (time.time() - start) / 5

pt = pl.read_parquet("data/raw/game_tags.parquet")
start = time.time()
for _ in range(5):
    r = pt.group_by("tag_id").len().sort("len", descending=True)
polars_time = (time.time() - start) / 5

pd_df = pd.read_parquet("data/raw/game_tags.parquet")
start = time.time()
for _ in range(5):
    r = pd_df.groupby("tag_id").size().sort_values(ascending=False)
pandas_time = (time.time() - start) / 5

print(f"\nPerformance: GROUP BY tag_id (avg of 5 runs)")
print(f"  DuckDB:  {duck_time*1000:.1f}ms")
print(f"  Polars:  {polars_time*1000:.1f}ms")
print(f"  Pandas:  {pandas_time*1000:.1f}ms")
'''
    code(perf_code)

    md("""## 3. Resumen

| Técnica | Librería | Resultado |
|---------|----------|-----------|
| LambdaRank | LightGBM 4.6 | Modelo de ranking competitivo con baseline |
| SQL Analytics | DuckDB 1.5 | Consultas directo sobre Parquet, sin carga |
| Window Functions | DuckDB SQL | `ROW_NUMBER() OVER (PARTITION BY)` |
| Approximate Count | DuckDB `approx_count_distinct()` | ~rápido para cardinalidad de tags |""")

    return nbf.v4.new_notebook(cells=cells, metadata=NB_META)


if __name__ == "__main__":
    print("Generating notebook 05 (MLflow + A/B + LLM)...")
    nb = make_notebook(nb05)
    with open("notebooks/05_mlflow_abtest_llm.ipynb", "w") as f:
        nbf.write(nb, f)
    print("  Done (05)")

    print("Generating notebook 06 (LambdaRank + DuckDB)...")
    nb = make_notebook(nb06)
    with open("notebooks/06_lambdarank_duckdb.ipynb", "w") as f:
        nbf.write(nb, f)
    print("  Done (06)")

    print("\nBoth notebooks generated successfully!")

    import sys
    sys.exit(0)
