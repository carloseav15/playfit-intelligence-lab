import nbformat as nbf

NB_NOTEBOOKS = {
    "notebooks/01_catalog_quality_audit.ipynb": {
        "title": "01 — Catalog Quality Audit",
        "description": "Análisis de cobertura, calidad y duplicados del catálogo Playfit.",
    },
}

NB = nbf.v4.new_notebook()
NB.metadata = {
    "kernelspec": {
        "display_name": "Python 3.12",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))


def generate_01():
    global cells
    cells = []
    md("# Playfit Intelligence Lab — 01: Catalog Quality Audit")
    md("""Este notebook analiza la calidad y cobertura del catálogo de juegos de Playfit.
La base de datos contiene **63,682 juegos** con datos de plataformas, tags, scores, ventas,
sentimiento de reseñas, matching externo y detección de duplicados.

**Objetivos:**
- Medir la cobertura real del catálogo (géneros, portadas, años, tags)
- Analizar la distribución del `data_confidence_score`
- Caracterizar los duplicados detectados y su estado de revisión
- Evaluar la calidad del matching externo (Metacritic, VGSales)
- Generar visualizaciones portfolio-grade""")

    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.catalog_features import (
    load_games, load_signals, load_platforms, load_tags,
    load_duplicates, load_external_match,
    coverage_analysis, signal_quality_analysis,
    duplicate_analysis, external_match_analysis,
)

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams['figure.figsize'] = (12, 6)
print("Librerías cargadas correctamente.")""")

    md("## 1. Cobertura del Catálogo\nCargamos los datos y calculamos métricas generales de cobertura.")

    code("""games = load_games()
cov = coverage_analysis(games)
for k, v in cov.items():
    if isinstance(v, float):
        print(f"{k:30s}: {v:.2f}%")
    else:
        print(f"{k:30s}: {v}")
print(f"\\nTotal juegos: {cov['total_games']:,}")""")

    md("""**Hallazgos típicos:**
- ~60% sin género asignado
- ~50% sin portada (cover_url)
- ~3-5% sin año de lanzamiento
- ~5% sin tags""")

    md("## 2. Fuentes y Estados de Publicación")

    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

source = cov['source_distribution']
ax1.bar(source['source_type'], source['len'], color=['#2ecc71', '#3498db', '#e74c3c'])
ax1.set_title('Distribución por Source Type')
ax1.set_ylabel('Cantidad de juegos')

states = cov['release_state_distribution']
ax2.bar(states['release_state'], states['len'], color=['#9b59b6', '#f39c12'])
ax2.set_title('Distribución por Release State')
ax2.set_ylabel('Cantidad de juegos')

plt.tight_layout()
plt.savefig('../reports/figures/source_distribution.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("## 3. Señales de Calidad (Data Confidence Score)\nLa vista `game_recommendation_enrichment_signals` expone un score de confianza (0-100) por juego.")

    code("""signals = load_signals()
sq = signal_quality_analysis(signals)
print("Distribución del Data Confidence Score:")
for k, v in sq.items():
    if isinstance(v, float):
        print(f"  {k:30s}: {v:.2f}")""")

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

confidence = signals['data_confidence_score'].to_numpy()
axes[0].hist(confidence, bins=50, color='#3498db', edgecolor='white')
axes[0].set_title('Distribución de Confidence Score')
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Juegos')

has_scores = signals['best_critic_score'].is_not_null().sum()
no_scores = len(signals) - has_scores
axes[1].pie([has_scores, no_scores], labels=['Con scores', 'Sin scores'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1].set_title('Cobertura de Scores')

has_sales = (signals['has_sales'] == True).sum()
no_sales = len(signals) - has_sales
axes[2].pie([has_sales, no_sales], labels=['Con ventas', 'Sin ventas'],
            autopct='%1.1f%%', colors=['#3498db', '#f39c12'])
axes[2].set_title('Cobertura de Ventas')

plt.tight_layout()
plt.savefig('../reports/figures/quality_coverage.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("## 4. Análisis de Duplicados")

    code("""da = duplicate_analysis()
print(f"Grupos de duplicados: {da['total_groups']}")
print(f"Candidatos afectados: {da['total_candidates']}")
print(f"Grupos con años diferentes: {da['groups_with_diff_years']}")
print(f"Max candidatos por grupo: {da['max_candidates_per_group']}")""")

    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

status = da['status_distribution']
ax1.bar(status['status'], status['len'], color=['#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#9b59b6', '#95a5a6'])
ax1.set_title('Estado de Grupos de Duplicados')
ax1.set_ylabel('Grupos')
ax1.tick_params(axis='x', rotation=45)

review = da['review_distribution']
ax2.bar(review['suggested_review'], review['len'], color=['#3498db', '#e74c3c', '#f39c12'])
ax2.set_title('Suggested Review')
ax2.set_ylabel('Grupos')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/duplicate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("## 5. Matching Externo (Metacritic + VGSales)")

    code("""matches = load_external_match()
ema = external_match_analysis(matches)
print(f"Total match candidates: {ema['total_matches']:,}")
print(f"Confianza promedio: {ema['avg_confidence']:.1f}/100")
print(f"Auto-aprobados (high confidence): {ema['high_confidence_approved']:,}")
print(f"Necesitan revisión: {ema['needs_review']:,}")""")

    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

status_m = ema['status_distribution']
ax1.bar(status_m['status'], status_m['len'],
        color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
ax1.set_title('Estado de Match Candidates')
ax1.tick_params(axis='x', rotation=45)

source_m = ema['source_distribution']
ax2.bar(source_m['source'], source_m['len'], color=['#3498db', '#e74c3c'])
ax2.set_title('Distribución por Fuente')
ax2.set_ylabel('Candidatos')

plt.tight_layout()
plt.savefig('../reports/figures/external_matching.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("## 6. Resumen y Conclusiones\n\n**Métricas clave de calidad del catálogo:**")

    code("""print("=" * 60)
print("RESUMEN DE CALIDAD DEL CATÁLOGO PLAYFIT")
print("=" * 60)
print(f"Total juegos:                 {cov['total_games']:>8,}")
print(f"Sin género:                   {cov['no_genre']:>7.1f}%")
print(f"Sin portada:                  {cov['no_cover']:>7.1f}%")
print(f"Sin año:                      {cov['no_year']:>7.1f}%")
print(f"Confianza promedio:           {sq['avg_confidence']:>7.1f}/100")
print(f"Con scores:                   {sq['has_scores_pct']:>7.1f}%")
print(f"Con ventas:                   {sq['has_sales_pct']:>7.1f}%")
print(f"Con sentimiento:              {sq['has_sentiment_pct']:>7.1f}%")
print(f"Grupos duplicados:            {da['total_groups']:>8,}")
print(f"Match candidates:             {ema['total_matches']:>8,}")
print("=" * 60)""")

    md("""**Conclusiones:**
1. El catálogo tiene **masa crítica** (63K+ juegos) pero con **huecos significativos** de cobertura
2. El `data_confidence_score` permite filtrar recomendaciones según calidad de datos
3. Hay **914 grupos de duplicados** que requieren limpieza manual
4. El matching externo (Metacritic + VGSales) añade **35K+ candidatos** con confianza variable
5. Para el sistema de recomendación, usaremos el confidence score como penalización y los duplicados para evitar recomendar versiones redundantes""")

    return nbf.v4.new_notebook(cells=cells, metadata=NB.metadata)


def generate_02():
    global cells
    cells = []
    md("# Playfit Intelligence Lab — 02: Feature Engineering\n\nConstrucción de la matriz de features para el motor de recomendación.")

    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
from src.models.content_based import build_content_model

sns.set_theme(style="darkgrid")
print("Librerías cargadas.")""")

    md("## 1. Feature Matrix\nConstruimos la matriz con tags (one-hot), plataformas (one-hot), scores, ventas, sentimiento y confianza.")

    code("""fm = build_feature_matrix()
fm = compute_popularity_score(fm)
fm = compute_richness_score(fm)
print(f"Feature matrix: {fm.shape[0]:,} juegos × {fm.shape[1]} columnas")
print(f"\\nColumnas por tipo:")
meta = {'game_id', 'title', 'release_year', 'genre_id', 'best_critic_score', 'best_user_score',
        'critic_review_count', 'user_review_count', 'critic_positive_ratio', 'user_positive_ratio',
        'max_global_sales_millions', 'total_global_sales_millions', 'data_confidence_score',
        'has_external_id', 'has_company', 'has_age_rating', 'has_summary', 'has_sales',
        'has_review_sentiment', 'log_sales', 'total_review_count', 'popularity_score', 'richness_score'}
content_cols = [c for c in fm.columns if c not in meta]
print(f"  Tags/Platforms (binary): {len(content_cols)}")
print(f"  Numéricas (scores, sales, etc.): {len(meta) - 2}")""")

    md("""## 2. Análisis de Features
### 2.1 Tags más comunes""")

    code("""tag_counts = {}
for col in content_cols:
    if col not in [c for c in fm.columns if c.startswith(('ps', 'nintendo', 'xbox', 'pc', 'sega', 'atari', 'arcade', 'switch', 'gamecube', 'game_gear', 'dreamcast', 'saturn', 'genesis', 'snes', 'nes', 'gb', 'gbc', 'gba', 'n64', 'ds', '3ds', 'psp', 'ps_vita', 'wii', 'wii_u', 'neo_geo'))]:
        tag_counts[col] = fm[col].sum()

sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
print("Top 20 tags más comunes:")
for tag, count in sorted_tags[:20]:
    print(f"  {tag:25s}: {int(count):>6,} juegos")""")

    code("""fig, ax = plt.subplots(figsize=(14, 7))
tags_names, tags_counts = zip(*sorted_tags[:20])
ax.barh(range(len(tags_names)), tags_counts, color='#3498db')
ax.set_yticks(range(len(tags_names)))
ax.set_yticklabels(tags_names)
ax.set_xlabel('Juegos')
ax.set_title('Top 20 Tags más comunes')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/top_tags.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("### 2.2 Distribución de Popularity Score")

    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(fm['popularity_score'].to_numpy(), bins=50, color='#2ecc71', edgecolor='white')
ax1.set_title('Distribución de Popularity Score')
ax1.set_xlabel('Popularity Score')

ax2.scatter(
    fm['popularity_score'].to_numpy(),
    fm['data_confidence_score'].to_numpy(),
    alpha=0.1, s=1, color='#3498db'
)
ax2.set_title('Popularity vs Data Confidence')
ax2.set_xlabel('Popularity Score')
ax2.set_ylabel('Confidence Score')

plt.tight_layout()
plt.savefig('../reports/figures/popularity_distribution.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("### 2.3 Reducción de dimensionalidad (SVD)")

    code("""cm = build_content_model(fm, n_components=100)
reduced = cm['reduced']
print(f"Dimensión original: {len(content_cols)}")
print(f"Dimensión reducida: {reduced.shape[1]}")
print(f"Varianza explicada (primeros componentes): {cm['svd'].explained_variance_ratio_[:5]}")
print(f"Varianza explicada acumulada (100 comps): {cm['svd'].explained_variance_ratio_.sum():.4f}")""")

    md("""## 3. Resumen de Features Creadas""")

    code("""print("\\nFeatures disponibles para el modelo:")
print(f"  - {len(content_cols)} features de contenido (tags + plataformas)")
print(f"  - Popularity score compuesto (scores + ventas + confianza)")
print(f"  - Richness score (cantidad de datos disponibles)")
print(f"  - Data confidence score (calidad de datos: 0-100)")
print(f"  - Embedding SVD-100 (representación densa de contenido)")
print(f"  - Features de señal: critic/user score, review counts, sales, sentiment")
print(f"\\nMatriz guardada en: data/processed/feature_matrix.parquet ({fm.shape[0]:,} × {fm.shape[1]})")""")

    return nbf.v4.new_notebook(cells=cells, metadata=NB.metadata)


def generate_03():
    global cells
    cells = []
    md("# Playfit Intelligence Lab — 03: Hybrid Recommender Model\n\nSistema de recomendación híbrido: content-based + popularity priors + penalización por calidad de datos.")

    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import polars as pl

from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
from src.models.content_based import build_content_model
from src.models.hybrid import HybridRecommender
from src.evaluation.explainability import make_explanation, get_game_details

print("Librerías y módulos cargados.")""")

    md("## 1. Preparación del Modelo\nCargamos la feature matrix y construimos el modelo content-based.")

    code("""fm = build_feature_matrix()
fm = compute_popularity_score(fm)
fm = compute_richness_score(fm)
print(f"Feature matrix: {fm.shape[0]:,} × {fm.shape[1]}")

cm = build_content_model(fm, n_components=100)
print(f"Content model construido (SVD-100)")""")

    md("## 2. Hybrid Recommender\n\n`final_score = α × content_score + β × popularity_score — γ × confidence_penalty`")

    code("""rec = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
rec.fit(fm, cm)
print("Modelo híbrido listo.")""")

    md("## 3. Ejemplo: Recomendar para juegos conocidos")

    code("""liked = ['the_legend_of_zelda_breath_of_the_wild', 'super_mario_odyssey']
results = rec.recommend(liked, k=5)

print(f"Basado en: {liked}")
print("=" * 80)
for r in results:
    details = get_game_details(r['game_id'])
    expl = make_explanation(r)
    print(f"  {r['game_id']}")
    print(f"    Score: {r['final_score']:.3f} | Content: {r['content_score']:.3f} | Pop: {r['popularity_score']:.3f} | Conf: {r['data_confidence']}")
    print(f"    → {expl}")
    print()""")

    md("## 4. Cold-Start: Nuevo usuario sin historial")

    code("""cold_results = rec.recommend([], k=5)
print("Cold-start (popularidad global):")
for r in cold_results:
    print(f"  {r['game_id']}: score={r['final_score']:.2f}, conf={r['data_confidence']}")""")

    md("## 5. Recomendación por Perfil de Tags")

    code("""profile_recs = rec.recommend_for_profile(['story_rich', 'atmospheric', 'exploration'], k=5)
print("Para perfil: story_rich + atmospheric + exploration")
for r in profile_recs:
    print(f"  {r['game_id']}: score={r['final_score']:.2f}")""")

    md("## 6. Experimentación con Hiperparámetros (α, β, γ)")

    code("""import itertools

param_grid = [
    (0.6, 0.3, 0.1), (0.5, 0.4, 0.1), (0.4, 0.5, 0.1),
    (0.7, 0.2, 0.1), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2),
]

test_liked = ['the_legend_of_zelda_breath_of_the_wild', 'hollow_knight', 'portal_2']
test_excluded = []

print(f"{'α':>5} {'β':>5} {'γ':>5}  {'Top-1':>40} {'Score':>8}")
print("-" * 65)

results_summary = []
for alpha, beta, gamma in param_grid:
    r = HybridRecommender(alpha=alpha, beta=beta, gamma=gamma)
    r.fit(fm, cm)
    recs = r.recommend(test_liked, k=3)
    top = recs[0]['game_id'] if recs else 'N/A'
    score = recs[0]['final_score'] if recs else 0.0
    print(f"{alpha:5.1f} {beta:5.1f} {gamma:5.1f}  {top:>40} {score:8.3f}")
    results_summary.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'top': top, 'score': score})

print("\\nLos mejores parámetros variarán según el perfil de usuario. En la práctica, se optimizarían con validación cruzada sobre datos de interacción reales.")""")

    md("""## 7. Resumen del Modelo

**Componentes:**
- **Content-Based (α):** Similaridad coseno sobre embedding SVD-100 de tags y plataformas
- **Popularity Prior (β):** Ranking compuesto de scores, ventas y sentimiento
- **Confidence Penalty (γ):** Penalización por baja calidad de datos

**Casos borde manejados:**
- **Cold-start:** Ranking por popularidad global cuando no hay historial
- **Datos incompletos:** Imputación de scores/ventas faltantes por mediana
- **Diversidad:** Embedding SVD captura relaciones semánticas entre juegos""")

    return nbf.v4.new_notebook(cells=cells, metadata=NB.metadata)


def generate_04():
    global cells
    cells = []
    md("# Playfit Intelligence Lab — 04: Model Evaluation\n\nEvaluación offline del recomendador híbrido con métricas de precisión, cobertura, novedad y diversidad.")

    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
from src.models.content_based import build_content_model
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k,
    hit_rate_at_k, coverage, novelty, diversity,
)

sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
print("Librerías y módulos cargados.")""")

    md("## 1. Setup del Modelo y Datos de Evaluación")

    code("""fm = build_feature_matrix()
fm = compute_popularity_score(fm)
fm = compute_richness_score(fm)
cm = build_content_model(fm, n_components=100)

rec = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
rec.fit(fm, cm)

popularity_map = dict(zip(fm['game_id'].to_list(), fm['popularity_score'].to_list()))

# Perfiles de usuario sintéticos para evaluación
test_profiles = [
    {"name": "Casual Gamer", "tags": ["casual", "puzzle", "family", "party", "rhythm"]},
    {"name": "Hardcore Gamer", "tags": ["action", "challenging", "souls_like", "rpg", "shooter"]},
    {"name": "Story Seeker", "tags": ["story_rich", "narrative", "adventure", "atmospheric", "single_player"]},
    {"name": "Strategy Fan", "tags": ["strategy", "tactical", "simulation", "management", "turn_based"]},
    {"name": "Retro Player", "tags": ["retro", "arcade", "platformer", "pixel_art", "2d_flat"]},
]

all_recommendations = []
all_relevant = []
for profile in test_profiles:
    recs = rec.recommend_for_profile(profile['tags'], k=20)
    all_recommendations.append([r['game_id'] for r in recs])
    relevant_tags = set(profile['tags'])
    relevant = set()
    for i, row in enumerate(fm.to_dicts()):
        if any(tag in relevant_tags for tag in profile['tags'] if tag in row and row[tag] == 1):
            relevant.add(row['game_id'])
    all_relevant.append(relevant)

print(f"{'Perfil':20s} {'Relevantes':>12s} {'Recomendados':>14s}")
print("-" * 50)
for p, recs, rel in zip(test_profiles, all_recommendations, all_relevant):
    print(f"{p['name']:20s} {len(rel):>12,} {len(recs):>14,}")""")

    md("## 2. Métricas Core: Precision@k, Recall@k, NDCG@k")

    code("""ks = [1, 3, 5, 10, 20]
metrics_by_k = {}

for k in ks:
    precs = [precision_at_k(rec, rel, k) for rec, rel in zip(all_recommendations, all_relevant)]
    recs_m = [recall_at_k(rec, rel, k) for rec, rel in zip(all_recommendations, all_relevant)]
    ndcgs = [ndcg_at_k(rec, rel, k) for rec, rel in zip(all_recommendations, all_relevant)]
    metrics_by_k[k] = {
        'precision': np.mean(precs),
        'recall': np.mean(recs_m),
        'ndcg': np.mean(ndcgs),
    }

print(f"{'k':>5} {'Precision@k':>14} {'Recall@k':>12} {'NDCG@k':>10}")
print("-" * 45)
for k in ks:
    m = metrics_by_k[k]
    print(f"{k:>5} {m['precision']:>14.4f} {m['recall']:>12.4f} {m['ndcg']:>10.4f}")""")

    code("""fig, ax = plt.subplots(figsize=(10, 6))
for metric in ['precision', 'recall', 'ndcg']:
    values = [metrics_by_k[k][metric] for k in ks]
    ax.plot(ks, values, marker='o', label=metric)
ax.set_xlabel('k')
ax.set_ylabel('Score')
ax.set_title('Métricas de Evaluación vs k')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('../reports/figures/metrics_vs_k.png', dpi=150, bbox_inches='tight')
plt.show()""")

    md("## 3. Cobertura del Catálogo")

    code("""total_catalog = len(fm)
for profile, recs in zip(test_profiles, all_recommendations):
    cov = coverage([recs], total_catalog)
    print(f"{profile['name']:20s} Coverage: {cov*100:.2f}% ({int(cov*total_catalog):,} de {total_catalog:,} juegos)")

combined_coverage = coverage(all_recommendations, total_catalog)
print(f"\\n{'Combinado':20s} Coverage: {combined_coverage*100:.2f}% ({int(combined_coverage*total_catalog):,} de {total_catalog:,} juegos)")""")

    md("## 4. Novedad (Novelty)")

    code("""for profile, recs in zip(test_profiles, all_recommendations):
    nov = novelty([recs], popularity_map)
    print(f"{profile['name']:20s} Novelty: {nov:.4f}")""")

    md("## 5. Diversidad Intra-lista")

    code("""for profile, recs in zip(test_profiles, all_recommendations):
    div = diversity([recs], rec, k=5)
    print(f"{profile['name']:20s} Diversity@5: {div:.4f}")""")

    md("## 6. Cold-Start Analysis")

    code("""confidence_segments = [(0, 30, 'Baja'), (30, 60, 'Media'), (60, 80, 'Alta'), (80, 100, 'Muy alta')]
print(f"{'Segmento':15s} {'Juegos':>10s} {'En Top-20':>10s} {'Proporción':>10s}")
print("-" * 47)
all_top20 = set()
for recs in all_recommendations:
    all_top20.update(recs[:20])

for lo, hi, label in confidence_segments:
    mask = (fm['data_confidence_score'] >= lo) & (fm['data_confidence_score'] < hi)
    count = mask.sum()
    in_top = sum(fm.filter(mask)['game_id'].is_in(list(all_top20)))
    prop = in_top / count if count > 0 else 0
    print(f"{label:15s} {count:>10,} {in_top:>10,} {prop:>10.3f}")""")

    md("""## 7. Resumen de Evaluación""")

    code("""print("=" * 60)
print("RESUMEN DE EVALUACIÓN - PLAYFIT RECOMMENDER")
print("=" * 60)
m5 = metrics_by_k[5]
print(f"Precision@5 (promedio):    {m5['precision']:.4f}")
print(f"Recall@5 (promedio):       {m5['recall']:.4f}")
print(f"NDCG@5 (promedio):         {m5['ndcg']:.4f}")
print(f"MAP@20 (promedio):         {map_at_k(all_recommendations, all_relevant, 20):.4f}")
print(f"Hit Rate@20:               {hit_rate_at_k(all_recommendations, all_relevant, 20):.4f}")
print(f"Cobertura del catálogo:    {combined_coverage*100:.2f}%")
print()
print("Mejoras potenciales:")
print("  - Incorporar feedback implícito (tiempo de juego, clicks)")
print("  - Añadir filtrado colaborativo (matrix factorization)")
print("  - Optimizar α/β/γ con validación cruzada sobre datos de interacción reales")
print("  - Añadir re-ranking de diversidad (MMR)")
print("=" * 60)""")

    return nbf.v4.new_notebook(cells=cells, metadata=NB.metadata)


if __name__ == "__main__":
    print("Generating notebook 01...")
    nb01 = generate_01()
    with open("notebooks/01_catalog_quality_audit.ipynb", "w") as f:
        nbf.write(nb01, f)
    print("  Done (01)")

    print("Generating notebook 02...")
    nb02 = generate_02()
    with open("notebooks/02_feature_engineering.ipynb", "w") as f:
        nbf.write(nb02, f)
    print("  Done (02)")

    print("Generating notebook 03...")
    nb03 = generate_03()
    with open("notebooks/03_recommender_model.ipynb", "w") as f:
        nbf.write(nb03, f)
    print("  Done (03)")

    print("Generating notebook 04...")
    nb04 = generate_04()
    with open("notebooks/04_model_evaluation.ipynb", "w") as f:
        nbf.write(nb04, f)
    print("  Done (04)")

    print("\nAll 4 notebooks generated successfully!")
