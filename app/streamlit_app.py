import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IS_DEMO = (Path(__file__).resolve().parent.parent / "data" / "demo").exists()

if not IS_DEMO:
    from src.features.catalog_features import (
        load_games, load_signals, load_platforms, load_tags,
        load_duplicates, load_external_match,
        coverage_analysis, signal_quality_analysis,
        duplicate_analysis, external_match_analysis,
    )
    from src.features.game_features import (
        build_feature_matrix, compute_popularity_score, compute_richness_score,
    )
    from src.models.content_based import build_content_model
    from src.models.hybrid import HybridRecommender
    from src.evaluation.explainability import make_explanation, get_game_details
    from src.evaluation.metrics import (
        precision_at_k, recall_at_k, ndcg_at_k, map_at_k,
        hit_rate_at_k, coverage as coverage_metric, novelty, diversity,
    )

from src.models.demo_loader import load_demo_recommender, search_games_in_demo

sns.set_theme(style="darkgrid")

st.set_page_config(
    page_title="Playfit Intelligence Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_data():
    if IS_DEMO:
        return None
    games = load_games()
    signals = load_signals()
    groups, candidates = load_duplicates()
    matches = load_external_match()
    return games, signals, groups, candidates, matches


@st.cache_resource
def load_model():
    if IS_DEMO:
        rec = load_demo_recommender()
        return rec.feature_matrix, rec.content_model, rec
    fm = build_feature_matrix()
    fm = compute_popularity_score(fm)
    fm = compute_richness_score(fm)
    cm = build_content_model(fm, n_components=100)
    rec = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
    rec.fit(fm, cm)
    return fm, cm, rec


st.title("🎮 Playfit Intelligence Lab")
st.markdown("**Data Quality · Recommendation Engine · Model Evaluation**")
if IS_DEMO:
    st.info("🌐 Demo mode — showing top 500 games by popularity. Full mode requires local DB access.")
st.markdown("---")

tab_catalog, tab_recommender, tab_evaluation = st.tabs([
    "📊 Catalog Health", "🎯 Recommendation Engine", "📈 Model Evaluation"
])

# ──────────────────────────────────────
# TAB 1: Catalog Health
# ──────────────────────────────────────
with tab_catalog:
    if IS_DEMO:
        st.info("📊 Catalog Health requires full dataset. Showing demo dataset overview instead.")
        fm_demo, _, _ = load_model()
        st.metric("Games in demo", len(fm_demo))
        st.markdown("""
        **Deploy full version:** clone repo locally and connect to your Supabase PostgreSQL.
        ```bash
        python3 scripts/feedback_to_db.py      # Write diagnostics to DB
        streamlit run app/streamlit_app.py      # Full Streamlit app
        ```
        """)
        st.stop()

    games, signals, groups, candidates, matches = load_all_data()

    cov = coverage_analysis(games)
    sq = signal_quality_analysis(signals)
    da = duplicate_analysis()
    ema = external_match_analysis(matches)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Games", f"{cov['total_games']:,}")
    with col2:
        st.metric("With Genre", f"{100 - cov['no_genre']:.1f}%")
    with col3:
        st.metric("With Cover", f"{100 - cov['no_cover']:.1f}%")
    with col4:
        st.metric("Avg Confidence", f"{sq['avg_confidence']:.1f}")
    with col5:
        st.metric("Duplicate Groups", f"{da['total_groups']:,}")

    st.markdown("### Quality Coverage")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ['Genre', 'Cover', 'Year', 'Scores', 'Sales', 'Sentiment']
        values = [
            100 - cov['no_genre'],
            100 - cov['no_cover'],
            100 - cov['no_year'],
            sq['has_scores_pct'],
            sq['has_sales_pct'],
            sq['has_sentiment_pct'],
        ]
        colors = ['#2ecc71' if v > 50 else '#f39c12' if v > 20 else '#e74c3c' for v in values]
        ax.barh(categories, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Coverage %')
        ax.set_title('Data Coverage by Category')
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=11)
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(8, 4))
        confidence = signals['data_confidence_score'].to_numpy()
        ax.hist(confidence, bins=40, color='#3498db', edgecolor='white')
        ax.set_xlabel('Data Confidence Score')
        ax.set_ylabel('Number of Games')
        ax.set_title('Confidence Score Distribution')
        st.pyplot(fig)
        plt.close()

    st.markdown("### External Match Quality")
    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(8, 4))
        status_data = ema['status_distribution']
        bars = ax.bar(status_data['status'], status_data['len'],
                       color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
        ax.set_title('Match Status Distribution')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    with col_d:
        fig, ax = plt.subplots(figsize=(8, 4))
        source_data = ema['source_distribution']
        ax.bar(source_data['source'], source_data['len'], color=['#3498db', '#e74c3c'])
        ax.set_title('Matches by Source')
        ax.set_ylabel('Candidates')
        st.pyplot(fig)
        plt.close()

    with st.expander("Catalog Quality Summary"):
        st.markdown(f"""
        - **{cov['total_games']:,}** games in catalog
        - **{100 - cov['no_genre']:.1f}%** have a genre assigned
        - **{100 - cov['no_cover']:.1f}%** have a cover image
        - **{100 - cov['no_year']:.1f}%** have a release year
        - **{sq['has_scores_pct']:.1f}%** have critic/user scores
        - **{sq['has_sales_pct']:.1f}%** have sales data
        - **{sq['has_sentiment_pct']:.1f}%** have review sentiment
        - **{da['total_groups']:,}** duplicate groups ({da['total_candidates']:,} candidates)
        - **{ema['total_matches']:,}** external match candidates (avg confidence: {ema['avg_confidence']:.0f})
        - High confidence ({sq['pct_high_confidence']:.0f}%), Medium ({sq['pct_medium_confidence']:.0f}%), Low ({sq['pct_low_confidence']:.0f}%)
        """)


# ──────────────────────────────────────
# TAB 2: Recommendation Engine
# ──────────────────────────────────────
with tab_recommender:
    fm, cm, rec = load_model()

    st.markdown("### Try the Recommender")
    st.caption("Select games you like, adjust parameters, and see recommendations with explanations.")

    if IS_DEMO:
        search_q = st.text_input("Search games by title:", placeholder="e.g. zelda, mario, final fantasy")
        if search_q:
            matches = search_games_in_demo(search_q, rec)
            if matches:
                game_options = {m["title"]: m["game_id"] for m in matches}
            else:
                game_options = {}
                st.warning("No games found matching that search.")
        else:
            game_options = {}
    else:
        game_options = fm.select(["game_id", "title"]).to_pandas()
        game_list = game_options["title"].tolist()
        game_id_map = dict(zip(game_options["title"], game_options["game_id"]))
        reverse_map = dict(zip(game_options["game_id"], game_options["title"]))

    if IS_DEMO and "game_options" in dir():
        liked_titles = st.multiselect(
            "Games you like (select 1-5):",
            options=list(game_options.keys()) if game_options else [],
            default=[],
            max_selections=5,
        )
        liked_ids = [game_options[t] for t in liked_titles] if game_options else []
    else:
        liked_titles = st.multiselect(
            "Games you like (select 1-5):",
            options=game_list if not IS_DEMO else [],
            default=[],
            max_selections=5,
        )
        liked_ids = [game_id_map[t] for t in liked_titles]

    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider("Content weight (α)", 0.0, 1.0, 0.5, 0.05)
    with col2:
        beta = st.slider("Popularity weight (β)", 0.0, 1.0, 0.4, 0.05)
    with col3:
        gamma = st.slider("Confidence penalty (γ)", 0.0, 0.5, 0.1, 0.05)

    col_a, col_b = st.columns(2)
    with col_a:
        n_recs = st.slider("Number of recommendations", 5, 30, 10, 5)
    with col_b:
        use_mmr = st.checkbox("Enable diversity re-ranking (MMR)", value=False)
        if use_mmr:
            mmr_lambda = st.slider("MMR λ (0=only diversity, 1=only relevance)", 0.0, 1.0, 0.5, 0.05)

    use_llm = st.checkbox("Use LLM explanations (requires API key)", value=False)

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        if not liked_ids:
            st.warning("Select at least one game, or try cold-start mode below.")
        else:
            rec.alpha, rec.beta, rec.gamma = alpha, beta, gamma
            results = rec.recommend(liked_ids, k=n_recs)

            if use_mmr:
                mmr_lam = mmr_lambda if use_mmr else 0.5
                results = rec.rerank_with_mmr(results, lambda_param=mmr_lam, k=n_recs)

            explainer = None
            if use_llm:
                from src.evaluation.llm_explainability import LLMExplainer
                explainer = LLMExplainer(model="gpt-4o-mini")

            st.markdown(f"### Top {n_recs} Recommendations")
            for i, r in enumerate(results):
                if IS_DEMO:
                    details = {"title": r["game_id"], "year": "", "genre": "", "platforms": []}
                    def make_demo_explanation(r2):
                        return (f"This game is recommended because it has high content similarity "
                                f"(score: {r2['content_score']:.2f}) to your selected games, "
                                f"coupled with strong popularity (score: {r2['popularity_score']:.2f}).")
                    expl = explainer.explain(r) if explainer else make_demo_explanation(r)
                else:
                    from src.evaluation.explainability import make_explanation, get_game_details
                    details = get_game_details(r['game_id'])
                    expl = explainer.explain(r) if explainer else make_explanation(r)

                with st.container(border=True):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        title = details.get('title', r['game_id'])
                        year = details.get('year', '')
                        genre = details.get('genre', '')
                        platforms = ', '.join(details.get('platforms', [])) if details.get('platforms') else ''
                        st.markdown(f"**{i+1}. {title}**  ")
                        st.caption(f"`{r['game_id']}`  ")
                        if year:
                            st.caption(f"📅 {year}  ")
                        if genre:
                            st.caption(f"🏷️ {genre}  ")
                        if platforms:
                            st.caption(f"🖥️ {platforms}  ")
                    with col_b:
                        st.metric("Score", f"{r['final_score']:.2f}")

                    st.markdown(f"**Why:** {expl}")
                    top_score = max(r['final_score'] for r in results)
                    st.progress(
                        min(r['final_score'] / top_score, 1.0),
                        text=f"Content: {r['content_score']:.2f} · Popularity: {r['popularity_score']:.2f} · Confidence: {r['data_confidence']}/100"
                    )

    st.markdown("---")
    st.markdown("### Cold-Start Mode")
    st.caption("No user history? The recommender falls back to global popularity ranking.")

    if st.button("Show Cold-Start Recommendations", use_container_width=True):
        rec.alpha, rec.beta, rec.gamma = alpha, beta, gamma
        cold_results = rec.recommend([], k=10)
        st.markdown("**Top 10 by Popularity (Cold-Start):**")
        for i, r in enumerate(cold_results):
            st.markdown(f"{i+1}. **{r['game_id']}** — Score: {r['final_score']:.2f} · Confidence: {r['data_confidence']}/100  ")


# ──────────────────────────────────────
# TAB 3: Model Evaluation
# ──────────────────────────────────────
with tab_evaluation:
    fm, cm, rec = load_model()

    test_profiles = [
        {"name": "Casual Gamer", "tags": ["casual", "puzzle", "family", "party", "rhythm"]},
        {"name": "Hardcore Gamer", "tags": ["action", "challenging", "souls_like", "rpg", "shooter"]},
        {"name": "Story Seeker", "tags": ["story_rich", "narrative", "adventure", "atmospheric", "single_player"]},
        {"name": "Strategy Fan", "tags": ["strategy", "tactical", "simulation", "management", "turn_based"]},
        {"name": "Retro Player", "tags": ["retro", "arcade", "platformer", "pixel_art", "2d_flat"]},
    ]

    @st.cache_resource
    def compute_evaluation():
        from src.models.hybrid import HybridRecommender
        rec_inner = HybridRecommender(alpha=0.5, beta=0.4, gamma=0.1)
        rec_inner.fit(fm, cm)

        all_recs = []
        all_rel = []
        for profile in test_profiles:
            r = rec_inner.recommend_for_profile(profile['tags'], k=20)
            all_recs.append([x['game_id'] for x in r])
            relevant_tags = set(profile['tags'])
            rel = set()
            for row in fm.to_dicts():
                if any(tag in relevant_tags for tag in profile['tags'] if tag in row and row[tag] == 1):
                    rel.add(row['game_id'])
            all_rel.append(rel)

        ks = [1, 3, 5, 10, 20]
        metrics = {}
        for k in ks:
            precs = [precision_at_k(r, rel, k) for r, rel in zip(all_recs, all_rel)]
            recs_m = [recall_at_k(r, rel, k) for r, rel in zip(all_recs, all_rel)]
            ndcgs = [ndcg_at_k(r, rel, k) for r, rel in zip(all_recs, all_rel)]
            metrics[k] = {
                'precision': float(np.mean(precs)),
                'recall': float(np.mean(recs_m)),
                'ndcg': float(np.mean(ndcgs)),
            }

        total_cat = len(fm)
        combined_cov = coverage_metric(all_recs, total_cat)
        map20 = map_at_k(all_recs, all_rel, 20)
        hr20 = hit_rate_at_k(all_recs, all_rel, 20)

        return {
            'metrics': metrics,
            'ks': ks,
            'coverage': combined_cov,
            'map20': map20,
            'hit_rate20': hr20,
        }

    if IS_DEMO:
        st.info("📈 Full evaluation requires complete dataset. Showing demo evaluation on 500 games.")
        from src.evaluation.metrics import (
            precision_at_k, recall_at_k, ndcg_at_k, map_at_k,
            hit_rate_at_k, coverage as coverage_metric,
        )

    eval_data = compute_evaluation()

    st.markdown("### Offline Evaluation Metrics")
    st.caption("Performance across 5 synthetic user profiles (Casual, Hardcore, Story, Strategy, Retro).")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAP@20", f"{eval_data['map20']:.4f}")
    with col2:
        st.metric("Hit Rate@20", f"{eval_data['hit_rate20']:.3f}")
    with col3:
        st.metric("Coverage", f"{eval_data['coverage']*100:.1f}%")
    with col4:
        m5 = eval_data['metrics'][5]
        st.metric("NDCG@5", f"{m5['ndcg']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ['precision', 'recall', 'ndcg']:
        values = [eval_data['metrics'][k][metric] for k in eval_data['ks']]
        ax.plot(eval_data['ks'], values, marker='o', label=metric)
    ax.set_xlabel('k')
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics vs k (averaged across profiles)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    with st.expander("Full Results Table"):
        st.markdown(f"""
| k | Precision@k | Recall@k | NDCG@k |
|---|---|---|---|
""" + "\n".join(
            f"| {k} | {eval_data['metrics'][k]['precision']:.4f} | {eval_data['metrics'][k]['recall']:.4f} | {eval_data['metrics'][k]['ndcg']:.4f} |"
            for k in eval_data['ks']
        ))

    st.markdown("---")
    st.markdown("### Cold-Start Analysis")
    st.caption("How does the model perform across data confidence segments?")

    confidence_segments = [(0, 30, 'Low'), (30, 60, 'Medium'), (60, 80, 'High'), (80, 100, 'Very High')]
    all_top20 = set()
    for profile in test_profiles:
        r = rec.recommend_for_profile(profile['tags'], k=20)
        all_top20.update(x['game_id'] for x in r)

    segment_data = []
    for lo, hi, label in confidence_segments:
        mask = (fm['data_confidence_score'] >= lo) & (fm['data_confidence_score'] < hi)
        count = mask.sum()
        in_top = sum(fm.filter(mask)['game_id'].is_in(list(all_top20)))
        prop = in_top / count if count > 0 else 0
        segment_data.append({'segment': label, 'total': count, 'in_top20': in_top, 'proportion': prop})

    fig, ax = plt.subplots(figsize=(8, 4))
    segs = [d['segment'] for d in segment_data]
    props = [d['proportion'] * 100 for d in segment_data]
    bars = ax.bar(segs, props, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax.set_ylabel('% in Top-20 recommendations')
    ax.set_title('Representation in Recommendations by Confidence Segment')
    for bar, prop in zip(bars, props):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{prop:.1f}%', ha='center', fontsize=11)
    st.pyplot(fig)
    plt.close()
