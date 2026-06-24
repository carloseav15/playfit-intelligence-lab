# Changelog — Playfit Intelligence Lab

## v1.0.0 (2026-06-23) — Initial Release

### Model
- Hybrid recommender: `final_score = α·content + β·pop - γ·confidence_penalty`
- Best params from grid search: α=0.7, β=0.2, γ=0.1
- NDCG@5: **0.5691**, Precision@5: **0.56**
- LambdaRank (LightGBM) baseline available
- MMR re-ranking for diversity (`λ=0.5`)

### Catalog Diagnostics (baseline snapshot)
| Metric | Value |
|--------|-------|
| Total games | 63,682 |
| Without genre | 28,842 (60.6%) |
| Without cover | 23,366 (51%) |
| Without year | 1,295 (3.5%) |
| Without tags | 4,844 (7.6%) |
| Duplicate groups | 914 |
| External matches (conf ≥ 70) | 5,000 |
| Low confidence (< 30) | 2,602 |

### Enhancements
- [x] MMR Re-Ranking
- [x] LightGBM LambdaRank
- [x] MLflow Experiment Tracking (10 runs)
- [x] A/B Test Simulator
- [x] LLM Explainability (OpenRouter)
- [x] DuckDB Analytics
- [x] Model Card + Data Sheet
- [x] Docker + Render deploy

### Database feedback
- Created `_diagnostics` schema in games-library DB
- Tables: `quality_gaps`, `duplicate_groups`, `external_matches`, `low_confidence_priorities`, `model_card_metrics`
