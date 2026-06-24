# Model Card: Playfit Hybrid Recommender

## Model Details
- **Name:** Playfit Hybrid Recommender
- **Version:** 1.0
- **Type:** Hybrid Recommendation System (Content-Based + Popularity Priors + Confidence Penalty)
- **Date:** June 2026
- **Authors:** Playfit Intelligence Lab

### Model Architecture
`final_score = α × content_similarity + β × popularity_prior - γ × confidence_penalty`

- **Content similarity:** Cosine similarity on SVD-100 embedding of 149 binary features (tags + platforms)
- **Popularity prior:** Composite score from normalized critic/user scores, sales, and sentiment ratios
- **Confidence penalty:** `(100 - data_confidence_score) / 100 × γ`

### Variants
| Variant | α | β | γ | Use Case |
|---------|---|---|---|----------|
| Balanced | 0.5 | 0.4 | 0.1 | Default |
| Content-first | 0.7 | 0.2 | 0.1 | Strong taste signal |
| Popularity-first | 0.3 | 0.6 | 0.1 | New users, cold-start |
| Conservative | 0.4 | 0.4 | 0.2 | High-quality data only |
| Diversity MMR | 0.5 | 0.4 | 0.1 | +MMR λ=0.5 re-ranking |

### Alternative Model
- **LambdaRank (LightGBM):** Gradient-boosted ranking model trained on popularity-derived relevance labels with `objective="lambdarank"`. Used as comparison baseline.

## Intended Use
- **Primary:** Generate personalized game recommendations for Playfit users
- **Secondary:** Cold-start recommendations for new users with no interaction history
- **Not intended for:** High-stakes decision making, minors without parental guidance, or as the sole discovery mechanism

## Factors
- **Data quality:** 60.6% of catalog has no genre, 51% has no cover, ~45% has critic scores
- **Cold-start:** Games with no tags/platforms/scores rely entirely on popularity prior
- **Temporal bias:** Recommender has no decay factor — older games may be over-represented
- **Platform bias:** PS2 (5,371), PS1 (5,354), Switch (4,514) dominate; less represented platforms may not appear in recommendations

## Metrics

### Primary
| Metric | Balanced (0.5,0.4,0.1) | Content-first (0.7,0.2,0.1) | Pop-first (0.3,0.6,0.1) | Best (0.7,0.2,0.1) |
|--------|--------------------------|-----------------------------|-------------------------|-------------------|
| Precision@5 | 0.3200 | **0.5600** | 0.2800 | **0.5600** |
| Recall@5 | 0.0132 | **0.0231** | 0.0116 | **0.0231** |
| NDCG@5 | 0.3059 | **0.5691** | 0.2555 | **0.5691** |
| MAP@20 | 0.2753 | **0.4975** | 0.2316 | **0.4975** |
| Hit Rate@20 | 0.3753 | **0.6422** | 0.3210 | **0.6422** |

### Secondary
| Metric | Value |
|--------|-------|
| Coverage | 0.08% (50 games / 63,682) |
| Novelty | 3.12 bits |
| Diversity@5 | 0.3576 (ILS on top-10) |
| Cold-start (low confidence) | ~3% of top-20 are low-confidence games |

### Grid Search (ranked by NDCG@5)
| α | β | γ | Precision@5 | NDCG@5 | Coverage |
|---|---|---|---|---|---|
| 0.70 | 0.20 | 0.10 | **0.56** | **0.5691** | 0.079% |
| 0.60 | 0.20 | 0.20 | 0.48 | 0.4856 | 0.068% |
| 0.60 | 0.30 | 0.10 | 0.40 | 0.3661 | 0.057% |
| 0.50 | 0.30 | 0.20 | 0.36 | 0.3322 | 0.055% |
| 0.50 | 0.35 | 0.15 | 0.36 | 0.3322 | 0.050% |
| 0.50 | 0.40 | 0.10 | 0.32 | 0.3059 | 0.046% |
| 0.40 | 0.50 | 0.10 | 0.28 | 0.2602 | 0.044% |

*Grid search over 10 α/β/γ combinations — best model: α=0.7, β=0.2, γ=0.1*

## Evaluation Data
- **5 synthetic user profiles:** Casual, Hardcore, Story Seeker, Strategy Fan, Retro Player
- **Relevance definition:** A game is relevant if it has at least one tag matching the user profile
- **Cold-start analysis:** Segmented by data_confidence_score quartiles

## Training Data
- **Source:** Playfit Supabase PostgreSQL catalog (`games_library` schema)
- **Size:** 63,682 games, 27 tables, 164K+ game-tag associations
- **Features:** 149 binary features (tags + platforms), 22 numeric features (scores, sales, confidence, sentiment)
- **Labels:** No explicit user feedback — relevance derived from tag overlap and popularity score

## Ethical Considerations
- The model may reinforce popularity bias — popular games are more likely to be recommended regardless of user taste
- Games with incomplete metadata are systematically under-recommended
- No demographic or sensitive attributes are used in the model
- The recommendation is explainable via rule-based or LLM-generated text

## Caveats and Recommendations
- Metrics are computed on synthetic profiles, not real user behavior — real-world performance may differ
- Consider implementing A/B testing before production deployment
- Add implicit feedback signals (play time, clicks, dwell time) for improved personalization
- Periodically re-evaluate coverage and diversity metrics to detect popularity drift
