# Datasheet: Playfit Game Catalog

## Motivation
- **Purpose:** The dataset was created to power a personalized game recommendation system (Playfit)
- **Creators:** Playfit project, extracted from the `games_library` schema of a Supabase PostgreSQL database
- **Funding:** Personal/open-source project
- **Date collected:** June 2026

## Composition
- **Total instances:** 63,682 games
- **Number of tables:** 27 (17 exported for this project)
- **Instance type:** Each instance is a video game with associated metadata
- **Data volume:** ~35 MB compressed (Parquet format), 636K+ total rows across all tables
- **Missing data:** Significant — 60.6% missing genre, 51% missing cover, 3.5% missing year, 95.5% missing series

### Data Dictionary (primary table: `games`)

| Column | Type | Description | Coverage |
|--------|------|-------------|----------|
| game_id | text | Unique game identifier (URL-safe slug) | 100% |
| title | text | Game title | 100% |
| release_year | integer | Year of release | 96.5% |
| genre_id | text | FK to genres table | 39.4% |
| cover_url | text | URL to cover image | 49% |
| source_type | text | Origin: catalog, universe, or finder | 100% |
| tags | text[] | Array of tag IDs (denormalized) | 94.4% |
| platforms | via game_platforms | Many-to-many join | 100% |

### External Data Sources
| Source | Type | Records | Confidence |
|--------|------|---------|------------|
| Metacritic (games) | Scores, reviews | ~35K match candidates | 0-100 |
| VGSales | Sales figures | ~10K snapshots | 0-100 |
| Metacritic (sentiment) | Review sentiment | ~2K snapshots | 0-100 |
| RAWG API | Cover art, metadata | via scraping pipeline | N/A |

## Collection Process
- **Initial import:** Data aggregated from multiple video game databases via scraping pipelines
- **Deduplication:** Automated fuzzy matching + manual review queue (914 groups, 1,162 candidates)
- **External matching:** Rule-based title normalization + confidence scoring against Metacritic and VGSales CSV exports
- **Enrichment:** Genre, series, tags, and platform normalization via separate enrichment scripts
- **Timeline:** Migrations span June 9-22, 2026

## Preprocessing
- **Data cleaning:** Duplicate group identification, redirect chain compression, slug canonicalization
- **Normalization:** Genres, platforms, series, tags extracted to normalized tables
- **Score normalization:** Scale normalization applied to game scores
- **Feature engineering:** One-hot encoding of 162 tags + 36 platforms, SVD dimensionality reduction (100 components)
- **Quality scoring:** `data_confidence_score` (0-100) computed from data availability and external match confidence

## Uses
### Intended uses
- Game recommendation systems
- Catalog quality analysis and monitoring
- Portfolio project demonstrating data science skills

### Potential inappropriate uses
- Making purchasing decisions without additional verification
- Inferring user preferences from incomplete data
- Benchmarking without accounting for selection bias

## Distribution
- **Format:** Parquet files in `data/raw/`
- **License:** Derived from the Playfit project (MIT)
- **Third-party restrictions:** Metacritic data may be subject to their terms of service; check before redistribution

## Maintenance
- **Frequency:** Not currently updated — snapshot as of June 2026
- **Feedback:** Issues can be reported via the Playfit project repository
- **Known issues:** 914 duplicate groups pending review, 35K+ match candidates need manual approval
