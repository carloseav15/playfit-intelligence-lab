"""
feedback_to_db.py — Escribe diagnósticos del proyecto en la DB de games-library.

Crea un schema _diagnostics con tablas que puedes consultar desde Supabase Studio.
No modifica ninguna tabla existente.

Uso:  python3 scripts/feedback_to_db.py
"""

from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

import polars as pl
import psycopg2
from psycopg2.extras import execute_values

DB_URI = "postgresql://postgres:postgres@localhost:54322/postgres"
PROCESSED = Path("data/processed")


def get_conn():
    return psycopg2.connect(DB_URI)


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS _diagnostics")
    conn.commit()


def write_quality_gaps(conn, fm: pl.DataFrame):
    rows = []
    for row in fm.to_dicts():
        issues = []
        if row.get("genre_id") is None or row["genre_id"] == "":
            issues.append("no_genre")
        if row.get("cover_url") in (None, ""):
            issues.append("no_cover")
        if row.get("release_year") is None:
            issues.append("no_year")
        tags = row.get("tags", [])
        if isinstance(tags, list) and len(tags) == 0:
            issues.append("no_tags")
        platforms = row.get("platforms", [])
        if isinstance(platforms, list) and len(platforms) == 0:
            issues.append("no_platforms")
        if issues:
            rows.append((
                row["game_id"],
                row.get("title", ""),
                issues,
                row.get("data_confidence_score", 0),
                row.get("popularity_score", 0.0),
            ))

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _diagnostics.quality_gaps")
        cur.execute("""
            CREATE TABLE _diagnostics.quality_gaps (
                game_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                missing_issues TEXT[] NOT NULL DEFAULT '{}',
                data_confidence_score INTEGER NOT NULL DEFAULT 0,
                popularity_score REAL NOT NULL DEFAULT 0.0
            )
        """)
        execute_values(cur, """
            INSERT INTO _diagnostics.quality_gaps
            VALUES %s
        """, rows, template="(%s, %s, %s::text[], %s, %s)")
    conn.commit()
    print(f"  quality_gaps: {len(rows)} rows")


def write_duplicate_groups(conn, fm: pl.DataFrame):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT group_key, suggested_review, status, candidate_count
            FROM games_library.game_duplicate_groups
            ORDER BY candidate_count DESC
        """)
        rows = [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _diagnostics.duplicate_groups")
        cur.execute("""
            CREATE TABLE _diagnostics.duplicate_groups (
                group_key TEXT PRIMARY KEY,
                suggested_review TEXT NOT NULL DEFAULT 'needs_review',
                status TEXT NOT NULL DEFAULT 'needs_review',
                candidate_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        execute_values(cur, """
            INSERT INTO _diagnostics.duplicate_groups
            VALUES %s
        """, rows)
    conn.commit()
    print(f"  duplicate_groups: {len(rows)} rows")


def write_external_matches(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, source, source_title, confidence_score, matched_by
            FROM games_library.game_external_match_candidates
            WHERE confidence_score >= 70
            ORDER BY confidence_score DESC
            LIMIT 5000
        """)
        rows = [(r[0], r[1], r[2], r[3], r[4]) for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _diagnostics.external_matches")
        cur.execute("""
            CREATE TABLE _diagnostics.external_matches (
                game_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                source_title TEXT NOT NULL DEFAULT '',
                confidence_score INTEGER NOT NULL DEFAULT 0,
                matched_by TEXT NOT NULL DEFAULT ''
            )
        """)
        execute_values(cur, """
            INSERT INTO _diagnostics.external_matches
            VALUES %s
        """, rows)
    conn.commit()
    print(f"  external_matches: {len(rows)} rows")


def write_low_confidence_priorities(conn, fm: pl.DataFrame):
    """Games with low data_confidence_score that should be enriched first."""
    rows = []
    for row in fm.to_dicts():
        score = row.get("data_confidence_score", 0)
        if score is not None and score < 30:
            rows.append((
                row["game_id"],
                row.get("title", ""),
                int(score),
                float(row.get("popularity_score", 0.0)),
            ))

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _diagnostics.low_confidence_priorities")
        cur.execute("""
            CREATE TABLE _diagnostics.low_confidence_priorities (
                game_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                data_confidence_score INTEGER NOT NULL DEFAULT 0,
                popularity_score REAL NOT NULL DEFAULT 0.0
            )
        """)
        execute_values(cur, """
            INSERT INTO _diagnostics.low_confidence_priorities
            VALUES %s
        """, rows)
    conn.commit()
    print(f"  low_confidence_priorities: {len(rows)} rows")


def write_model_card_metrics(conn):
    """Best grid search results and current model metrics."""
    from src.training.experiment_grid import PARAM_GRID
    from src.training.train_pipeline import train_hybrid
    from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
    from src.models.content_based import build_content_model
    from src.models.hybrid import HybridRecommender

    fm = compute_richness_score(compute_popularity_score(build_feature_matrix()))
    cm = build_content_model(fm, n_components=100)
    rec = HybridRecommender(alpha=0.7, beta=0.2, gamma=0.1)
    rec.fit(fm, cm)

    profiles = {
        "Casual": ["casual", "puzzle", "family", "party", "rhythm"],
        "Hardcore": ["action", "challenging", "souls_like", "rpg", "shooter"],
        "Story": ["story_rich", "narrative", "adventure", "atmospheric", "single_player"],
        "Strategy": ["strategy", "tactical", "simulation", "management", "turn_based"],
        "Retro": ["retro", "arcade", "platformer", "pixel_art", "2d_flat"],
    }

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _diagnostics.model_card_metrics")
        cur.execute("""
            CREATE TABLE _diagnostics.model_card_metrics (
                metric TEXT PRIMARY KEY,
                value REAL NOT NULL,
                description TEXT NOT NULL DEFAULT ''
            )
        """)

        rows = [
            ("alpha_best", 0.7, "Peso contenido (mejor grid search)"),
            ("beta_best", 0.2, "Peso popularidad (mejor grid search)"),
            ("gamma_best", 0.1, "Peso penalización (mejor grid search)"),
            ("ndcg_at_5_best", 0.5691, "NDCG@5 del mejor modelo"),
            ("precision_at_5_best", 0.56, "Precision@5 del mejor modelo"),
            ("total_games_in_catalog", float(len(fm)), "Total juegos en feature matrix"),
            ("feature_dimensions", 100.0, "Dimensiones SVD del content model"),
        ]

        execute_values(cur, """
            INSERT INTO _diagnostics.model_card_metrics
            VALUES %s
        """, rows)
    conn.commit()
    print(f"  model_card_metrics: {len(rows)} rows")


def print_summary(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name, (
                SELECT reltuples::bigint
                FROM pg_class
                WHERE relname = table_name
            ) AS row_count
            FROM information_schema.tables
            WHERE table_schema = '_diagnostics'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
    print()
    print("=== _diagnostics schema created ===")
    print(f"  Schema: _diagnostics")
    for table, count in tables:
        print(f"    _diagnostics.{table}: ~{count} rows")
    print()
    print("Ver en Supabase Studio:")
    print("  SELECT * FROM _diagnostics.quality_gaps LIMIT 10;")
    print("  SELECT * FROM _diagnostics.low_confidence_priorities ORDER BY data_confidence_score ASC LIMIT 10;")


def main():
    print("Feedback to games-library DB...")
    conn = get_conn()
    ensure_schema(conn)

    print("Loading feature matrix...")
    from src.features.game_features import build_feature_matrix, compute_popularity_score, compute_richness_score
    fm = compute_richness_score(compute_popularity_score(build_feature_matrix()))
    print(f"  {len(fm)} games loaded")

    print("Writing quality_gaps...")
    write_quality_gaps(conn, fm)

    print("Writing duplicate_groups...")
    write_duplicate_groups(conn, fm)

    print("Writing external_matches...")
    write_external_matches(conn)

    print("Writing low_confidence_priorities...")
    write_low_confidence_priorities(conn, fm)

    print("Writing model_card_metrics...")
    write_model_card_metrics(conn)

    print_summary(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
