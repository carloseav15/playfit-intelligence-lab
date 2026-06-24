import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("SUPABASE_DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("SUPABASE_DB_PORT", "54322")),
    "dbname": os.getenv("SUPABASE_DB_NAME", "postgres"),
    "user": os.getenv("SUPABASE_DB_USER", "postgres"),
    "password": os.getenv("SUPABASE_DB_PASSWORD", "postgres"),
}

SCHEMA = os.getenv("SUPABASE_SCHEMA", "games_library")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


EXPORT_TABLES = {
    "games": f"SELECT * FROM {SCHEMA}.games",
    "game_platforms": f"SELECT * FROM {SCHEMA}.game_platforms",
    "game_tags": f"SELECT * FROM {SCHEMA}.game_tags",
    "genres": f"SELECT * FROM {SCHEMA}.genres",
    "platforms": f"SELECT * FROM {SCHEMA}.platforms",
    "tags": f"SELECT * FROM {SCHEMA}.tags",
    "game_scores": f"SELECT * FROM {SCHEMA}.game_scores",
    "game_sales_snapshots": f"SELECT * FROM {SCHEMA}.game_sales_snapshots",
    "game_review_sentiment_snapshots": f"SELECT * FROM {SCHEMA}.game_review_sentiment_snapshots",
    "game_external_ids": f"SELECT * FROM {SCHEMA}.game_external_ids",
    "game_external_match_candidates": f"SELECT * FROM {SCHEMA}.game_external_match_candidates",
    "game_duplicate_candidates": f"SELECT * FROM {SCHEMA}.game_duplicate_candidates",
    "game_duplicate_groups": f"SELECT * FROM {SCHEMA}.game_duplicate_groups",
    "user_game_states": f"SELECT * FROM {SCHEMA}.user_game_states",
    "game_recommendation_enrichment_signals": f"SELECT * FROM {SCHEMA}.game_recommendation_enrichment_signals",
    "game_aliases": f"SELECT * FROM {SCHEMA}.game_aliases",
    "game_redirects": f"SELECT * FROM {SCHEMA}.game_redirects",
}


def export_table(name: str, query: str) -> dict:
    print(f"  Exporting {name}...")
    conn = get_conn()
    try:
        df = pd.read_sql(query, conn)
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return {"table": name, "rows": len(df), "columns": list(df.columns), "file": str(path)}
    finally:
        conn.close()


def export_all() -> list[dict]:
    manifest = []
    for name, query in EXPORT_TABLES.items():
        info = export_table(name, query)
        manifest.append(info)
    import json
    manifest_path = RAW_DIR / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    print(f"Exporting {len(EXPORT_TABLES)} tables from {SCHEMA}...")
    manifest = export_all()
    total = sum(m["rows"] for m in manifest)
    print(f"Done. Exported {total:,} total rows across {len(manifest)} tables.")
