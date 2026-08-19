from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")


def make_explanation(rec: dict,
                     feature_matrix: pl.DataFrame | None = None) -> str:
    parts = []
    game_id = rec["game_id"]

    if rec.get("content_score", 0) > 0.05:
        parts.append("matches the tags/genre of games you like")
    if rec.get("popularity_score", 0) > 0.5:
        parts.append("high global popularity")
    if rec.get("data_confidence", 50) >= 70:
        parts.append(f"high data confidence ({rec['data_confidence']}/100)")
    elif rec.get("data_confidence", 50) < 40:
        parts.append("limited data - check external reviews")
    if rec.get("confidence_penalty", 0) > 0.02:
        parts.append("penalized for incomplete quality data")

    if not parts:
        return "recommended by general popularity"

    return " · ".join(parts)


def get_game_details(game_id: str) -> dict:
    games = pl.read_parquet(RAW_DIR / "games.parquet")
    row = games.filter(pl.col("game_id") == game_id)
    if len(row) == 0:
        return {}
    r = row.to_dicts()[0]
    platforms = pl.read_parquet(RAW_DIR / "game_platforms.parquet")
    platform_ids = platforms.filter(
        pl.col("game_id") == game_id
    )["platform_id"].to_list()
    platform_catalog = pl.read_parquet(RAW_DIR / "platforms.parquet")
    platform_names = platform_catalog.filter(
        pl.col("id").is_in(platform_ids)
    )["name"].to_list()

    return {
        "title": r.get("title", ""),
        "year": r.get("release_year", ""),
        "genre": r.get("genre_id", ""),
        "platforms": platform_names,
    }
