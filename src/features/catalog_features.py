from pathlib import Path

import pandas as pd
import polars as pl

RAW_DIR = Path("data/raw")


def load_games() -> pl.DataFrame:
    return pl.read_parquet(RAW_DIR / "games.parquet")


def load_signals() -> pl.DataFrame:
    return pl.read_parquet(RAW_DIR / "game_recommendation_enrichment_signals.parquet")


def load_platforms() -> pl.DataFrame:
    return pl.read_parquet(RAW_DIR / "game_platforms.parquet")


def load_tags() -> pl.DataFrame:
    return pl.read_parquet(RAW_DIR / "game_tags.parquet")


def load_duplicates() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet(RAW_DIR / "game_duplicate_groups.parquet"),
        pl.read_parquet(RAW_DIR / "game_duplicate_candidates.parquet"),
    )


def load_external_match() -> pl.DataFrame:
    return pl.read_parquet(RAW_DIR / "game_external_match_candidates.parquet")


def coverage_analysis(games: pl.DataFrame | None = None) -> dict:
    if games is None:
        games = load_games()
    total = len(games)
    return {
        "total_games": total,
        "no_genre": (games["genre_id"].is_null().sum() / total * 100),
        "no_cover": ((games["cover_url"] == "") | games["cover_url"].is_null()).sum() / total * 100,
        "no_year": games["release_year"].is_null().sum() / total * 100,
        "no_tags": games["tags"].list.len().is_null().sum() / total * 100,
        "avg_tags_per_game": games["tags"].list.len().mean(),
        "no_title": (games["title"] == "").sum() / total * 100,
        "source_distribution": games.group_by("source_type").len().to_dict(as_series=False),
        "release_state_distribution": games.group_by("release_state").len().to_dict(as_series=False),
    }


def signal_quality_analysis(signals: pl.DataFrame | None = None) -> dict:
    if signals is None:
        signals = load_signals()
    confidence = signals["data_confidence_score"]
    return {
        "avg_confidence": confidence.mean(),
        "median_confidence": confidence.median(),
        "min_confidence": confidence.min(),
        "max_confidence": confidence.max(),
        "pct_high_confidence": (confidence >= 70).sum() / len(signals) * 100,
        "pct_medium_confidence": ((confidence >= 40) & (confidence < 70)).sum() / len(signals) * 100,
        "pct_low_confidence": (confidence < 40).sum() / len(signals) * 100,
        "has_scores_pct": (signals["best_critic_score"].is_not_null()).sum() / len(signals) * 100,
        "has_sales_pct": (signals["has_sales"] == True).sum() / len(signals) * 100,
        "has_sentiment_pct": (signals["has_review_sentiment"] == True).sum() / len(signals) * 100,
        "has_external_id_pct": (signals["has_external_id"] == True).sum() / len(signals) * 100,
        "has_company_pct": (signals["has_company"] == True).sum() / len(signals) * 100,
        "has_summary_pct": (signals["has_summary"] == True).sum() / len(signals) * 100,
    }


def duplicate_analysis() -> dict:
    groups, candidates = load_duplicates()
    return {
        "total_groups": len(groups),
        "total_candidates": len(candidates),
        "status_distribution": groups.group_by("status").len().to_dict(as_series=False),
        "review_distribution": groups.group_by("suggested_review").len().to_dict(as_series=False),
        "groups_with_diff_years": (groups["known_year_count"] > 1).sum(),
        "max_candidates_per_group": groups["candidate_count"].max(),
    }


def external_match_analysis(matches: pl.DataFrame | None = None) -> dict:
    if matches is None:
        matches = load_external_match()
    return {
        "total_matches": len(matches),
        "avg_confidence": matches["confidence_score"].mean(),
        "status_distribution": matches.group_by("status").len().to_dict(as_series=False),
        "source_distribution": matches.group_by("source").len().to_dict(as_series=False),
        "high_confidence_approved": ((matches["confidence_score"] >= 70) & (matches["status"] == "auto_approved")).sum(),
        "needs_review": (matches["status"] == "needs_review").sum(),
    }
