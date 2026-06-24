from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_matrix() -> pl.DataFrame:
    games = pl.read_parquet(RAW_DIR / "games.parquet")
    signals = pl.read_parquet(RAW_DIR / "game_recommendation_enrichment_signals.parquet")

    tags_wide = _pivot_tags()
    platforms_wide = _pivot_platforms()

    features = games.select([
        "game_id",
        "title",
        "release_year",
        "genre_id",
    ]).join(tags_wide, on="game_id", how="left").join(platforms_wide, on="game_id", how="left")

    for col in features.columns:
        if col not in {"game_id", "title", "release_year", "genre_id"}:
            features = features.with_columns(
                pl.col(col).fill_null(0)
            )

    sig = signals.select([
        "game_id",
        "best_critic_score",
        "best_user_score",
        "critic_review_count",
        "user_review_count",
        "critic_positive_ratio",
        "user_positive_ratio",
        "max_global_sales_millions",
        "total_global_sales_millions",
        "data_confidence_score",
        "has_external_id",
        "has_company",
        "has_age_rating",
        "has_summary",
        "has_sales",
        "has_review_sentiment",
    ])

    features = features.join(sig, on="game_id", how="left")
    features = features.with_columns([
        pl.col("best_critic_score").fill_null(pl.median("best_critic_score")),
        pl.col("best_user_score").fill_null(pl.median("best_user_score")),
        pl.col("critic_review_count").fill_null(0),
        pl.col("user_review_count").fill_null(0),
        pl.col("critic_positive_ratio").fill_null(0.5),
        pl.col("user_positive_ratio").fill_null(0.5),
        pl.col("max_global_sales_millions").fill_null(0),
        pl.col("total_global_sales_millions").fill_null(0),
        pl.col("data_confidence_score").fill_null(50),
        pl.col("has_external_id").fill_null(False),
        pl.col("has_company").fill_null(False),
        pl.col("has_age_rating").fill_null(False),
        pl.col("has_summary").fill_null(False),
        pl.col("has_sales").fill_null(False),
        pl.col("has_review_sentiment").fill_null(False),
        pl.col("release_year").fill_null(0).cast(pl.Int32),
    ])

    features = features.with_columns([
        (pl.col("total_global_sales_millions") + 0.001).log10().alias("log_sales"),
        ((pl.col("critic_review_count") + pl.col("user_review_count"))).alias("total_review_count"),
    ])

    features.write_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    return features


def _pivot_tags() -> pl.DataFrame:
    tags = pl.read_parquet(RAW_DIR / "game_tags.parquet")
    tag_catalog = pl.read_parquet(RAW_DIR / "tags.parquet")
    tags = tags.join(tag_catalog.select(["id"]), left_on="tag_id", right_on="id", how="inner")
    tags = tags.with_columns(pl.lit(1).alias("val"))
    tag_wide = tags.pivot(
        index="game_id",
        columns="tag_id",
        values="val",
        aggregate_function="first",
    ).fill_null(0)
    return tag_wide


def _pivot_platforms() -> pl.DataFrame:
    platforms = pl.read_parquet(RAW_DIR / "game_platforms.parquet")
    platforms = platforms.with_columns(pl.lit(1).alias("val"))
    pivot = platforms.pivot(
        index="game_id",
        columns="platform_id",
        values="val",
        aggregate_function="first",
    ).fill_null(0)
    return pivot


def compute_popularity_score(features: pl.DataFrame | None = None) -> pl.DataFrame:
    if features is None:
        features = pl.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    score = (
        features["best_critic_score"].fill_null(0) * 0.25
        + features["best_user_score"].fill_null(0) * 0.20
        + features["log_sales"].fill_null(0) * 0.15
        + features["critic_positive_ratio"].fill_null(0.5) * 0.15
        + features["user_positive_ratio"].fill_null(0.5) * 0.10
        + (features["data_confidence_score"] / 100) * 0.15
    )
    return features.with_columns(score.alias("popularity_score"))


def compute_richness_score(features: pl.DataFrame | None = None) -> pl.DataFrame:
    if features is None:
        features = pl.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    richness = (
        features["has_external_id"].cast(pl.Int32) * 0.20
        + features["has_company"].cast(pl.Int32) * 0.15
        + features["has_age_rating"].cast(pl.Int32) * 0.15
        + features["has_summary"].cast(pl.Int32) * 0.15
        + features["has_sales"].cast(pl.Int32) * 0.15
        + features["has_review_sentiment"].cast(pl.Int32) * 0.20
    )
    return features.with_columns(richness.alias("richness_score"))
