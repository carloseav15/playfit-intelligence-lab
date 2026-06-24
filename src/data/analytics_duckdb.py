from pathlib import Path

import duckdb
import polars as pl

RAW_DIR = Path("data/raw")


class DuckDBAnalytics:
    def __init__(self):
        self.con = duckdb.connect()

    def query(self, sql: str) -> pl.DataFrame:
        return pl.from_arrow(self.con.execute(sql).fetch_arrow_table())

    def games_by_year(self) -> pl.DataFrame:
        return self.query("""
            SELECT release_year, count(*) as game_count
            FROM read_parquet('data/raw/games.parquet')
            WHERE release_year IS NOT NULL AND release_year > 0
            GROUP BY release_year
            ORDER BY release_year
        """)

    def top_platforms(self, n: int = 10) -> pl.DataFrame:
        return self.query(f"""
            SELECT p.name, count(*) as game_count
            FROM read_parquet('data/raw/game_platforms.parquet') gp
            JOIN read_parquet('data/raw/platforms.parquet') p ON gp.platform_id = p.id
            GROUP BY p.name
            ORDER BY game_count DESC
            LIMIT {n}
        """)

    def avg_confidence_by_genre(self) -> pl.DataFrame:
        return self.query("""
            SELECT g.name as genre,
                   avg(s.data_confidence_score) as avg_confidence,
                   count(*) as game_count
            FROM read_parquet('data/raw/game_recommendation_enrichment_signals.parquet') s
            JOIN read_parquet('data/raw/games.parquet') ga ON s.game_id = ga.game_id
            LEFT JOIN read_parquet('data/raw/genres.parquet') g ON ga.genre_id = g.id
            WHERE ga.genre_id IS NOT NULL
            GROUP BY g.name
            ORDER BY avg_confidence DESC
        """)

    def ranked_games_by_platform(self) -> pl.DataFrame:
        return self.query("""
            SELECT g.title, p.name as platform, s.best_critic_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY gp.platform_id
                       ORDER BY s.best_critic_score DESC NULLS LAST
                   ) as rank_in_platform
            FROM read_parquet('data/raw/games.parquet') g
            JOIN read_parquet('data/raw/game_platforms.parquet') gp ON g.game_id = gp.game_id
            JOIN read_parquet('data/raw/platforms.parquet') p ON gp.platform_id = p.id
            LEFT JOIN read_parquet('data/raw/game_recommendation_enrichment_signals.parquet') s
                ON g.game_id = s.game_id
            WHERE s.best_critic_score IS NOT NULL
            ORDER BY platform, rank_in_platform
            LIMIT 50
        """)

    def coverage_pivot(self) -> pl.DataFrame:
        return self.query("""
            SELECT
                COUNT(*) as total_games,
                SUM(CASE WHEN genre_id IS NOT NULL THEN 1 ELSE 0 END) as with_genre,
                SUM(CASE WHEN cover_url != '' THEN 1 ELSE 0 END) as with_cover,
                SUM(CASE WHEN release_year IS NOT NULL AND release_year > 0 THEN 1 ELSE 0 END) as with_year,
                COUNT(DISTINCT game_id) as unique_games
            FROM read_parquet('data/raw/games.parquet')
        """)

    def duplicate_summary(self) -> pl.DataFrame:
        return self.query("""
            SELECT status, suggested_review, count(*) as group_count,
                   SUM(candidate_count) as total_candidates
            FROM read_parquet('data/raw/game_duplicate_groups.parquet')
            GROUP BY status, suggested_review
            ORDER BY group_count DESC
        """)

    def tag_popularity_by_year_window(self) -> pl.DataFrame:
        return self.query("""
            SELECT g.release_year, gt.tag_id,
                   COUNT(*) OVER (
                       PARTITION BY gt.tag_id
                       ORDER BY g.release_year
                       ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                   ) as rolling_5yr_count
            FROM read_parquet('data/raw/game_tags.parquet') gt
            JOIN read_parquet('data/raw/games.parquet') g ON gt.game_id = g.game_id
            WHERE g.release_year IS NOT NULL AND g.release_year > 0
            ORDER BY g.release_year
            LIMIT 100
        """)

    def approximate_tag_diversity(self) -> pl.DataFrame:
        return self.query("""
            SELECT tag_id,
                   COUNT(DISTINCT game_id) as exact_game_count,
                   approx_count_distinct(game_id) as approx_game_count
            FROM read_parquet('data/raw/game_tags.parquet')
            GROUP BY tag_id
            ORDER BY exact_game_count DESC
        """)

    def close(self):
        self.con.close()
