"""
scripts/snapshot_diagnostics.py — Toma un snapshot del schema _diagnostics y lo guarda
en docs/diagnostics_snapshots/ con timestamp. Permite comparar mejoras antes/después.

Uso:
    python3 scripts/snapshot_diagnostics.py
    python3 scripts/snapshot_diagnostics.py --diff    # comparar con snapshot anterior

Salida:
    docs/diagnostics_snapshots/2026-06-23_21-00-00/
        quality_gaps.csv
        duplicate_groups.csv
        external_matches.csv
        low_confidence_priorities.csv
        model_card_metrics.csv
        summary.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import psycopg2
import polars as pl

DB_URI = "postgresql://postgres:postgres@localhost:54322/postgres"
SNAPSHOTS_DIR = Path("docs/diagnostics_snapshots")


def get_conn():
    return psycopg2.connect(DB_URI)


def fetch_table(conn, table: str) -> pl.DataFrame:
    import warnings; warnings.filterwarnings("ignore")
    if table == "quality_gaps":
        return pl.read_database(
            "SELECT game_id, title, array_to_string(missing_issues, ',') AS missing_issues, "
            "data_confidence_score, popularity_score FROM _diagnostics.quality_gaps",
            conn,
        )
    return pl.read_database(f"SELECT * FROM _diagnostics.{table}", conn)


def take_snapshot() -> Path:
    conn = get_conn()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = SNAPSHOTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = [
        "quality_gaps",
        "duplicate_groups",
        "external_matches",
        "low_confidence_priorities",
        "model_card_metrics",
    ]

    summary = {}
    for table in tables:
        try:
            df = fetch_table(conn, table)
            path = out_dir / f"{table}.csv"
            df.write_csv(path)
            summary[table] = {
                "rows": len(df),
                "columns": df.columns,
                "path": str(path),
            }
            print(f"  {table}: {len(df)} rows → {path}")
        except Exception as e:
            print(f"  {table}: SKIPPED ({e})")
            summary[table] = {"rows": 0, "error": str(e)}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    conn.close()
    print(f"\nSnapshot saved to {out_dir}")
    return out_dir


def diff_snapshots():
    snapshots = sorted(SNAPSHOTS_DIR.iterdir())
    if len(snapshots) < 2:
        print("Need at least 2 snapshots to diff. Run without --diff first.")
        return

    prev = snapshots[-2]
    curr = snapshots[-1]
    print(f"Comparing {prev.name} → {curr.name}")

    for table in ["quality_gaps", "low_confidence_priorities"]:
        prev_file = prev / f"{table}.csv"
        curr_file = curr / f"{table}.csv"
        if not prev_file.exists() or not curr_file.exists():
            continue

        prev_df = pl.read_csv(prev_file)
        curr_df = pl.read_csv(curr_file)
        delta = len(prev_df) - len(curr_df)

        if table == "quality_gaps":
            # Count by issue type
            for issue in ["no_genre", "no_cover", "no_year", "no_tags", "no_platforms"]:
                prev_count = prev_df.filter(pl.col("missing_issues").str.contains(issue)).height
                curr_count = curr_df.filter(pl.col("missing_issues").str.contains(issue)).height
                diff = prev_count - curr_count
                sign = "+" if diff > 0 else ""
                print(f"  {issue}: {prev_count} → {curr_count} ({sign}{diff})")
        else:
            print(f"  {table}: {len(prev_df)} → {len(curr_df)} rows ({delta:+,} delta)")


def main():
    parser = argparse.ArgumentParser(description="Snapshot _diagnostics schema")
    parser.add_argument("--diff", action="store_true", help="Compare with last snapshot")
    args = parser.parse_args()

    if args.diff:
        diff_snapshots()
    else:
        take_snapshot()


if __name__ == "__main__":
    main()
