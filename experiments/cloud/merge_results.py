"""Merge per-task parquets from multiple run directories into one file.

Typical use: combine CPU-VM results (non-NN models) with GPU-VM results
(NN models) so that analysis scripts see a single results parquet.

Usage:
    uv run python experiments/cloud/merge_results.py \
        experiments/results/run_results_binary_<ts1> \
        experiments/results/run_results_binary_<ts2> \
        -o experiments/results/results_binary_merged.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEDUP_KEYS = [
    "dataset_name",
    "model_name",
    "strategy_name",
    "optimization_objective_name",
    "random_seed",
]


def merge(run_dirs: list[Path], output: Path) -> pd.DataFrame:
    """Read parquets from dirs or files, concat, deduplicate, write."""
    frames: list[pd.DataFrame] = []
    reference_cols: set[str] | None = None
    reference_src: str | None = None

    def _add(frame: pd.DataFrame, src: str) -> None:
        nonlocal reference_cols, reference_src
        cols = set(frame.columns)
        if reference_cols is None:
            reference_cols, reference_src = cols, src
        elif cols != reference_cols:
            # CPU-VM and GPU-VM runs are merged here; a schema drift
            # between them would silently NaN-fill the gap columns and
            # quietly corrupt downstream analysis. Make it loud.
            print(
                f"  SCHEMA MISMATCH: {src} columns differ from "
                f"{reference_src} (missing: "
                f"{sorted(reference_cols - cols)}; extra: "
                f"{sorted(cols - reference_cols)}). Concatenating "
                f"anyway -- gap columns become NaN."
            )
        frames.append(frame)

    for d in run_dirs:
        if d.is_file() and d.suffix == ".parquet":
            _add(pd.read_parquet(d), d.name)
            print(f"  {d.name}: single file")
            continue
        parts = sorted(d.glob("*.parquet"))
        if not parts:
            print(f"  warning: no parquet files in {d}")
            continue
        for p in parts:
            _add(pd.read_parquet(p), f"{d.name}/{p.name}")
        print(f"  {d.name}: {len(parts)} files")

    if not frames:
        raise SystemExit("No parquet files found in any of the input dirs.")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    after = len(df)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print(
        f"\nMerged {before} rows -> {after} (dropped {before - after} dupes)"
    )
    print(f"Models:   {sorted(df['model_name'].unique())}")
    print(f"Datasets: {sorted(df['dataset_name'].unique())}")
    print(f"Seeds:    {sorted(df['random_seed'].unique())}")
    print(f"Written:  {output}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge parquets from multiple experiment run directories.",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directories or parquet files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path for the merged parquet.",
    )
    args = parser.parse_args()

    for d in args.run_dirs:
        if not d.exists():
            raise SystemExit(f"Not found: {d}")

    merge(args.run_dirs, args.output)


if __name__ == "__main__":
    main()
