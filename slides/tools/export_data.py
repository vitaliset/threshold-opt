"""Export the numbers behind the interactive result charts of the HTML deck.

Writes two JavaScript files under slides/figures/data/ (plain scripts,
because a page opened from file:// cannot fetch JSON):

* win_rates.js: the share of cells won or tied by each strategy, overall
  and restricted to one base classifier or one objective, with 95%
  bootstrap half-widths. It calls the aggregation of
  experiments.analysis.generate_win_counts, the code behind the thesis
  figures, with the same seed, so the overall slice equals the bars of
  slides/figures/tex/win_bars_*.tex.
* scaling_variance.js: the curves of tex/figures/scaling_variance.tex, read
  from the coordinates in that generated file.
* datasets.js: name, sample count, number of classes and imbalance ratio of
  the benchmark datasets, read from tex/tables/dataset_summary_*.tex.

Needs pandas and the published result files (make fetch-results).

Usage: python slides/tools/export_data.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.analysis import generate_win_counts as gwc  # noqa: E402

OUT = ROOT / "slides" / "figures" / "data"
N_BOOT, SEED = 1000, 0


def slice_stats(
    table: pd.DataFrame, strats: list[str]
) -> dict[str, list[float]]:
    """Rate, bootstrap half-width (both in %) and cell count per strategy."""
    W, A = gwc._slice_arrays(table, strats)
    lo, hi = gwc._bootstrap_lo_hi(W, A, np.random.default_rng(SEED), N_BOOT)
    n = A.sum(axis=0)
    out = {}
    for j, s in enumerate(strats):
        if n[j]:
            out[s] = [
                round(100 * W[:, j].sum() / n[j], 3),
                round(100 * (hi[j] - lo[j]) / 2, 3),
                int(n[j]),
            ]
    return out


def win_rates() -> dict:
    data = {}
    for scope in ("binary", "multiclass"):
        cfg = gwc.SCOPE_CONFIG[scope]
        path = gwc._find_latest_parquet(cfg["parquet_pattern"])
        print(f"Reading: {path}")
        df = pd.read_parquet(path)
        df = gwc._apply_legacy_strategy_renames(df)
        df = gwc._normalize_balanced_error_objective(df)
        df = df[df["dataset_name"].isin(cfg["datasets"])]
        max_metrics = cfg["max_metrics"]
        for family, objs in (
            ("cost", cfg["cost_objectives"]),
            ("max", max_metrics),
        ):
            sub = df[df["optimization_objective_name"].isin(objs)]
            table = gwc.compute_win_table(sub, max_metrics)
            strats = sorted(table["strategy_name"].unique())
            overall = slice_stats(table, strats)
            shown = [
                s
                for s in gwc.SLIDE_BAR_STRATEGIES[(scope, family)]
                if s in overall
            ]
            shown.sort(key=lambda s: -overall[s][0])
            models = [
                m for m in gwc.MODEL_ORDER if m in set(table["model_name"])
            ]
            objectives = [
                o
                for o in objs
                if o in set(table["optimization_objective_name"])
            ]
            slices = {"all": overall}
            for m in models:
                slices[f"model:{m}"] = slice_stats(
                    table[table["model_name"] == m], strats
                )
            for o in objectives:
                slices[f"objective:{o}"] = slice_stats(
                    table[table["optimization_objective_name"] == o], strats
                )
            cells = table[
                ["dataset_name", "model_name", "optimization_objective_name"]
            ].drop_duplicates()
            data[f"{scope}_{family}"] = {
                "shown": shown,
                "family": {s: gwc._strategy_family(s) for s in strats},
                "models": models,
                "objectives": objectives,
                "cells": len(cells),
                # Every strategy is kept: the paired charts use some that the bars do not show.
                "slices": slices,
            }
            print(
                f"  {scope} {family}: {len(cells)} cells; "
                + ", ".join(f"{s} {overall[s][0]:.0f}" for s in shown[:4])
            )
    data["families"] = {
        key: {"label": label, "color": "#" + hex_}
        for key, label, hex_ in gwc.SLIDE_BAR_FAMILIES
    }
    return data


def scaling_variance() -> dict:
    """Mean and 95% band of the relative excess risk per K, from the generated figure."""
    tex = (ROOT / "tex" / "figures" / "scaling_variance.tex").read_text(
        encoding="utf-8"
    )
    bands: dict[int, dict[str, list[list[float]]]] = {}
    for side, k, body in re.findall(
        r"name path=(lo|hi)_log_K(\d+)[^\]]*\] coordinates \{(.*?)\};",
        tex,
        re.S,
    ):
        pts = [
            [float(a), float(b)]
            for a, b in re.findall(r"\(([-\d.e]+),\s*([-\d.e]+)\)", body)
        ]
        bands.setdefault(int(k), {})[side] = pts
    curves = []
    for k in sorted(bands):
        lo, hi = bands[k]["lo"], bands[k]["hi"]
        curves.append(
            {
                "K": k,
                "lambda": [p[0] for p in lo],
                "lo": [p[1] for p in lo],
                "hi": [p[1] for p in hi],
                "mean": [round((a[1] + b[1]) / 2, 6) for a, b in zip(lo, hi)],
            }
        )
    title = re.search(r"title=\{([^}]*)\}", tex).group(1)
    print(
        f"  scaling_variance: K = {[c['K'] for c in curves]}, {len(curves[0]['lambda'])} points each, {title}"
    )
    return {"title": title, "curves": curves}


def datasets() -> dict:
    """Size, classes and imbalance ratio of the benchmark datasets, from the thesis tables."""
    out = {}
    for scope in ("binary", "multiclass"):
        tex = (
            ROOT / "tex" / "tables" / f"dataset_summary_{scope}.tex"
        ).read_text(encoding="utf-8")
        rows = []
        for line in tex.splitlines():
            m = re.match(r"([A-Za-z0-9\\_]+)\\textsuperscript", line)
            if not m:
                continue
            cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
            row = {
                "name": m.group(1).replace("\\_", "_"),
                "n": int(cells[1]),
                "ir": float(cells[-1]),
            }
            if scope == "multiclass":
                row["K"] = int(cells[2])
            rows.append(row)
        out[scope] = rows
        print(
            f"  datasets {scope}: {len(rows)} rows, n from {min(r['n'] for r in rows)} to {max(r['n'] for r in rows)}"
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("win_rates", win_rates()),
        ("scaling_variance", scaling_variance()),
        ("datasets", datasets()),
    ):
        text = (
            f"// Generated by slides/tools/export_data.py; do not edit.\n"
            f"window.Deck = window.Deck || {{}};\nDeck.data = Deck.data || {{}};\n"
            f"Deck.data.{name} = {json.dumps(payload, separators=(',', ':'))};\n"
        )
        (OUT / f"{name}.js").write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {OUT / (name + '.js')}")


if __name__ == "__main__":
    main()
