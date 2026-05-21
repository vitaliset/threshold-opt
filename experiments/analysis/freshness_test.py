"""Freshness tests: regenerate artifacts + diff vs. committed copies.

Table tests are opt-in via ``--freshness`` because they depend on a parquet
that may be absent on a fresh clone.  Figure tests always run — figure
generators are deterministic functions of the source, so staleness is a
real bug regardless of experiment state.  A failure means the committed
artifact no longer matches what the current code would produce — run
``make tables`` or ``make figures`` to refresh.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pandas as pd
import pytest

from experiments.analysis import (
    generate_cost_partition_figure,
    generate_marchetti_figure,
    generate_partition_figure,
    generate_simplex_regions_figure,
)
from experiments.analysis.generate_ci_plots import (
    PLOTS_DIR,
    write_ci_plots,
)
from experiments.analysis.generate_obrien_figure import (
    OUT_A as OBRIEN_FIG_A,
)
from experiments.analysis.generate_obrien_figure import (
    OUT_B as OBRIEN_FIG_B,
)
from experiments.analysis.generate_obrien_figure import (
    build_figure_a,
    build_figure_b,
)
from experiments.analysis.generate_tables import (
    RESULTS_DIR,
    SCOPE_CONFIG,
    SHORTEN_MODEL_NAMES,
    SUPPLEMENT_TABLES_DIR,
    TABLES_DIR,
    _find_latest_parquet,
    _sanitize_filename,
    write_rotational_bias_table,
    write_strategy_tables,
)
from experiments.analysis.generate_win_counts import (
    WIN_COUNTS_DIR,
    write_win_count_figures,
)
from experiments.common.git_metadata import check_parquet_freshness


def _diff(expected: str, actual: str, name: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile=f"committed/{name}",
        tofile=f"regenerated/{name}",
        lineterm="",
    )
    return "\n".join(diff)


def _assert_matches_committed(
    generated: Path, directory: Path = TABLES_DIR
) -> None:
    committed = directory / generated.name
    assert committed.exists(), (
        f"{committed} is missing — regenerated {generated.name} has no "
        f"committed counterpart."
    )
    expected = committed.read_text(encoding="utf-8")
    actual = generated.read_text(encoding="utf-8")
    if expected != actual:
        pytest.fail(
            f"{generated.name} is stale: committed copy differs from what "
            f"the current code + latest parquet would produce. Run "
            f"`make tables` to refresh.\n\n"
            + _diff(expected, actual, generated.name)
        )


def _assert_parquet_fresh(parquet: Path) -> None:
    status = check_parquet_freshness(parquet)
    if not status.fresh:
        pytest.fail(status.reason)


@pytest.mark.freshness
def test_strategy_tables_fresh(tmp_path):
    try:
        parquet = _find_latest_parquet()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    _assert_parquet_fresh(parquet)
    written = write_strategy_tables(parquet_path=parquet, tables_dir=tmp_path)
    assert written, "write_strategy_tables returned no outputs"
    for generated in written:
        _assert_matches_committed(generated, directory=SUPPLEMENT_TABLES_DIR)


def _shorten_model_name(name: str) -> str:
    for old, new in SHORTEN_MODEL_NAMES.items():
        name = name.replace(old, new)
    return name


_MULTIROW_MODEL_RE = re.compile(r"\\multirow\[t\]\{\d+\}\{\*\}\{([^}]+)\}")


@pytest.mark.freshness
def test_per_dataset_tables_cover_all_run_models():
    """Every model that produced at least one row for a (dataset, variant)
    combination in the parquet should appear as a row in the corresponding
    committed ``results_<dataset>_<variant>.tex``. Catches silent omissions
    where a model ran but didn't make it into the rendered table."""

    for scope, cfg in SCOPE_CONFIG.items():
        try:
            parquet = _find_latest_parquet(cfg["parquet_pattern"])
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        df = pd.read_parquet(parquet)
        df = df[df["dataset_name"].isin(cfg["datasets"])]

        for variant, objs in (
            ("cost", cfg["cost_objectives"]),
            ("max", cfg["max_metrics"]),
        ):
            df_v = df[df["optimization_objective_name"].isin(objs)]
            for ds in cfg["datasets"]:
                ds_df = df_v[df_v["dataset_name"] == ds]
                if ds_df.empty:
                    continue
                expected = {
                    _shorten_model_name(m)
                    for m in ds_df["model_name"].unique()
                }
                file_path = (
                    SUPPLEMENT_TABLES_DIR
                    / f"results_{_sanitize_filename(ds)}_{variant}.tex"
                )
                assert file_path.exists(), (
                    f"{file_path} is missing but the {scope} parquet "
                    f"has rows for {ds}/{variant}."
                )
                content = file_path.read_text(encoding="utf-8")
                actual = set(_MULTIROW_MODEL_RE.findall(content))
                missing = expected - actual
                assert not missing, (
                    f"{file_path.name} is missing model rows for "
                    f"{sorted(missing)} that the parquet has data for. "
                    f"Run `make tables` to refresh."
                )


def _assert_matches_committed_in(directory: Path, generated: Path) -> None:
    committed = directory / generated.name
    assert committed.exists(), (
        f"{committed} is missing — regenerated {generated.name} has no "
        f"committed counterpart."
    )
    expected = committed.read_text(encoding="utf-8")
    actual = generated.read_text(encoding="utf-8")
    if expected != actual:
        pytest.fail(
            f"{generated.name} is stale: committed copy differs from what "
            f"the current code + latest parquet would produce. Run "
            f"`make ci-plots` to refresh.\n\n"
            + _diff(expected, actual, generated.name)
        )


@pytest.mark.freshness
def test_ci_plots_fresh(tmp_path):
    """The inline CI plots embedded in chapters 4 and 8 must match what the
    current code + latest binary parquet would produce."""
    try:
        parquet = _find_latest_parquet(
            SCOPE_CONFIG["binary"]["parquet_pattern"]
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    _assert_parquet_fresh(parquet)
    written = write_ci_plots(
        parquet_path=parquet, plots_dir=tmp_path, scope="binary"
    )
    assert written, "write_ci_plots returned no outputs"
    for generated in written:
        _assert_matches_committed_in(PLOTS_DIR, generated)


@pytest.mark.freshness
def test_win_count_figures_fresh(tmp_path):
    """The four per-strategy win/tie heatmaps embedded in chapters 4 and 8
    must match what the current code + latest parquets would produce.

    Loading the dataset metadata (n, IR) hits the OpenML cache; we read it
    from the committed dataset stats so the freshness check stays
    self-contained and doesn't redownload on slow networks."""
    try:
        parquet_paths = {
            scope: _find_latest_parquet(SCOPE_CONFIG[scope]["parquet_pattern"])
            for scope in ("binary", "multiclass")
        }
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    for parquet in parquet_paths.values():
        _assert_parquet_fresh(parquet)
    written = write_win_count_figures(
        parquet_paths=parquet_paths, plots_dir=tmp_path
    )
    assert written, "write_win_count_figures returned no outputs"
    for generated in written:
        _assert_matches_committed_in(WIN_COUNTS_DIR, generated)


@pytest.mark.freshness
def test_rotational_bias_table_fresh(tmp_path):
    files = sorted(RESULTS_DIR.glob("results_rotational_bias_*.parquet"))
    if not files:
        pytest.skip(
            f"No results_rotational_bias_*.parquet files in {RESULTS_DIR}"
        )
    parquet = files[-1]
    _assert_parquet_fresh(parquet)
    generated = write_rotational_bias_table(
        parquet_path=parquet, tables_dir=tmp_path
    )
    _assert_matches_committed(generated)


def _assert_figure_matches_committed(committed: Path, actual: str) -> None:
    assert committed.exists(), f"{committed} is missing from tex/figures/."
    expected = committed.read_text(encoding="utf-8")
    if expected != actual:
        pytest.fail(
            f"{committed.name} is stale: committed copy differs from what "
            f"the current code would produce. Run `make figures` to "
            f"refresh.\n\n" + _diff(expected, actual, committed.name)
        )


def test_obrien_parallel_shifts_figure_fresh():
    _assert_figure_matches_committed(OBRIEN_FIG_A, build_figure_a())


def test_obrien_vs_column_scaling_figure_fresh():
    _assert_figure_matches_committed(OBRIEN_FIG_B, build_figure_b())


def test_partition_strategies_figure_fresh():
    _assert_figure_matches_committed(
        generate_partition_figure.OUTPUT, generate_partition_figure.build()
    )


def test_marchetti_thresholding_figure_fresh():
    _assert_figure_matches_committed(
        generate_marchetti_figure.OUTPUT, generate_marchetti_figure.build()
    )


def test_simplex_cost_partition_figure_fresh():
    _assert_figure_matches_committed(
        generate_cost_partition_figure.OUTPUT,
        generate_cost_partition_figure.build(),
    )


# Unlike the cheap pure-TikZ figure tests above, this one is gated behind
# ``--freshness``: ``build()`` is deterministic but expensive — it refits five
# strategies at N_ITER=1000 — so running it on every ``make test`` / CI would
# materially slow the default suite. The fast smoke test below gives the
# default suite import/API-break coverage of the same module.
@pytest.mark.freshness
def test_simplex_regions_figure_fresh():
    _assert_figure_matches_committed(
        generate_simplex_regions_figure.OUTPUT,
        generate_simplex_regions_figure.build(),
    )


def test_simplex_regions_figure_smoke():
    """Fast structural guard against import/API breakage in the simplex
    figure pipeline. Uses a tiny search budget, so it does NOT reproduce the
    committed figure; it only asserts shape-invariant TikZ structure."""
    tex = generate_simplex_regions_figure.build(n_iter=5)
    assert tex.strip(), "build() returned empty TikZ"
    assert "\\documentclass" in tex, "missing standalone preamble"
    for label in generate_simplex_regions_figure.PANEL_LABELS:
        assert label in tex, f"panel label {label} missing from figure"
