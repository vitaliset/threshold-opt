"""Unit tests for experiments/analysis/generate_win_counts.py."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from experiments.analysis.generate_win_counts import (
    IR_BUCKETS,
    MODEL_LABELS,
    MODEL_ORDER,
    SIZE_BUCKETS,
    _bucket,
    _green_pct,
    _winners_one_cell,
    aggregate_win_counts,
    bootstrap_breakdown_cis,
    compute_win_table,
    render_win_count_figure,
)

# Synthetic-data fixtures


def _make_cell(
    *,
    dataset: str,
    model: str,
    objective: str,
    strategy_means: dict[str, float],
    seeds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Build *seeds* observations per strategy on one (ds, model, obj) cell."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for strat, m in strategy_means.items():
        for _ in range(seeds):
            rows.append(
                {
                    "dataset_name": dataset,
                    "model_name": model,
                    "optimization_objective_name": objective,
                    "strategy_name": strat,
                    "objective_value": m + 0.001 * rng.standard_normal(),
                }
            )
    return pd.DataFrame(rows)


def _binary_cost_df() -> pd.DataFrame:
    """Two-dataset binary cost benchmark with two strategies + a bystander.

    Strategy means are tuned so:
    - small_ds (Symmetric): Empirical wins, EmpiricalCV ties, Argmax loses.
    - large_ds (Asymmetric 100x): EmpiricalCV wins solo, others lose.

    Net: Empirical 1 win, EmpiricalCV 2 wins, Argmax 0 wins.
    """
    parts = []
    parts.append(
        _make_cell(
            dataset="small_ds",
            model="LGBMClassifier()",
            objective="Symmetric Cost",
            strategy_means={
                "Empirical": 0.20,
                "EmpiricalCV": 0.20,
                "Argmax": 0.50,
            },
        )
    )
    parts.append(
        _make_cell(
            dataset="large_ds",
            model="LGBMClassifier()",
            objective="Asymmetric 100x Cost",
            strategy_means={
                "Empirical": 0.40,
                "EmpiricalCV": 0.10,
                "Argmax": 0.55,
            },
        )
    )
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def synthetic_metadata() -> dict[str, dict[str, float]]:
    return {
        "small_ds": {"n": 500.0, "ir": 1.2},
        "large_ds": {"n": 50000.0, "ir": 25.0},
    }


# Bucketing helpers


def test_bucket_size_partitions_rows():
    assert _bucket(500.0, SIZE_BUCKETS) == "small"
    assert _bucket(1500.0, SIZE_BUCKETS) == "medium"
    assert _bucket(99999.0, SIZE_BUCKETS) == "large"
    # Boundary: 1000 -> medium (half-open at the lower edge).
    assert _bucket(1000.0, SIZE_BUCKETS) == "medium"
    assert _bucket(10000.0, SIZE_BUCKETS) == "large"


def test_bucket_ir_binary_three_buckets():
    bins = IR_BUCKETS["binary"]
    assert _bucket(1.5, bins) == "low"
    assert _bucket(2.0, bins) == "moderate"
    assert _bucket(8.0, bins) == "moderate"
    assert _bucket(15.0, bins) == "high"


def test_bucket_ir_multiclass_two_buckets():
    bins = IR_BUCKETS["multiclass"]
    assert _bucket(1.0, bins) == "balanced"
    assert _bucket(1.99, bins) == "balanced"
    assert _bucket(2.0, bins) == "imbalanced"
    assert _bucket(2160.0, bins) == "imbalanced"


# CI overlap winner detection


def test_winners_one_cell_picks_unique_best_for_cost():
    means = pd.Series({"A": 0.10, "B": 0.30, "C": 0.55})
    stds = pd.Series({"A": 0.001, "B": 0.001, "C": 0.001})
    counts = pd.Series({"A": 5, "B": 5, "C": 5})
    winners = _winners_one_cell(means, stds, counts, is_max=False)
    assert winners == {"A"}


def test_winners_one_cell_picks_unique_best_for_max():
    means = pd.Series({"A": 0.55, "B": 0.30, "C": 0.10})
    stds = pd.Series({"A": 0.001, "B": 0.001, "C": 0.001})
    counts = pd.Series({"A": 5, "B": 5, "C": 5})
    winners = _winners_one_cell(means, stds, counts, is_max=True)
    assert winners == {"A"}


def test_winners_one_cell_ties_via_ci_overlap():
    """B's CI overlaps with A's even though A has the lower mean."""
    means = pd.Series({"A": 0.20, "B": 0.22, "C": 0.55})
    # SE = std / sqrt(5). std = 0.05 -> SE \approx 0.0224.
    # A: [0.178, 0.222]; B: [0.198, 0.242]. Overlap.
    stds = pd.Series({"A": 0.05, "B": 0.05, "C": 0.05})
    counts = pd.Series({"A": 5, "B": 5, "C": 5})
    winners = _winners_one_cell(means, stds, counts, is_max=False)
    assert "A" in winners and "B" in winners and "C" not in winners


def test_winners_one_cell_handles_nan_strategies():
    means = pd.Series({"A": 0.20, "B": np.nan, "C": 0.55})
    stds = pd.Series({"A": 0.001, "B": np.nan, "C": 0.001})
    counts = pd.Series({"A": 5, "B": 0, "C": 5})
    winners = _winners_one_cell(means, stds, counts, is_max=False)
    assert winners == {"A"}


def test_winners_one_cell_returns_empty_when_all_nan():
    means = pd.Series({"A": np.nan, "B": np.nan})
    stds = pd.Series({"A": np.nan, "B": np.nan})
    counts = pd.Series({"A": 0, "B": 0})
    winners = _winners_one_cell(means, stds, counts, is_max=True)
    assert winners == set()


# Win table + aggregation


def test_compute_win_table_marks_winners_per_cell():
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    assert set(wt["strategy_name"].unique()) == {
        "Empirical",
        "EmpiricalCV",
        "Argmax",
    }
    # On small_ds Symmetric, Empirical and EmpiricalCV tie; Argmax loses.
    small = wt[
        (wt["dataset_name"] == "small_ds")
        & (wt["optimization_objective_name"] == "Symmetric Cost")
    ]
    winners = set(small[small["is_winner"]]["strategy_name"])
    assert winners == {"Empirical", "EmpiricalCV"}
    # On large_ds Asymmetric 100x, EmpiricalCV wins solo.
    large = wt[
        (wt["dataset_name"] == "large_ds")
        & (wt["optimization_objective_name"] == "Asymmetric 100x Cost")
    ]
    winners = set(large[large["is_winner"]]["strategy_name"])
    assert winners == {"EmpiricalCV"}


def test_aggregate_win_counts_breakdowns_match_expected(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    agg = aggregate_win_counts(wt, synthetic_metadata, scope="binary")

    # ---- Total ----
    total_count, total_denom = agg["total"]
    assert total_count.loc["EmpiricalCV", "all"] == 2
    assert total_count.loc["Empirical", "all"] == 1
    assert total_count.loc["Argmax", "all"] == 0
    assert (total_denom["all"] == 2).all()  # every strategy has 2 cells

    # ---- By model ----
    bm_count, _bm_denom = agg["by_model"]
    # Only LGBM is present in the synthetic data.
    assert list(bm_count.columns) == ["LGBMClassifier()"]
    assert bm_count.loc["EmpiricalCV", "LGBMClassifier()"] == 2

    # ---- By size ----
    bs_count, bs_denom = agg["by_size"]
    # small_ds (n=500) -> small bucket; large_ds (n=50k) -> large bucket.
    # No medium-bucket cells, so that column is all zeros.
    assert bs_count.loc["EmpiricalCV", "small"] == 1
    assert bs_count.loc["EmpiricalCV", "large"] == 1
    assert bs_count.loc["EmpiricalCV", "medium"] == 0
    assert bs_count.loc["Empirical", "small"] == 1
    assert bs_count.loc["Empirical", "large"] == 0
    assert bs_denom.loc["EmpiricalCV", "medium"] == 0

    # ---- By IR (binary scheme: low / moderate / high) ----
    bi_count, _bi_denom = agg["by_ir"]
    # small_ds IR=1.2 -> low; large_ds IR=25 -> high.
    assert bi_count.loc["EmpiricalCV", "low"] == 1
    assert bi_count.loc["EmpiricalCV", "high"] == 1
    assert bi_count.loc["EmpiricalCV", "moderate"] == 0
    assert list(bi_count.columns) == ["low", "moderate", "high"]


def test_aggregate_win_counts_uses_multiclass_ir_buckets(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    agg = aggregate_win_counts(wt, synthetic_metadata, scope="multiclass")
    # Multiclass uses two IR buckets, not three.
    assert list(agg["by_ir"][0].columns) == ["balanced", "imbalanced"]


# Rendering


def test_green_pct_clamps_and_scales():
    # Quadratic scale: rate=0 -> green!0, rate=1 -> green!80, and
    # 80 * 0.5**2 = 20 -> green!20, so mid-rate cells stay pale and the
    # high-rate tier (where most cells live) gets the brunt of the green
    # range.
    assert _green_pct(0.0) == 0
    assert _green_pct(1.0) == 80
    assert _green_pct(0.5) == 20
    assert _green_pct(-0.5) == 0
    assert _green_pct(1.5) == 80


def test_render_win_count_figure_has_expected_structure(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    agg = aggregate_win_counts(wt, synthetic_metadata, scope="binary")
    tex = render_win_count_figure(
        scope="binary", family="cost", breakdowns=agg
    )
    # Headers and trailing structures.
    assert tex.startswith("% Auto-generated")
    assert "\\begin{figure}[H]" in tex
    assert "\\begin{tikzpicture}" in tex
    assert "\\end{tikzpicture}%" in tex
    assert "\\caption{" in tex
    assert "\\label{fig:win-counts-binary-cost}" in tex
    # Group labels (short forms).
    assert "{Total}" in tex
    assert "{Base classifier}" in tex
    assert "{Size}" in tex
    assert "{IR}" in tex
    # Per-column labels include their denominator.
    assert "all ($n{=}2$)" in tex
    # Strategy rows are sorted by total wins, EmpiricalCV first.
    rowlbl_lines = [
        ln for ln in tex.splitlines() if ln.startswith("\\node[rowlbl]")
    ]
    rowlbl_strats = [ln.split("{")[-1].rstrip("};") for ln in rowlbl_lines]
    assert rowlbl_strats[0] == "EmpiricalCV"
    # Argmax (0 wins) should be last.
    assert rowlbl_strats[-1] == "Argmax"


def test_render_win_count_figure_winner_cell_is_dark_green(
    synthetic_metadata,
):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    agg = aggregate_win_counts(wt, synthetic_metadata, scope="binary")
    tex = render_win_count_figure(
        scope="binary", family="cost", breakdowns=agg
    )
    # The Total cell for EmpiricalCV is 2/2 = 100%. With clamp at 80, it
    # should produce ``green!80``. EmpiricalCV is the top row -> y == n_rows.
    assert "fill=green!80" in tex
    # Argmax's Total cell is 0/2 = 0% -> green!0.
    assert "fill=green!0" in tex


def test_bootstrap_returns_aligned_lo_hi_dfs(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    cis = bootstrap_breakdown_cis(
        wt, synthetic_metadata, scope="binary", n_boot=200, seed=42
    )
    # Same keys as aggregate_win_counts.
    assert set(cis.keys()) == {"total", "by_model", "by_size", "by_ir"}
    for key, (lo, hi) in cis.items():
        # Same shape on both sides.
        assert lo.shape == hi.shape
        # 0 <= lo <= hi <= 1 everywhere.
        assert (lo.values >= 0).all()
        assert (hi.values <= 1).all()
        assert (lo.values <= hi.values + 1e-12).all()
        # Indexed by strategy_name.
        assert lo.index.name == "strategy_name"
        assert hi.index.name == "strategy_name"


def test_bootstrap_is_seed_deterministic(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    a = bootstrap_breakdown_cis(
        wt, synthetic_metadata, scope="binary", n_boot=100, seed=7
    )
    b = bootstrap_breakdown_cis(
        wt, synthetic_metadata, scope="binary", n_boot=100, seed=7
    )
    for key in a:
        pd.testing.assert_frame_equal(a[key][0], b[key][0])
        pd.testing.assert_frame_equal(a[key][1], b[key][1])


def test_render_with_cis_uses_pm_format(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    agg = aggregate_win_counts(wt, synthetic_metadata, scope="binary")
    cis = bootstrap_breakdown_cis(
        wt, synthetic_metadata, scope="binary", n_boot=200, seed=0
    )
    tex = render_win_count_figure(
        scope="binary",
        family="cost",
        breakdowns=agg,
        ci_breakdowns=cis,
    )
    # Cells should use math-mode "$N{\pm}H$" format and the caption
    # should mention the bootstrap.
    assert "{\\pm}" in tex
    assert "bootstrap" in tex
    # No raw "%" sign on cells once CIs are present (caption notes the
    # values are percentages).
    rate_node_lines = [
        ln for ln in tex.splitlines() if ln.startswith("\\node[cell")
    ]
    assert rate_node_lines, "expected rendered cells"
    for ln in rate_node_lines:
        assert "\\%" not in ln, ln


def test_render_without_cis_keeps_percent_sign(synthetic_metadata):
    df = _binary_cost_df()
    wt = compute_win_table(df, max_metrics=())
    agg = aggregate_win_counts(wt, synthetic_metadata, scope="binary")
    tex = render_win_count_figure(
        scope="binary", family="cost", breakdowns=agg
    )
    assert "\\%" in tex
    assert "{\\pm}" not in tex


def test_render_win_count_figure_raises_on_empty_breakdowns():
    empty_count = pd.DataFrame({"all": []})
    empty_count.index.name = "strategy_name"
    empty_denom = empty_count.copy()
    breakdowns = {
        "total": (empty_count, empty_denom),
        "by_model": (empty_count.copy(), empty_denom.copy()),
        "by_size": (empty_count.copy(), empty_denom.copy()),
        "by_ir": (empty_count.copy(), empty_denom.copy()),
    }
    with pytest.raises(ValueError, match="No strategies"):
        render_win_count_figure(
            scope="binary", family="cost", breakdowns=breakdowns
        )


# Module-level constants


def test_model_order_and_labels_cover_benchmark_models():
    benchmark = {
        "LGBMClassifier()",
        "XGBoost",
        "CatBoost",
        "RandomForest",
        "ExtraTrees",
        "LogisticRegression()",
        "GaussianNB",
        "TabM",
        "TabPFN",
        "TabICL",
    }
    assert benchmark.issubset(set(MODEL_ORDER))
    assert benchmark.issubset(set(MODEL_LABELS))


def test_size_buckets_partition_real_dataset_range():
    # The benchmark spans roughly 150--300k rows; the three buckets must
    # together cover that range with no gaps.
    edges = sorted(
        {0.0}
        | {hi for _, _, hi in SIZE_BUCKETS}
        | {lo for _, lo, _ in SIZE_BUCKETS}
    )
    edges = [e for e in edges if e != float("inf")]
    for lo, hi in itertools.pairwise(edges):
        # Pick a value strictly inside (lo, hi) and confirm it falls in
        # exactly one bucket.
        mid = (lo + hi) / 2.0
        labels = [b[0] for b in SIZE_BUCKETS if b[1] <= mid < b[2]]
        assert len(labels) == 1
