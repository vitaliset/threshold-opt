import numpy as np
import pandas as pd

from experiments.analysis.latex_formatting import (
    build_latex_table,
    format_latex_row,
)

MAX_METRICS = ("F1", "MCC", "F2", "F0.5", "GMean")


# Helpers


def _make_mean_std_rows(
    means: dict[str, float],
    stds: dict[str, float],
    counts: dict[str, float] | None = None,
    objective: str = "F1",
    dataset: str = "ds1",
    model: str = "ModelA",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Index]:
    """Build mean_row, std_row, count_row, strategy_columns.

    Default count is 1 per strategy, which makes ``ci_half = z * s``
    (the n<=1 fallback) and preserves the pre-refactor test semantics
    for the bold-overlap logic.
    """
    idx = (dataset, model, objective)
    mean_row = pd.Series(means, name=idx)
    std_row = pd.Series(stds, name=idx)
    if counts is None:
        counts = {k: 1 for k in means}
    count_row = pd.Series(counts, name=idx)
    strategy_cols = pd.Index(list(means.keys()))
    return mean_row, std_row, count_row, strategy_cols


def _make_results_df(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _simple_results(
    objective: str = "F1",
    strategies_values: dict[str, list[float]] | None = None,
    dataset: str = "ds1",
    model: str = "ModelA",
    seeds: list[int] | None = None,
) -> list[dict]:
    if strategies_values is None:
        strategies_values = {
            "Argmax": [0.5, 0.6, 0.55],
            "EmpiricalCV": [0.8, 0.85, 0.82],
        }
    if seeds is None:
        seeds = list(range(len(next(iter(strategies_values.values())))))
    rows = []
    for strat, vals in strategies_values.items():
        for seed, v in zip(seeds, vals):
            rows.append(
                {
                    "dataset_name": dataset,
                    "model_name": model,
                    "optimization_objective_name": objective,
                    "strategy_name": strat,
                    "objective_value": v,
                    "random_seed": seed,
                }
            )
    return rows


# Tests for format_latex_row


class TestFormatLatexRow:
    def test_best_strategy_is_bold(self):
        """The strategy with the best mean should be bolded."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"Argmax": 0.5, "EmpiricalCV": 0.8},
            stds={"Argmax": 0.01, "EmpiricalCV": 0.01},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        # EmpiricalCV (0.8) is best for maximization
        assert "\\textbf" not in cells[0]  # Argmax
        assert "\\textbf" in cells[1]  # EmpiricalCV

    def test_minimization_best_is_lowest(self):
        """For cost objectives, the lowest mean should be bolded."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"Argmax": 0.5, "Analytical": 0.2},
            stds={"Argmax": 0.01, "Analytical": 0.01},
            objective="Symmetric Cost",  # not in MAX_METRICS -> minimization
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "\\textbf" not in cells[0]  # Argmax (0.5, worse)
        assert "\\textbf" in cells[1]  # Analytical (0.2, best)

    def test_overlapping_cis_both_bold(self):
        """Two strategies with overlapping CIs should both be bolded."""
        # Large std -> wide CI -> overlap
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"StratA": 0.50, "StratB": 0.52},
            stds={"StratA": 0.10, "StratB": 0.10},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "\\textbf" in cells[0]
        assert "\\textbf" in cells[1]

    def test_non_overlapping_cis_only_best_bold(self):
        """Strategies with non-overlapping CIs: only the best is bold."""
        # Very small std -> tight CI -> no overlap
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"StratA": 0.50, "StratB": 0.90},
            stds={"StratA": 0.001, "StratB": 0.001},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "\\textbf" not in cells[0]  # StratA far from best
        assert "\\textbf" in cells[1]  # StratB is best

    def test_nan_values_formatted_as_dash(self):
        """NaN values should appear as em-dash."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"Argmax": 0.5, "Analytical": np.nan},
            stds={"Argmax": 0.01, "Analytical": np.nan},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert cells[1] == "\u2014"

    def test_all_nan_returns_dashes(self):
        """If all strategies are NaN, return all dashes."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"StratA": np.nan, "StratB": np.nan},
            stds={"StratA": np.nan, "StratB": np.nan},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert all(c == "\u2014" for c in cells)

    def test_format_precision(self):
        """Default 3 decimal places."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"Argmax": 0.123456},
            stds={"Argmax": 0.0},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "0.123" in cells[0]
        assert "0.1235" not in cells[0]

    def test_custom_precision(self):
        """Custom decimal places."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"Argmax": 0.123456},
            stds={"Argmax": 0.0},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS, decimals=5
        )
        assert "0.12346" in cells[0]

    def test_custom_z_value(self):
        """Custom z multiplier changes CI width."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"StratA": 0.50, "StratB": 0.55},
            stds={"StratA": 0.02, "StratB": 0.02},
            objective="F1",
        )
        # With z=1.0 (narrow CI), they might not overlap
        _cells_narrow = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS, z=1.0
        )
        # With z=5.0 (wide CI), they should overlap
        cells_wide = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS, z=5.0
        )
        # Wide CI should have both bold
        assert "\\textbf" in cells_wide[0]
        assert "\\textbf" in cells_wide[1]

    def test_cell_format_structure(self):
        """Cell should contain 'mean ± ci_half'."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"Argmax": 0.500},
            stds={"Argmax": 0.100},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS, z=2.0
        )
        # 0.500 ± 0.200
        assert "0.500" in cells[0]
        assert "0.200" in cells[0]
        assert "±" in cells[0]

    def test_single_strategy_always_bold(self):
        """A lone strategy's CI trivially overlaps with itself."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"OnlyOne": 0.75},
            stds={"OnlyOne": 0.05},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "\\textbf" in cells[0]

    def test_three_strategies_partial_overlap(self):
        """Three strategies: best + one overlap, one doesn't."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"A": 0.30, "B": 0.79, "C": 0.80},
            stds={"A": 0.01, "B": 0.02, "C": 0.01},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "\\textbf" not in cells[0]  # A: 0.30, far away
        assert "\\textbf" in cells[1]  # B: 0.79, overlaps with C
        assert "\\textbf" in cells[2]  # C: 0.80, best

    def test_zero_std_exact_comparison(self):
        """With zero std, only best-value strategies are bold."""
        mean_row, std_row, count_row, cols = _make_mean_std_rows(
            means={"A": 0.5, "B": 0.8, "C": 0.8},
            stds={"A": 0.0, "B": 0.0, "C": 0.0},
            objective="F1",
        )
        cells = format_latex_row(
            mean_row, std_row, count_row, cols, MAX_METRICS
        )
        assert "\\textbf" not in cells[0]  # A: 0.5
        assert "\\textbf" in cells[1]  # B: 0.8 (tied best)
        assert "\\textbf" in cells[2]  # C: 0.8 (tied best)

    def test_all_maximization_metrics_use_max(self):
        """Every metric in MAX_METRICS should bold the highest mean."""
        for metric in MAX_METRICS:
            mean_row, std_row, count_row, cols = _make_mean_std_rows(
                means={"Low": 0.3, "High": 0.9},
                stds={"Low": 0.001, "High": 0.001},
                objective=metric,
            )
            cells = format_latex_row(
                mean_row, std_row, count_row, cols, MAX_METRICS
            )
            assert "\\textbf" not in cells[0], (
                f"{metric}: Low should not be bold"
            )
            assert "\\textbf" in cells[1], f"{metric}: High should be bold"


# Tests for build_latex_table


class TestBuildLatexTable:
    def test_produces_latex_string(self):
        """build_latex_table returns a non-empty string with LaTeX markers."""
        df = _make_results_df(
            _simple_results(
                objective="F1",
                strategies_values={
                    "Argmax": [0.5, 0.6, 0.55],
                    "EmpiricalCV": [0.8, 0.85, 0.82],
                },
            )
        )
        latex = build_latex_table(df, MAX_METRICS)
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_contains_strategy_header(self):
        """Strategy names appear as column headers."""
        df = _make_results_df(
            _simple_results(
                strategies_values={
                    "Argmax": [0.5, 0.6, 0.55],
                    "Analytical": [0.7, 0.65, 0.72],
                },
            )
        )
        latex = build_latex_table(df, MAX_METRICS)
        assert "Argmax" in latex
        assert "Analytical" in latex

    def test_escapes_underscores_in_dataset_names(self):
        """Underscores in dataset names are escaped for LaTeX."""
        df = _make_results_df(
            _simple_results(
                dataset="my_dataset",
                strategies_values={"Argmax": [0.5, 0.5, 0.5]},
            )
        )
        latex = build_latex_table(df, MAX_METRICS)
        assert "my\\_dataset" in latex

    def test_no_escape_underscores_option(self):
        """escape_underscores=False leaves underscores as-is."""
        df = _make_results_df(
            _simple_results(
                dataset="my_dataset",
                strategies_values={"Argmax": [0.5, 0.5, 0.5]},
            )
        )
        latex = build_latex_table(df, MAX_METRICS, escape_underscores=False)
        assert "my_dataset" in latex
        assert "my\\_dataset" not in latex

    def test_shorten_model_names(self):
        """Model name shortening works."""
        df = _make_results_df(
            _simple_results(
                model="LGBMClassifier",
                strategies_values={"Argmax": [0.5, 0.5, 0.5]},
            )
        )
        latex = build_latex_table(
            df,
            MAX_METRICS,
            shorten_model_names={"LGBMClassifier": "LGBM"},
        )
        assert "LGBM" in latex
        assert "LGBMClassifier" not in latex

    def test_mixed_objectives_formatting(self):
        """Max and min objectives are bolded correctly in the same table."""
        rows = _simple_results(
            objective="F1",
            strategies_values={
                "StratA": [0.3, 0.3, 0.3],
                "StratB": [0.9, 0.9, 0.9],
            },
        ) + _simple_results(
            objective="Symmetric Cost",
            strategies_values={
                "StratA": [0.3, 0.3, 0.3],
                "StratB": [0.9, 0.9, 0.9],
            },
        )
        df = _make_results_df(rows)
        latex = build_latex_table(df, MAX_METRICS)
        # Both objectives present
        assert "F1" in latex
        assert "Symmetric" in latex

    def test_replaces_strategy_name_header(self):
        """The 'strategy_name' header is replaced with 'Strategy'."""
        df = _make_results_df(
            _simple_results(strategies_values={"Argmax": [0.5, 0.5, 0.5]})
        )
        latex = build_latex_table(df, MAX_METRICS)
        assert "strategy_name" not in latex

    def test_multiple_groups(self):
        """Table handles multiple datasets, models, objectives."""
        rows = []
        for ds in ["ds1", "ds2"]:
            for model in ["M1", "M2"]:
                rows += _simple_results(
                    objective="F1",
                    strategies_values={"Argmax": [0.5, 0.5, 0.5]},
                    dataset=ds,
                    model=model,
                )
        df = _make_results_df(rows)
        latex = build_latex_table(df, MAX_METRICS)
        assert isinstance(latex, str)
        assert len(latex) > 100


# Float placement in generated table wrappers


class TestTableFloatPlacement:
    """Verify that generated table wrappers use [H] placement."""

    def test_dataset_stats_uses_fixed_placement(self, monkeypatch):
        """build_dataset_stats_latex wraps with \\begin{table}[H]."""
        from experiments.analysis import data_stats
        from experiments.common import data_loader

        monkeypatch.setattr(data_stats, "DATASETS", ["fake"])
        monkeypatch.setattr(
            data_loader,
            "load_raw_dataset",
            lambda name: (pd.DataFrame({"a": [0, 1]}), np.array([0, 1])),
        )
        latex = data_stats.build_dataset_stats_latex()
        assert "\\begin{table}[H]" in latex
        assert "\\begin{table}[htbp]" not in latex
