"""Unit tests for experiments/analysis/generate_ci_plots.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import experiments.analysis.generate_ci_plots as gp
from experiments.analysis.generate_ci_plots import (
    _STATIC_SHOWCASE_FALLBACK,
    MODEL_LABELS,
    SCOPE_CONFIG,
    SELECTION_PURPOSES,
    SHOWCASE_PLOTS,
    VALID_X_SCALES,
    _cell_representativeness,
    _global_mean_ranks,
    _spearman,
    pivot_dataset_objective,
    render_plot,
    resolve_showcase_plots,
    select_illustrative_cell,
    write_ci_plots,
    write_showcase_plots,
)

# Synthetic-data fixtures


def _binary_max_slice(seeds: int = 5) -> pd.DataFrame:
    """One (dataset, max-objective) slice with two strategies on one model."""
    rng = np.random.default_rng(1)
    rows = []
    for strat, base in (("Argmax", 0.65), ("EmpiricalCV", 0.72)):
        for _ in range(seeds):
            rows.append(
                {
                    "dataset_name": "fake_ds",
                    "model_name": "CatBoost",
                    "optimization_objective_name": "F2",
                    "strategy_name": strat,
                    "objective_value": base + 0.01 * rng.standard_normal(),
                }
            )
    return pd.DataFrame(rows)


def _multi_dataset_df() -> pd.DataFrame:
    """Two datasets x both objective families on a single model."""
    rows = []
    rng = np.random.default_rng(2)
    for ds in ("dataset_a", "dataset_b"):
        for obj in ("Symmetric Cost", "F1"):
            for strat in ("Argmax", "EmpiricalCV"):
                for _ in range(3):
                    rows.append(
                        {
                            "dataset_name": ds,
                            "model_name": "LGBMClassifier()",
                            "optimization_objective_name": obj,
                            "strategy_name": strat,
                            "objective_value": 0.3
                            + 0.05 * rng.standard_normal(),
                        }
                    )
    return pd.DataFrame(rows)


def _single_model_max_fixture(strategy_means: dict[str, float]) -> str:
    """Render a single-model F2 plot from a means-by-strategy mapping."""
    rows = []
    for strat, m in strategy_means.items():
        for seed in range(3):
            rows.append(
                {
                    "dataset_name": "fake_ds",
                    "model_name": "CatBoost",
                    "optimization_objective_name": "F2",
                    "strategy_name": strat,
                    "objective_value": m + 0.0001 * seed,
                }
            )
    df = pd.DataFrame(rows)
    df_mean, df_se = pivot_dataset_objective(df)
    return render_plot(
        dataset="fake_ds",
        objective="F2",
        family="max",
        df_mean=df_mean,
        df_se=df_se,
    )


# pivot_dataset_objective


def test_pivot_dataset_objective_shapes_and_means():
    df = _binary_max_slice()
    df_mean, df_se = pivot_dataset_objective(df)
    assert sorted(df_mean.index) == ["Argmax", "EmpiricalCV"]
    assert list(df_mean.columns) == ["CatBoost"]
    assert df_mean.shape == df_se.shape == (2, 1)
    assert math.isclose(df_mean.loc["Argmax", "CatBoost"], 0.65, abs_tol=0.05)
    assert math.isclose(
        df_mean.loc["EmpiricalCV", "CatBoost"], 0.72, abs_tol=0.05
    )


def test_pivot_dataset_objective_se_uses_sqrt_n():
    df = _binary_max_slice(seeds=5)
    _df_mean, df_se = pivot_dataset_objective(df)
    raw = df[df["strategy_name"] == "EmpiricalCV"]["objective_value"]
    expected_se = raw.std() / math.sqrt(len(raw))
    assert math.isclose(
        df_se.loc["EmpiricalCV", "CatBoost"], expected_se, rel_tol=1e-9
    )


def test_pivot_dataset_objective_se_fallback_when_n_is_one():
    df = _binary_max_slice(seeds=1)
    _, df_se = pivot_dataset_objective(df)
    assert pd.isna(df_se.loc["EmpiricalCV", "CatBoost"])


# Defaults / showcase constants


def _cost_cell_df(
    dataset: str,
    model: str,
    objective: str,
    means: dict[str, list[float]],
) -> pd.DataFrame:
    """A single (dataset, model, objective) cost cell from explicit
    per-strategy seed values."""
    rows = []
    for strat, vals in means.items():
        for v in vals:
            rows.append(
                {
                    "dataset_name": dataset,
                    "model_name": model,
                    "optimization_objective_name": objective,
                    "strategy_name": strat,
                    "objective_value": v,
                }
            )
    return pd.DataFrame(rows)


# Six strategies with a clear cost ordering S1 (best) .. S6 (worst),
# and its reversal, for the representativeness rule.
_ORD = {f"S{i}": [round(0.1 * i, 2)] * 3 for i in range(1, 7)}
_REV = {f"S{i}": [round(0.1 * (7 - i), 2)] * 3 for i in range(1, 7)}


def _campaign_df(
    cells: list[tuple[str, str, str, dict[str, list[float]]]],
) -> pd.DataFrame:
    """Concatenate several (dataset, model, objective, means) cells into
    one synthetic campaign frame."""
    return pd.concat(
        [_cost_cell_df(ds, mdl, obj, means) for ds, mdl, obj, means in cells],
        ignore_index=True,
    )


def test_resolve_showcase_plots_one_entry_per_purpose():
    """The resolver returns one cell per SELECTION_PURPOSES entry, in
    order, each with a scope known to SCOPE_CONFIG and a valid x_scale."""
    resolved = resolve_showcase_plots()
    assert len(resolved) == len(SELECTION_PURPOSES)
    for (purpose_scope, _family, _purpose), entry in zip(
        SELECTION_PURPOSES, resolved
    ):
        scope, _ds, _mdl, _obj, x_scale = entry
        assert scope == purpose_scope
        assert scope in SCOPE_CONFIG
        assert x_scale in VALID_X_SCALES
    # SHOWCASE_PLOTS is the module-level resolution.
    assert len(SHOWCASE_PLOTS) == len(SELECTION_PURPOSES)


def test_resolve_showcase_plots_falls_back_without_parquet(
    tmp_path, monkeypatch
):
    """With no results directory the resolver returns the static
    fallback so the module still imports for unit tests."""
    import experiments.analysis.generate_tables as gt

    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    assert resolve_showcase_plots() == _STATIC_SHOWCASE_FALLBACK


def test_spearman_identical_reversed_and_degenerate():
    a = pd.Series({"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0})
    assert math.isclose(_spearman(a, a), 1.0, rel_tol=1e-9)
    rev = pd.Series({k: -v for k, v in a.items()})
    assert math.isclose(_spearman(a, rev), -1.0, rel_tol=1e-9)
    # Fewer than three common strategies, or a constant input: undefined.
    assert _spearman(a, pd.Series({"S1": 1.0, "S2": 2.0})) == float("-inf")
    flat = pd.Series({"S1": 1.0, "S2": 1.0, "S3": 1.0, "S4": 1.0})
    assert _spearman(a, flat) == float("-inf")


def test_global_mean_ranks_reflects_ordering():
    df = _campaign_df(
        [
            ("adult", "M", "Symmetric Cost", _ORD),
            ("sonar", "M", "Symmetric Cost", _ORD),
        ]
    )
    ranks = _global_mean_ranks(df, "cost")
    # _ORD has S1 cheapest, so S1 gets the lowest mean rank.
    assert list(ranks.sort_values().index) == [f"S{i}" for i in range(1, 7)]


def test_cell_representativeness_matches_and_reverses():
    # Two ORD cells and one REV cell, so the campaign-wide ordering is
    # the (non-degenerate) ORD ordering.
    df = _campaign_df(
        [
            ("adult", "M", "Symmetric Cost", _ORD),
            ("bank_marketing", "M", "Symmetric Cost", _ORD),
            ("sonar", "M", "Symmetric Cost", _REV),
        ]
    )
    global_ranks = _global_mean_ranks(df, "cost")
    ord_grp = df[df["dataset_name"] == "adult"]
    rev_grp = df[df["dataset_name"] == "sonar"]
    assert _cell_representativeness(ord_grp, "cost", global_ranks) > (
        _cell_representativeness(rev_grp, "cost", global_ranks)
    )
    # Too few strategies present: rejected.
    tiny = _cost_cell_df(
        "adult", "M", "Symmetric Cost", {"S1": [0.1] * 3, "S2": [0.2] * 3}
    )
    assert _cell_representativeness(tiny, "cost", global_ranks) == float(
        "-inf"
    )


def test_select_illustrative_cell_picks_representative_with_lex_tiebreak(
    tmp_path,
):
    """The cell whose ordering matches the campaign-wide ordering wins;
    among equally representative cells the lexicographically first
    (dataset, model, objective) wins, and the choice is stable."""
    df = _campaign_df(
        [
            ("bank_marketing", "M", "Symmetric Cost", _ORD),
            ("adult", "M", "Symmetric Cost", _ORD),
            ("sonar", "M", "Symmetric Cost", _REV),
        ]
    )
    parquet = tmp_path / "results_binary_synth.parquet"
    df.to_parquet(parquet)

    first = select_illustrative_cell(
        "binary", "cost", "representative", parquet
    )
    second = select_illustrative_cell(
        "binary", "cost", "representative", parquet
    )
    assert first == second
    # Both ORD cells tie at Spearman 1; lexicographic tie-break wins.
    assert first[1] == "adult"
    assert first[3] == "Symmetric Cost"
    assert first[4] in VALID_X_SCALES


def test_select_illustrative_cell_rejects_unknown_purpose(tmp_path):
    df = _campaign_df([("adult", "M", "Symmetric Cost", _ORD)])
    parquet = tmp_path / "results_binary_synth.parquet"
    df.to_parquet(parquet)
    with pytest.raises(ValueError, match="Unknown selection purpose"):
        select_illustrative_cell("binary", "cost", "bogus", parquet)


def test_model_labels_cover_all_palette_models():
    """Every model name in the parquet should resolve to a display label."""
    expected = {
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
    assert expected.issubset(MODEL_LABELS)


# render_plot --- structural assertions


def test_render_plot_emits_figure_and_axis():
    tex = _single_model_max_fixture({"Argmax": 0.65, "EmpiricalCV": 0.72})
    assert "\\begin{figure}[H]" in tex
    assert "\\begin{tikzpicture}" in tex
    assert "\\begin{axis}[" in tex
    assert "\\end{axis}" in tex
    assert "\\end{figure}" in tex


def test_render_plot_label_format():
    tex = _single_model_max_fixture({"Argmax": 0.65, "EmpiricalCV": 0.72})
    assert "\\label{fig:ci-fake_ds-f2}" in tex


def test_render_plot_caption_includes_direction_and_dataset():
    tex = _single_model_max_fixture({"Argmax": 0.65, "EmpiricalCV": 0.72})
    assert "higher is better" in tex
    assert "fake\\_ds" in tex
    # The display name of the model appears in the caption.
    assert "CatBoost" in tex


def test_render_plot_max_family_uses_higher_is_better():
    tex = _single_model_max_fixture({"Argmax": 0.65, "EmpiricalCV": 0.72})
    assert "higher is better" in tex
    assert "lower is better" not in tex


def test_render_plot_one_addplot_per_strategy_with_data():
    tex = _single_model_max_fixture(
        {
            "Argmax": 0.65,
            "Empirical": 0.70,
            "EmpiricalCV": 0.72,
        }
    )
    assert tex.count("\\addplot") == 3


def test_render_plot_skips_strategies_with_only_nan_values():
    """A strategy whose mean is NaN should not contribute an \\addplot
    block or a y-axis row."""
    df = _binary_max_slice()
    df_mean, df_se = pivot_dataset_objective(df)
    # Inject a fully-NaN row for a recognised strategy.
    df_mean.loc["Empirical"] = float("nan")
    df_se.loc["Empirical"] = float("nan")
    tex = render_plot(
        dataset="fake_ds",
        objective="F2",
        family="max",
        df_mean=df_mean,
        df_se=df_se,
    )
    assert ",Empirical}" not in tex


def test_render_plot_uses_full_strategy_names_no_abbreviations():
    """We dropped the STRATEGY_SHORTEN map; long names appear verbatim on
    the y-axis."""
    df = pd.DataFrame(
        [
            {
                "dataset_name": "fake_ds",
                "model_name": "CatBoost",
                "optimization_objective_name": "F2",
                "strategy_name": "DirichletCalibrationAnalytical",
                "objective_value": 0.7 + 0.001 * s,
            }
            for s in range(3)
        ]
    )
    df_mean, df_se = pivot_dataset_objective(df)
    tex = render_plot(
        dataset="fake_ds",
        objective="F2",
        family="max",
        df_mean=df_mean,
        df_se=df_se,
    )
    assert "DirichletCalibrationAnalytical" in tex
    # No previously-defined abbreviation should have leaked in.
    assert "DirichletAna" not in tex


def test_render_plot_rejects_multi_model_pivot():
    """render_plot now operates strictly on single-model slices; passing a
    pivot with multiple model columns is a programmer error."""
    rows = []
    for mdl in ("LGBMClassifier()", "XGBoost"):
        for _ in range(3):
            rows.append(
                {
                    "dataset_name": "fake_ds",
                    "model_name": mdl,
                    "optimization_objective_name": "F2",
                    "strategy_name": "Argmax",
                    "objective_value": 0.5,
                }
            )
    df = pd.DataFrame(rows)
    df_mean, df_se = pivot_dataset_objective(df)
    with pytest.raises(ValueError, match="single-model slice"):
        render_plot(
            dataset="fake_ds",
            objective="F2",
            family="max",
            df_mean=df_mean,
            df_se=df_se,
        )


# Sort + emphasis (bold winner + italic ties)


def test_single_model_sorts_strategies_by_mean_for_max_metric():
    """Highest mean appears first in symbolic y coords when family=max."""
    tex = _single_model_max_fixture(
        {
            "Argmax": 0.55,
            "Empirical": 0.65,
            "EmpiricalCV": 0.75,
        }
    )
    sym_line = next(
        line for line in tex.splitlines() if "symbolic y coords" in line
    )
    pos_cv = sym_line.index("EmpiricalCV")
    pos_emp = sym_line.index(",{Empirical},")
    pos_def = sym_line.index("Argmax")
    assert pos_cv < pos_emp < pos_def


def test_single_model_sort_for_cost_metric_is_ascending():
    """For cost objectives the lowest mean wins, so sort is ascending."""
    rows = []
    for strat, m in (
        ("Argmax", 5.0),
        ("Analytical", 0.5),
        ("EmpiricalCV", 0.3),
    ):
        for _ in range(3):
            rows.append(
                {
                    "dataset_name": "fake_ds",
                    "model_name": "CatBoost",
                    "optimization_objective_name": "Symmetric Cost",
                    "strategy_name": strat,
                    "objective_value": m,
                }
            )
    df = pd.DataFrame(rows)
    df_mean, df_se = pivot_dataset_objective(df)
    tex = render_plot(
        dataset="fake_ds",
        objective="Symmetric Cost",
        family="cost",
        df_mean=df_mean,
        df_se=df_se,
    )
    sym_line = next(
        line for line in tex.splitlines() if "symbolic y coords" in line
    )
    pos_cv = sym_line.index("EmpiricalCV")
    pos_ana = sym_line.index("Analytical")
    pos_def = sym_line.index("Argmax")
    assert pos_cv < pos_ana < pos_def


def test_single_model_italicizes_unique_winner_label():
    """The strategy with the best mean gets a \\textit-wrapped tick label."""
    tex = _single_model_max_fixture(
        {
            "Argmax": 0.50,
            "Empirical": 0.60,
            "EmpiricalCV": 0.80,  # clear winner, gap >> SE
        }
    )
    yticklabels_line = next(
        line for line in tex.splitlines() if "yticklabels=" in line
    )
    assert "\\textit{EmpiricalCV}" in yticklabels_line
    # Argmax and Empirical should NOT be italicized.
    assert "\\textit{Argmax}" not in yticklabels_line
    assert "\\textit{Empirical}" not in yticklabels_line


def test_single_model_italicizes_ties_with_winner():
    """A strategy whose mean ± SE overlaps the winner's interval should
    also be wrapped in \\textit (matching the appendix table convention)."""
    rows = []
    fixed_values = {
        "Argmax": [0.50, 0.50, 0.50],  # SE=0, far from winner
        "Empirical": [0.78, 0.79, 0.80],  # mean 0.79, SE ~0.006
        "EmpiricalCV": [0.79, 0.80, 0.81],  # mean 0.80, SE ~0.006 -> ties
    }
    for strat, vals in fixed_values.items():
        for v in vals:
            rows.append(
                {
                    "dataset_name": "fake_ds",
                    "model_name": "CatBoost",
                    "optimization_objective_name": "F2",
                    "strategy_name": strat,
                    "objective_value": v,
                }
            )
    df = pd.DataFrame(rows)
    df_mean, df_se = pivot_dataset_objective(df)
    tex = render_plot(
        dataset="fake_ds",
        objective="F2",
        family="max",
        df_mean=df_mean,
        df_se=df_se,
    )
    yticklabels_line = next(
        line for line in tex.splitlines() if "yticklabels=" in line
    )
    assert "\\textit{EmpiricalCV}" in yticklabels_line
    assert "\\textit{Empirical}" in yticklabels_line  # tied -> also italic
    # Bold should not appear anywhere.
    assert "\\textbf{" not in yticklabels_line
    # Argmax (mean 0.5, SE 0) is far away -> plain text.
    assert "\\textit{Argmax}" not in yticklabels_line


def test_single_model_caption_documents_emphasis_convention():
    """The caption has to tell the reader what italicizing means."""
    tex = _single_model_max_fixture({"Argmax": 0.5, "EmpiricalCV": 0.7})
    assert "italicized" in tex
    # No bold emphasis is used anywhere now.
    assert "\\textbf{" not in tex


# write_ci_plots


def test_write_ci_plots_one_file_per_dataset_objective(tmp_path):
    df = _multi_dataset_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"

    written = write_ci_plots(
        parquet_path=parquet, plots_dir=out_dir, scope="binary"
    )

    # 2 datasets x 2 objectives x 1 model = 4 plot files.
    assert {p.name for p in written} == {
        "dataset_a_symmetric_cost.tex",
        "dataset_b_symmetric_cost.tex",
        "dataset_a_f1.tex",
        "dataset_b_f1.tex",
    }
    for p in written:
        assert p.read_text(encoding="utf-8").startswith(
            "% Auto-generated by experiments.analysis.generate_ci_plots"
        )


def test_write_ci_plots_skips_unrecognized_objective(tmp_path):
    """Objectives outside SCOPE_CONFIG[scope]'s cost / max sets are ignored
    rather than raising, so a mixed-scope parquet doesn't crash."""
    df = _multi_dataset_df()
    extra = pd.DataFrame(
        [
            {
                "dataset_name": "dataset_a",
                "model_name": "LGBMClassifier()",
                "optimization_objective_name": "MysteryObjective",
                "strategy_name": "Argmax",
                "objective_value": 0.5,
            }
        ]
    )
    df = pd.concat([df, extra], ignore_index=True)
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"

    written = write_ci_plots(
        parquet_path=parquet, plots_dir=out_dir, scope="binary"
    )

    names = {p.name for p in written}
    assert all("mystery" not in n for n in names)
    assert "dataset_a_symmetric_cost.tex" in names


def test_write_ci_plots_creates_output_directory(tmp_path):
    df = _multi_dataset_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "nonexistent" / "nested"
    assert not out_dir.exists()
    write_ci_plots(parquet_path=parquet, plots_dir=out_dir, scope="binary")
    assert out_dir.is_dir()


def test_write_ci_plots_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown scope"):
        write_ci_plots(scope="ternary")


def test_write_ci_plots_uses_latest_parquet_when_path_omitted(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gp, "PLOTS_DIR", tmp_path / "plots")
    import experiments.analysis.generate_tables as gt

    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    df = _multi_dataset_df()
    df.to_parquet(tmp_path / "results_binary_synth.parquet")
    written = write_ci_plots(scope="binary")
    assert len(written) == 4


def test_write_ci_plots_dataset_filter(tmp_path):
    df = _multi_dataset_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"

    written = write_ci_plots(
        parquet_path=parquet,
        plots_dir=out_dir,
        scope="binary",
        dataset_filter=("dataset_a",),
    )
    names = {p.name for p in written}
    assert all(n.startswith("dataset_a_") for n in names)
    assert not any(n.startswith("dataset_b_") for n in names)


def test_write_ci_plots_objective_filter(tmp_path):
    df = _multi_dataset_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"

    written = write_ci_plots(
        parquet_path=parquet,
        plots_dir=out_dir,
        scope="binary",
        objective_filter=("F1",),
    )
    names = {p.name for p in written}
    assert all(n.endswith("_f1.tex") for n in names)
    assert not any("symmetric_cost" in n for n in names)


def test_write_ci_plots_model_filter_renders_named_model(tmp_path):
    """model_filter restricts the data to the named base classifiers; the
    caption then names that model."""
    df = _multi_dataset_df()  # has only LGBMClassifier()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"

    written = write_ci_plots(
        parquet_path=parquet,
        plots_dir=out_dir,
        scope="binary",
        model_filter=("LGBMClassifier()",),
    )
    for p in written:
        assert "LGBM" in p.read_text(encoding="utf-8")


def test_write_ci_plots_combined_filters_select_single_cell(tmp_path):
    """The combined (dataset, model, objective) filter that
    write_showcase_plots applies to a resolved cell yields exactly one
    plot file for that cell."""
    df = _multi_dataset_df()
    rows = []
    for strat in ("Argmax", "EmpiricalCV"):
        for _ in range(2):
            rows.append(
                {
                    "dataset_name": "mammography",
                    "model_name": "CatBoost",
                    "optimization_objective_name": "F2",
                    "strategy_name": strat,
                    "objective_value": 0.7 if strat == "EmpiricalCV" else 0.6,
                }
            )
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"

    written = write_ci_plots(
        parquet_path=parquet,
        plots_dir=out_dir,
        scope="binary",
        dataset_filter=("mammography",),
        objective_filter=("F2",),
        model_filter=("CatBoost",),
    )
    assert {p.name for p in written} == {"mammography_f2.tex"}


# write_showcase_plots


def test_write_showcase_plots_emits_one_file_per_entry(tmp_path):
    """write_showcase_plots writes one output per
    (scope, dataset, model, objective) tuple."""
    rows = []
    for ds, mdl, obj, val in [
        ("alpha", "LGBMClassifier()", "F2", 0.7),
        ("beta", "XGBoost", "Symmetric Cost", 0.05),
    ]:
        for strat in ("Argmax", "EmpiricalCV"):
            for seed in range(2):
                rows.append(
                    {
                        "dataset_name": ds,
                        "model_name": mdl,
                        "optimization_objective_name": obj,
                        "strategy_name": strat,
                        "objective_value": val + 0.01 * seed,
                    }
                )
    df = pd.DataFrame(rows)
    parquet = tmp_path / "results_binary_synth.parquet"
    df.to_parquet(parquet)
    import unittest.mock as um

    import experiments.analysis.generate_tables as gt

    with um.patch.object(gt, "RESULTS_DIR", tmp_path):
        showcase = (
            ("binary", "alpha", "LGBMClassifier()", "F2", "linear"),
            ("binary", "beta", "XGBoost", "Symmetric Cost", "linear"),
        )
        out_dir = tmp_path / "out"
        written = write_showcase_plots(plots_dir=out_dir, showcase=showcase)
    assert {p.name for p in written} == {
        "alpha_f2.tex",
        "beta_symmetric_cost.tex",
    }


def test_render_plot_log_scale_emits_xmode_log():
    """x_scale='log' switches the x axis to log mode."""
    rows = []
    for strat, m in (
        ("Argmax", 5.0),
        ("Analytical", 0.5),
        ("EmpiricalCV", 0.3),
    ):
        for _ in range(3):
            rows.append(
                {
                    "dataset_name": "fake_ds",
                    "model_name": "CatBoost",
                    "optimization_objective_name": "Asymmetric 100x Cost",
                    "strategy_name": strat,
                    "objective_value": m,
                }
            )
    df = pd.DataFrame(rows)
    df_mean, df_se = pivot_dataset_objective(df)
    tex = render_plot(
        dataset="fake_ds",
        objective="Asymmetric 100x Cost",
        family="cost",
        df_mean=df_mean,
        df_se=df_se,
        x_scale="log",
    )
    assert "xmode=log" in tex
    # Filename suffix and label suffix differentiate scales.
    assert "fig:ci-fake_ds-asymmetric_100x_cost-log" in tex
    assert "log scale" in tex


def test_render_plot_sqrt_scale_uses_x_coord_trafo():
    """x_scale='sqrt' applies a square-root transform via x coord trafo."""
    rows = [
        {
            "dataset_name": "fake_ds",
            "model_name": "CatBoost",
            "optimization_objective_name": "Asymmetric 100x Cost",
            "strategy_name": "Argmax",
            "objective_value": 1.0,
        }
    ]
    df = pd.DataFrame(rows * 3)
    df_mean, df_se = pivot_dataset_objective(df)
    tex = render_plot(
        dataset="fake_ds",
        objective="Asymmetric 100x Cost",
        family="cost",
        df_mean=df_mean,
        df_se=df_se,
        x_scale="sqrt",
    )
    assert "x coord trafo" in tex
    assert "sqrt(#1)" in tex
    assert "fig:ci-fake_ds-asymmetric_100x_cost-sqrt" in tex
    assert "square-root scale" in tex


def test_render_plot_unknown_x_scale_raises():
    df = pd.DataFrame(
        [
            {
                "dataset_name": "fake_ds",
                "model_name": "CatBoost",
                "optimization_objective_name": "F2",
                "strategy_name": "Argmax",
                "objective_value": 0.5,
            }
        ]
        * 2
    )
    df_mean, df_se = pivot_dataset_objective(df)
    with pytest.raises(ValueError, match="Unknown x_scale"):
        render_plot(
            dataset="fake_ds",
            objective="F2",
            family="max",
            df_mean=df_mean,
            df_se=df_se,
            x_scale="logarithmic",
        )
