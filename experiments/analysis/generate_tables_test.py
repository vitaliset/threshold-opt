"""Unit tests for experiments/analysis/generate_tables.py."""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

import experiments.analysis.generate_tables as gt
from experiments.analysis.generate_tables import (
    _find_latest_parquet,
    _sanitize_filename,
    build_model_auc_latex,
    build_model_ece_latex,
    build_model_vertex_latex,
    build_rotational_bias_latex,
    write_model_auc_tables,
    write_model_diagnostic_tables,
    write_rotational_bias_table,
    write_strategy_tables,
)


def _rotational_bias_df(
    seed_count: int = 5,
    voronoi_bonus: float = 0.1,
    *,
    tie: bool = False,
) -> pd.DataFrame:
    """Synthetic rotational-bias results.

    If *tie* is False, Voronoi wins by *voronoi_bonus* and the affine
    strategies trail with slight offsets.  If *tie* is True, all five
    strategies share the same base score (within seed noise) so their
    95% CIs overlap.
    """
    rows = []
    if tie:
        base_scores = dict.fromkeys(
            (
                "Argmax",
                "ThresholdRatio",
                "MultidimensionalThreshold",
                "VoronoiTuning",
                "FullMatrixPartitioning",
            ),
            0.80,
        )
    else:
        base_scores = {
            "Argmax": 0.80,
            "ThresholdRatio": 0.81,
            "MultidimensionalThreshold": 0.82,
            "FullMatrixPartitioning": 0.83,
            "VoronoiTuning": 0.80 + voronoi_bonus,
        }
    for metric in ("Accuracy", "Macro-F1"):
        for strat, base in base_scores.items():
            for seed in range(seed_count):
                rows.append(
                    {
                        "metric": metric,
                        "seed": seed,
                        "strategy": strat,
                        "score": base + 0.001 * seed,
                    }
                )
    return pd.DataFrame(rows)


def _strategy_tables_df() -> pd.DataFrame:
    """Minimal synthetic binary results covering two datasets."""
    rows = []
    for ds in ("DatasetA", "DatasetB"):
        for obj in ("F1",):
            for strat in ("StratA", "StratB"):
                for seed in range(3):
                    rows.append(
                        {
                            "dataset_name": ds,
                            "model_name": "ModelX",
                            "optimization_objective_name": obj,
                            "strategy_name": strat,
                            "objective_value": 0.8
                            + 0.01 * seed
                            + (0.1 if strat == "StratB" else 0.0),
                        }
                    )
    return pd.DataFrame(rows)


def _model_auc_df() -> pd.DataFrame:
    """Synthetic binary results for the base-model ROC AUC table.

    Uses real dataset names (the builder filters rows to the study
    config's dataset list) and real ``MODELS`` keys. ``model_roc_auc_full``
    is replicated across two strategies and two objectives per
    ``(dataset, model, seed)`` to exercise the dedup. The
    ``(wdbc, LogisticRegression())`` cell is intentionally absent so it
    renders as an em-dash.
    """
    base_auc = {
        ("sonar", "LGBMClassifier()"): 0.90,
        ("sonar", "LogisticRegression()"): 0.80,
        ("wdbc", "LGBMClassifier()"): 0.99,
    }
    rows = []
    for (ds, model), base in base_auc.items():
        for seed in range(3):
            for strat in ("StratA", "StratB"):
                for obj in ("Symmetric Cost", "F1"):
                    rows.append(
                        {
                            "dataset_name": ds,
                            "model_name": model,
                            "random_seed": seed,
                            "strategy_name": strat,
                            "optimization_objective_name": obj,
                            "objective_value": 0.5,
                            "model_roc_auc_full": base + 0.001 * seed,
                        }
                    )
    return pd.DataFrame(rows)


# --- build_model_auc_latex --------------------------------------------------


def test_build_model_auc_latex_structure():
    tex = build_model_auc_latex(_model_auc_df(), "binary")
    # Upright (no landscape page); scaled to the text width instead.
    assert "\\begin{landscape}" not in tex
    assert "\\begin{table}[H]" in tex
    assert "\\adjustbox{max width=\\linewidth}" in tex
    assert "\\label{tab:model-auc-binary}" in tex
    # Transposed (transpose=True): datasets are row labels (config
    # order), models are column headers (MODELS order, so LGBM before
    # LogReg). The header row leads with "Dataset".
    header = next(
        ln for ln in tex.splitlines() if ln.strip().startswith("Dataset")
    )
    assert "LGBM" in header and "LogReg" in header
    assert header.index("LGBM") < header.index("LogReg")
    assert any(ln.strip().startswith("sonar") for ln in tex.splitlines())
    assert any(ln.strip().startswith("wdbc") for ln in tex.splitlines())


def test_build_model_auc_latex_dedups_then_means_over_seeds():
    """The 4 duplicated (strategy, objective) rows per seed collapse to
    one; mean over seeds of 0.90 + 0.001*seed (seed in 0..2) is 0.901."""
    tex = build_model_auc_latex(_model_auc_df(), "binary")
    assert "0.901" in tex


def test_build_model_auc_latex_bolds_per_dataset_best():
    """LGBM (~0.901) beats LogReg (~0.801) on sonar by far more than the
    SE, so the sonar LGBM cell is bolded and the LogReg cell is not."""
    tex = build_model_auc_latex(_model_auc_df(), "binary")
    assert "\\textbf{0.901" in tex
    assert "\\textbf{0.801" not in tex


def test_build_model_auc_latex_missing_cell_is_emdash():
    """(wdbc, LogisticRegression()) has no rows. wdbc is a row and
    LogReg a column in the transposed layout, so the wdbc row carries
    the em-dash for its LogReg cell while the fully populated sonar row
    has none."""
    tex = build_model_auc_latex(_model_auc_df(), "binary")
    wdbc_line = next(
        ln for ln in tex.splitlines() if ln.strip().startswith("wdbc")
    )
    sonar_line = next(
        ln for ln in tex.splitlines() if ln.strip().startswith("sonar")
    )
    assert "—" in wdbc_line
    assert "—" not in sonar_line


def test_build_model_auc_latex_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown scope"):
        build_model_auc_latex(_model_auc_df(), "nope")


# --- write_model_auc_tables -------------------------------------------------


def test_write_model_auc_tables_writes_scoped_file(tmp_path):
    df = _model_auc_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    out = write_model_auc_tables(
        scope="binary", parquet_path=parquet, tables_dir=out_dir
    )
    assert out.name == "model_auc_binary.tex"
    assert out.read_text(encoding="utf-8") == build_model_auc_latex(
        df, "binary"
    )


def test_write_model_auc_tables_missing_parquet_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No parquet files"):
        write_model_auc_tables(scope="binary", tables_dir=tmp_path)


# --- build_model_ece_latex / build_model_vertex_latex -----------------------


def _model_diag_df() -> pd.DataFrame:
    """Synthetic binary results for the ECE / vertex-distance tables.

    Real dataset names and ``MODELS`` keys. ``model_ece_full`` and
    ``model_mean_vertex_distance_full`` are replicated across two
    strategies and two objectives per ``(dataset, model, seed)`` to
    exercise the dedup. On sonar, LGBM has a far lower ECE than LogReg.
    ``(wdbc, LogisticRegression())`` is intentionally absent so it
    renders as an em-dash.
    """
    base = {
        ("sonar", "LGBMClassifier()"): (0.02, 0.30),
        ("sonar", "LogisticRegression()"): (0.15, 0.10),
        ("wdbc", "LGBMClassifier()"): (0.01, 0.40),
    }
    rows = []
    for (ds, model), (ece, vtx) in base.items():
        for seed in range(3):
            for strat in ("StratA", "StratB"):
                for obj in ("Symmetric Cost", "F1"):
                    rows.append(
                        {
                            "dataset_name": ds,
                            "model_name": model,
                            "random_seed": seed,
                            "strategy_name": strat,
                            "optimization_objective_name": obj,
                            "objective_value": 0.5,
                            "model_ece_full": ece + 0.001 * seed,
                            "model_mean_vertex_distance_full": (
                                vtx + 0.001 * seed
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_build_model_ece_latex_structure():
    tex = build_model_ece_latex(_model_diag_df(), "binary")
    assert "\\begin{table}[H]" in tex
    assert "\\adjustbox{max width=\\linewidth}" in tex
    assert "\\label{tab:model-ece-binary}" in tex
    # Transposed: models are row labels, datasets are column headers.
    assert tex.index("LGBM") < tex.index("LogReg")
    assert "sonar" in tex and "wdbc" in tex


def test_build_model_ece_latex_bolds_per_dataset_lowest():
    """Lower ECE is better. On sonar LGBM (~0.021) beats LogReg
    (~0.151) by far more than the SE, so the LGBM cell is bolded and
    the LogReg cell is not."""
    tex = build_model_ece_latex(_model_diag_df(), "binary")
    assert "\\textbf{0.021" in tex
    assert "\\textbf{0.151" not in tex


def test_build_model_ece_latex_missing_cell_is_emdash():
    tex = build_model_ece_latex(_model_diag_df(), "binary")
    logreg_line = next(
        ln for ln in tex.splitlines() if ln.strip().startswith("LogReg")
    )
    assert "—" in logreg_line


def test_build_model_vertex_latex_never_bolds():
    """Vertex distance has no better/worse direction, so no cell is
    bolded even when one model dominates."""
    tex = build_model_vertex_latex(_model_diag_df(), "binary")
    assert "\\textbf" not in tex
    assert "\\label{tab:model-vertex-binary}" in tex
    # Dedup then mean over seeds of 0.30 + 0.001*seed (seed 0..2).
    assert "0.301" in tex


def test_build_model_vertex_latex_missing_cell_is_emdash():
    tex = build_model_vertex_latex(_model_diag_df(), "binary")
    logreg_line = next(
        ln for ln in tex.splitlines() if ln.strip().startswith("LogReg")
    )
    assert "—" in logreg_line


def test_build_model_ece_latex_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown scope"):
        build_model_ece_latex(_model_diag_df(), "nope")


# --- write_model_diagnostic_tables ------------------------------------------


@pytest.mark.parametrize(
    ("kind", "builder", "expected_name"),
    [
        ("auc", build_model_auc_latex, "model_auc_binary.tex"),
        ("ece", build_model_ece_latex, "model_ece_binary.tex"),
        (
            "vertex",
            build_model_vertex_latex,
            "model_vertex_distance_binary.tex",
        ),
    ],
)
def test_write_model_diagnostic_tables_writes_scoped_file(
    tmp_path, kind, builder, expected_name
):
    df = _model_auc_df() if kind == "auc" else _model_diag_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    out = write_model_diagnostic_tables(
        kind, scope="binary", parquet_path=parquet, tables_dir=out_dir
    )
    assert out.name == expected_name
    assert out.read_text(encoding="utf-8") == builder(df, "binary")


def test_write_model_diagnostic_tables_unknown_kind_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown kind"):
        write_model_diagnostic_tables(
            "nope", scope="binary", tables_dir=tmp_path
        )


def test_write_model_diagnostic_tables_missing_parquet_raises(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No parquet files"):
        write_model_diagnostic_tables(
            "ece", scope="binary", tables_dir=tmp_path
        )


# --- _sanitize_filename -----------------------------------------------------


def test_sanitize_filename_lowercases_and_replaces_spaces():
    assert _sanitize_filename("Foo Bar") == "foo_bar"
    assert _sanitize_filename("ALLCAPS") == "allcaps"
    assert _sanitize_filename("a b  c") == "a_b__c"


# --- build_rotational_bias_latex -------------------------------------------


def test_build_rotational_bias_latex_structure():
    tex = build_rotational_bias_latex(_rotational_bias_df())
    assert "\\begin{table}[H]" in tex
    # Transposed layout: strategies are rows, the two metrics are the
    # only columns, so the column spec is l + one c per metric.
    assert "\\begin{tabular}{lcc}" in tex
    assert "\\label{tab:results-rotational-bias}" in tex
    for metric in ("Accuracy", "Macro-F1"):
        assert metric in tex
    for strat in (
        "Argmax",
        "ThresholdRatio",
        "MultidimensionalThreshold",
        "VoronoiTuning",
        "FullMatrixPartitioning",
    ):
        assert strat in tex


def test_build_rotational_bias_latex_bolds_clear_winner():
    """Voronoi's lead (~0.10) is far larger than the CI half-width, so it
    is the unique best in both rows and should be bolded exactly twice."""
    tex = build_rotational_bias_latex(_rotational_bias_df(voronoi_bonus=0.1))
    assert tex.count("\\textbf{") == 2


def test_build_rotational_bias_latex_bolds_all_ties_when_overlap():
    """When scores are identical up to noise, every strategy's CI overlaps
    with the best, so every cell in that row should be bolded."""
    df = _rotational_bias_df(tie=True)
    tex = build_rotational_bias_latex(df)
    # 2 metrics x 5 strategies all within the overlap band
    assert tex.count("\\textbf{") == 2 * 5


def test_build_rotational_bias_latex_decimals_respected():
    tex = build_rotational_bias_latex(_rotational_bias_df(), decimals=2)
    # decimals=2 means means/CIs look like "0.80" not "0.800"
    assert "0.80 ±" in tex or "0.81 ±" in tex
    assert "0.800 ±" not in tex


# --- write_rotational_bias_table -------------------------------------------


def test_write_rotational_bias_table_with_df(tmp_path):
    df = _rotational_bias_df()
    out = write_rotational_bias_table(df=df, tables_dir=tmp_path)
    assert out.exists()
    assert out.name == "results_rotational_bias.tex"
    assert out.read_text(encoding="utf-8") == build_rotational_bias_latex(df)


def test_write_rotational_bias_table_with_parquet(tmp_path):
    df = _rotational_bias_df()
    parquet = tmp_path / "input.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = write_rotational_bias_table(parquet_path=parquet, tables_dir=out_dir)
    assert out.read_text(encoding="utf-8") == build_rotational_bias_latex(df)


def test_write_rotational_bias_table_missing_parquet_raises(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="results_rotational_bias"):
        write_rotational_bias_table(tables_dir=tmp_path)


# --- _find_latest_parquet ---------------------------------------------------


def test_find_latest_parquet_returns_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    old = tmp_path / "results_binary_old.parquet"
    new = tmp_path / "results_binary_new.parquet"
    old.touch()
    new.touch()
    old_time = time.time() - 100
    os.utime(old, (old_time, old_time))
    assert _find_latest_parquet() == new


def test_find_latest_parquet_searches_run_subdirs(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    run_dir = tmp_path / "run_results_binary_ts"
    run_dir.mkdir()
    nested = run_dir / "results_binary_nested.parquet"
    nested.touch()
    assert _find_latest_parquet() == nested


def test_find_latest_parquet_skips_profile_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    profile = tmp_path / "results_binary_profile_20260101.parquet"
    real = tmp_path / "results_binary_20260101.parquet"
    profile.touch()
    real.touch()
    assert _find_latest_parquet() == real


def test_find_latest_parquet_raises_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No parquet files"):
        _find_latest_parquet()


# --- write_strategy_tables --------------------------------------------------


def test_write_strategy_tables_one_file_per_dataset(tmp_path):
    df = _strategy_tables_df()
    parquet = tmp_path / "results_binary_fake.parquet"
    df.to_parquet(parquet)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    written = write_strategy_tables(
        parquet_path=parquet,
        tables_dir=out_dir,
        datasets=("DatasetA", "DatasetB"),
    )

    # Fixture uses F1 (a maximization metric), so only the max-variant
    # file is emitted per dataset.
    assert {p.name for p in written} == {
        "results_dataseta_max.tex",
        "results_datasetb_max.tex",
    }
    for p in written:
        content = p.read_text(encoding="utf-8")
        assert "\\begin{table}[H]" in content
        assert "StratA" in content
        assert "StratB" in content
