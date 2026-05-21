import dataclasses

import pandas as pd
import pytest

from experiments.cloud.merge_results import merge
from experiments.common import config


def _make_row(**overrides):
    row = {
        "dataset_name": "sonar",
        "model_name": "LGBM",
        "strategy_name": "Argmax",
        "optimization_objective_name": "F1",
        "random_seed": 0,
        "objective_value": 0.8,
        "test_size": 0.2,
        "val_size": 0.1,
        "noise_level": 0.0,
        "learned_params": 0.5,
        "model_roc_auc_split": 0.85,
        "model_roc_auc_full": 0.9,
        "model_mean_vertex_distance_split": 0.12,
        "model_mean_vertex_distance_full": 0.10,
        "model_ece_split": 0.05,
        "model_ece_full": 0.04,
        "strategy_fit_seconds": 0.42,
    }
    row.update(overrides)
    return row


def _write_parquet(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_make_row_covers_experiment_result_schema():
    """Test fixture must include every column the pipeline writes to
    parquet, so merge tests exercise realistic row shape. Catches
    fixture drift when a new field is added to ExperimentResult.
    """
    expected = {
        f.name for f in dataclasses.fields(config.ExperimentConfig)
    } | {
        f.name
        for f in dataclasses.fields(config.ExperimentResult)
        if f.name != "config"
    }
    missing = expected - set(_make_row().keys())
    assert not missing, f"_make_row missing columns: {missing}"


def test_merge_combines_non_overlapping(tmp_path):
    """Two dirs with different models should produce all rows."""
    cpu_dir = tmp_path / "run_cpu"
    gpu_dir = tmp_path / "run_gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()

    _write_parquet(cpu_dir / "sonar_seed0.parquet", [_make_row()])
    _write_parquet(
        gpu_dir / "sonar_seed0.parquet",
        [_make_row(model_name="TabM", objective_value=0.82)],
    )

    out = tmp_path / "merged.parquet"
    df = merge([cpu_dir, gpu_dir], out)

    assert len(df) == 2
    assert set(df["model_name"]) == {"LGBM", "TabM"}
    assert out.exists()


def test_merge_deduplicates_keeps_last(tmp_path):
    """Duplicate (dataset, model, strategy, obj, seed) keeps last."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    _write_parquet(
        dir1 / "sonar_seed0.parquet",
        [_make_row(objective_value=0.7)],
    )
    _write_parquet(
        dir2 / "sonar_seed0.parquet",
        [_make_row(objective_value=0.9)],
    )

    out = tmp_path / "merged.parquet"
    df = merge([dir1, dir2], out)

    assert len(df) == 1
    assert df.iloc[0]["objective_value"] == 0.9


def test_merge_multiple_seeds(tmp_path):
    """Multiple seeds across dirs should all be preserved."""
    dir1 = tmp_path / "run1"
    dir1.mkdir()

    _write_parquet(
        dir1 / "sonar_seed0.parquet",
        [_make_row(random_seed=0)],
    )
    _write_parquet(
        dir1 / "sonar_seed1.parquet",
        [_make_row(random_seed=1)],
    )

    out = tmp_path / "merged.parquet"
    df = merge([dir1], out)

    assert len(df) == 2
    assert set(df["random_seed"]) == {0, 1}


def test_merge_empty_dir_warns(tmp_path, capsys):
    """A dir with no parquets should warn but not crash."""
    empty = tmp_path / "run_empty"
    full = tmp_path / "run_full"
    empty.mkdir()
    full.mkdir()

    _write_parquet(full / "sonar_seed0.parquet", [_make_row()])

    out = tmp_path / "merged.parquet"
    df = merge([empty, full], out)

    assert len(df) == 1
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()


def test_merge_all_empty_exits(tmp_path):
    """All empty dirs should raise SystemExit."""
    empty = tmp_path / "run_empty"
    empty.mkdir()

    out = tmp_path / "merged.parquet"
    with pytest.raises(SystemExit, match="No parquet"):
        merge([empty], out)


def test_merge_creates_parent_dirs(tmp_path):
    """Output path with non-existent parents should be created."""
    dir1 = tmp_path / "run1"
    dir1.mkdir()
    _write_parquet(dir1 / "sonar_seed0.parquet", [_make_row()])

    out = tmp_path / "nested" / "deep" / "merged.parquet"
    df = merge([dir1], out)

    assert out.exists()
    assert len(df) == 1


def test_merge_warns_on_schema_mismatch_but_keeps_all_rows(tmp_path, capsys):
    """A CPU/GPU schema drift must be announced loudly and must not
    drop rows (gap columns become NaN, but every row survives)."""
    cpu_dir = tmp_path / "run_cpu"
    gpu_dir = tmp_path / "run_gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()

    _write_parquet(cpu_dir / "sonar_seed0.parquet", [_make_row()])
    # GPU run drifted: a subset of columns plus an unexpected extra one.
    _write_parquet(
        gpu_dir / "sonar_seed1.parquet",
        [
            {
                "dataset_name": "sonar",
                "model_name": "TabM",
                "strategy_name": "Argmax",
                "optimization_objective_name": "F1",
                "random_seed": 1,
                "objective_value": 0.5,
                "rogue_col": 1,
            }
        ],
    )

    out = tmp_path / "merged.parquet"
    df = merge([cpu_dir, gpu_dir], out)

    captured = capsys.readouterr()
    assert "SCHEMA MISMATCH" in captured.out
    assert len(df) == 2  # no rows lost
    assert "rogue_col" in df.columns
    assert df["rogue_col"].isna().any()
