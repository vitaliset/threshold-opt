import dataclasses
import logging
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import metrics as sklearn_metrics

from experiments.common import config, pipeline
from threshold_opt import base, cv, metrics


class MockModel(sklearn_base.BaseEstimator):
    def fit(self, X, y):
        self.fitted_ = True
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MockStrategy(base.DecisionStrategy):
    def _fit(self, X, y, optimization_objective):
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def setup():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 3, 100)
    dataset_name = "test_dataset"
    model_name = "MockModel"
    strat_name = "MockStrategy"
    objective_name = "Accuracy"
    metric_objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    models = {model_name: MockModel()}
    strategies = {strat_name: lambda m: MockStrategy(m)}
    objectives = {objective_name: metric_objective}
    return dict(
        X=X,
        y=y,
        dataset_name=dataset_name,
        model_name=model_name,
        strat_name=strat_name,
        objective_name=objective_name,
        metric_objective=metric_objective,
        models=models,
        strategies=strategies,
        objectives=objectives,
    )


def test_split_data_success(setup):
    splits = pipeline._split_data(setup["X"], setup["y"], random_seed=42)
    assert splits is not None
    (
        _X_train,
        _X_val,
        _X_test,
        y_train,
        y_val,
        y_test,
        _X_train_full,
        y_train_full,
    ) = splits

    assert len(y_test) == 20
    assert len(y_train_full) == 80
    assert len(y_val) == 10
    assert len(y_train) == 70


def test_split_data_fail_small_dataset(caplog):
    X_small = np.random.rand(3, 4)
    y_small = np.array([0, 1, 2])

    with caplog.at_level(logging.ERROR, logger="experiments.common.pipeline"):
        splits = pipeline._split_data(X_small, y_small, random_seed=42)

    assert splits is None
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_train_model_success(setup):
    model = pipeline._train_model(
        "test_model", MockModel(), "test_ds", setup["X"], setup["y"]
    )
    assert isinstance(model, MockModel)
    assert model.fitted_ is True
    assert model.n_features_ == setup["X"].shape[1]


def test_train_model_fail(setup, caplog):
    class FailingModel(sklearn_base.BaseEstimator):
        def fit(self, X, y):
            raise ValueError("Boom")

    with caplog.at_level(logging.ERROR, logger="experiments.common.pipeline"):
        model = pipeline._train_model(
            "fail_model", FailingModel(), "test_ds", setup["X"], setup["y"]
        )

    assert model is None
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_evaluate_result_success_metric(setup):
    strategy = MockStrategy(MockModel())
    y_test = setup["y"][:10]
    result = pipeline._evaluate_result(
        dataset_name=setup["dataset_name"],
        random_seed=42,
        model_name=setup["model_name"],
        strat_name=setup["strat_name"],
        objective_name=setup["objective_name"],
        strategy=strategy,
        opt_objective=setup["metric_objective"],
        X_test=setup["X"][:10],
        y_test=y_test,
        n_classes=3,
        param_extractor=None,
        val_size=0.2,
        noise_level=0.0,
    )
    assert isinstance(result, config.ExperimentResult)
    # MockStrategy predicts all zeros → accuracy = fraction of y_test == 0
    expected_accuracy = float(np.mean(y_test == 0))
    assert result.objective_value == pytest.approx(expected_accuracy)


def test_evaluate_result_success_cost_matrix(setup):
    strategy = MockStrategy(MockModel())
    cost_matrix = np.ones((3, 3)) - np.eye(3)
    y_test = setup["y"][:10]
    result = pipeline._evaluate_result(
        dataset_name=setup["dataset_name"],
        random_seed=42,
        model_name=setup["model_name"],
        strat_name=setup["strat_name"],
        objective_name="Cost",
        strategy=strategy,
        opt_objective=cost_matrix,
        X_test=setup["X"][:10],
        y_test=y_test,
        n_classes=3,
        param_extractor=None,
        val_size=0.2,
        noise_level=0.0,
    )
    assert isinstance(result, config.ExperimentResult)
    # MockStrategy predicts all zeros; cost[y,0]=1 for y!=0, 0 otherwise
    # → average cost = fraction of y_test != 0
    expected_cost = float(np.mean(y_test != 0))
    assert result.objective_value == pytest.approx(expected_cost)


def test_run_experiment_pipeline(setup):
    X_df = pd.DataFrame(setup["X"], columns=["a", "b", "c", "d"])
    results = pipeline.run_experiment_pipeline(
        dataset_name=setup["dataset_name"],
        X=X_df,
        y=setup["y"],
        random_seed=42,
        models=setup["models"],
        strategies=setup["strategies"],
        objectives=setup["objectives"],
    )
    assert len(results) == 1
    res = results[0]
    assert res.config.dataset_name == setup["dataset_name"]
    assert res.config.model_name == setup["model_name"]
    assert res.config.strategy_name == setup["strat_name"]


# Reproducibility: same seed -> identical outputs, different seed -> differ


def _repro_setup():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from threshold_opt import binary_strategies

    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=4,
        random_state=0,
    )
    models = {
        "LR": LogisticRegression(random_state=42, max_iter=500),
    }
    strategies = {
        "EmpiricalThreshold": binary_strategies.EmpiricalThreshold,
    }
    objectives = {
        "accuracy": metrics.MetricObjective(
            metric_fn=sklearn_metrics.accuracy_score,
            higher_is_better=True,
        ),
    }
    return X, y, models, strategies, objectives


def _threshold_extractor(strategy):
    return strategy.threshold


@pytest.mark.reproducibility
def test_run_experiment_pipeline_same_seed_is_deterministic():
    """Two runs with identical seed produce identical objective values."""
    X, y, models, strategies, objectives = _repro_setup()
    r1 = pipeline.run_experiment_pipeline(
        dataset_name="syn",
        X=X,
        y=y,
        random_seed=123,
        models=models,
        strategies=strategies,
        objectives=objectives,
        param_extractor=_threshold_extractor,
    )
    r2 = pipeline.run_experiment_pipeline(
        dataset_name="syn",
        X=X,
        y=y,
        random_seed=123,
        models=models,
        strategies=strategies,
        objectives=objectives,
        param_extractor=_threshold_extractor,
    )
    assert len(r1) == len(r2) == 1
    assert r1[0].objective_value == r2[0].objective_value
    assert r1[0].learned_params == r2[0].learned_params
    assert r1[0].model_roc_auc_split == r2[0].model_roc_auc_split
    assert r1[0].model_ece_split == r2[0].model_ece_split


@pytest.mark.reproducibility
def test_run_experiment_pipeline_different_seeds_differ():
    """Different seeds => different splits => different learned thresholds.

    We compare the *learned threshold* rather than the test objective
    because on a small test set the objective can collide across seeds
    (e.g. the same count of correct classifications out of 60). The
    threshold is a continuous quantity fit from the validation fold and
    is essentially never identical across seeds for a non-degenerate
    problem.
    """
    X, y, models, strategies, objectives = _repro_setup()
    thresholds = []
    for seed in (1, 2, 3):
        res = pipeline.run_experiment_pipeline(
            dataset_name="syn",
            X=X,
            y=y,
            random_seed=seed,
            models=models,
            strategies=strategies,
            objectives=objectives,
            param_extractor=_threshold_extractor,
        )
        thresholds.append(res[0].learned_params)
    # Not all thresholds are identical — varying the split must propagate.
    assert len(set(thresholds)) > 1


# _compute_ece


def test_compute_ece_perfect_calibration_is_zero():
    """A perfectly calibrated synthetic estimator yields ECE = 0."""
    rng = np.random.default_rng(0)
    # Sample y_true ~ Bernoulli(p) for known p, then set p_hat = p.
    probs = rng.uniform(0.0, 1.0, size=20_000)
    y_true = (rng.uniform(size=probs.size) < probs).astype(int)
    y_score = np.column_stack([1.0 - probs, probs])
    ece = pipeline._compute_ece(y_true, y_score, n_classes=2)
    assert ece is not None
    assert ece < 0.02


def test_compute_ece_constant_wrong_prediction_is_one():
    """If the model predicts P(y=1)=1.0 with constant confidence and is
    always wrong, ECE equals 1.0 (single bin, |0 - 1| weighted by 1)."""
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_score = np.column_stack([np.zeros(n), np.ones(n)])
    ece = pipeline._compute_ece(y_true, y_score, n_classes=2)
    assert ece == pytest.approx(1.0)


def test_compute_ece_multiclass_top_label():
    """Multiclass branch uses top-label calibration: confidence is
    argmax probability, target is correctness of the argmax."""
    # 3-class problem: argmax is always class 0 with confidence 0.6.
    # If labels are also class 0 for all samples, top-label ECE
    # measures |1.0 - 0.6| in a single bin: 0.4.
    n = 50
    y_true = np.zeros(n, dtype=int)
    y_score = np.tile([0.6, 0.2, 0.2], (n, 1))
    ece = pipeline._compute_ece(y_true, y_score, n_classes=3)
    assert ece == pytest.approx(0.4)


def test_compute_ece_empty_returns_none():
    ece = pipeline._compute_ece(
        np.array([], dtype=int), np.empty((0, 2)), n_classes=2
    )
    assert ece is None


# SkippedModelError


def test_train_model_skipped_model_error_logs_info_not_error(setup, caplog):
    class SkippingModel(sklearn_base.BaseEstimator):
        def fit(self, X, y):
            raise pipeline.SkippedModelError("too big")

    with caplog.at_level(logging.INFO, logger="experiments.common.pipeline"):
        result = pipeline._train_model(
            "skip_model", SkippingModel(), "test_ds", setup["X"], setup["y"]
        )

    assert result is pipeline._SKIPPED
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("skip" in r.getMessage().lower() for r in info_records)
    assert not any(r.levelno == logging.ERROR for r in caplog.records)


# export_results


def _make_result(
    dataset="ds", model="m", strategy="s", objective="F1", value=0.9
):
    cfg = config.ExperimentConfig(
        dataset_name=dataset,
        model_name=model,
        optimization_objective_name=objective,
        strategy_name=strategy,
        random_seed=0,
        test_size=0.2,
        val_size=0.1,
        noise_level=0.0,
    )
    return config.ExperimentResult(
        config=cfg,
        learned_params=0.5,
        objective_value=value,
        model_roc_auc_split=0.8,
        model_roc_auc_full=0.85,
        model_ece_split=0.05,
        model_ece_full=0.04,
    )


def test_export_results_writes_parquet(tmp_path):
    results = [_make_result(), _make_result(model="m2", value=0.7)]

    with mock.patch.object(pipeline, "RESULTS_DIR", tmp_path):
        pipeline.export_results(results, "test_prefix")

    parquet_files = list(tmp_path.glob("test_prefix_*.parquet"))
    assert len(parquet_files) == 1

    df = pd.read_parquet(parquet_files[0])
    assert len(df) == 2
    assert "objective_value" in df.columns


def test_export_results_custom_param_column(tmp_path):
    results = [_make_result()]

    with mock.patch.object(pipeline, "RESULTS_DIR", tmp_path):
        pipeline.export_results(
            results, "pfx", learned_params_column_name="threshold"
        )

    df = pd.read_parquet(next(tmp_path.glob("pfx_*.parquet")))
    assert "threshold" in df.columns


def test_export_results_empty_list_writes_empty_parquet(tmp_path):
    with mock.patch.object(pipeline, "RESULTS_DIR", tmp_path):
        pipeline.export_results([], "empty")

    files = list(tmp_path.glob("empty_*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert len(df) == 0


# Schema parity: every ExperimentResult field must reach the parquet


def _expected_parquet_columns() -> set[str]:
    """Column set the pipeline is expected to write, derived from the
    dataclasses. ExperimentConfig is unpacked into top-level columns
    (``**r.config.__dict__``), and every non-``config`` field on
    ExperimentResult becomes its own column under the default
    ``learned_params_column_name``.
    """
    return {f.name for f in dataclasses.fields(config.ExperimentConfig)} | {
        f.name
        for f in dataclasses.fields(config.ExperimentResult)
        if f.name != "config"
    }


def test_save_task_results_includes_all_experiment_result_fields(tmp_path):
    """Guard against forgetting to wire a newly added ExperimentResult
    field into ``_save_task_results``: every dataclass field must end up
    in the parquet.
    """
    cfg = config.ExperimentConfig(
        dataset_name="ds",
        model_name="m",
        optimization_objective_name="o",
        strategy_name="s",
        random_seed=0,
        test_size=0.2,
        val_size=0.1,
        noise_level=0.0,
    )
    result = config.ExperimentResult(config=cfg, objective_value=0.5)

    out = tmp_path / "out.parquet"
    pipeline._save_task_results([result], out)
    df = pd.read_parquet(out)

    missing = _expected_parquet_columns() - set(df.columns)
    assert not missing, (
        f"_save_task_results parquet missing columns: {missing}"
    )


def test_export_results_includes_all_experiment_result_fields(tmp_path):
    """Same guard for ``export_results``, the other serialization path."""
    cfg = config.ExperimentConfig(
        dataset_name="ds",
        model_name="m",
        optimization_objective_name="o",
        strategy_name="s",
        random_seed=0,
        test_size=0.2,
        val_size=0.1,
        noise_level=0.0,
    )
    result = config.ExperimentResult(config=cfg, objective_value=0.5)

    with mock.patch.object(pipeline, "RESULTS_DIR", tmp_path):
        pipeline.export_results([result], "schema_check")

    df = pd.read_parquet(next(tmp_path.glob("schema_check_*.parquet")))
    missing = _expected_parquet_columns() - set(df.columns)
    assert not missing, f"export_results parquet missing columns: {missing}"


# run_experiment_pipeline — needs_val / self_tunes_with_cv branching


class NeedsValStrategy(base.DecisionStrategy):
    needs_validation_set = True

    def _fit(self, X, y, opt_obj):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class CVTunesStrategy(base.DecisionStrategy):
    needs_validation_set = False
    self_tunes_with_cv = True

    def _fit(self, X, y, opt_obj):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class NoDataStrategy(base.DecisionStrategy):
    needs_validation_set = False

    def _fit(self, X, y, opt_obj):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_run_pipeline_needs_val_strategy_uses_val_split(setup):
    """needs_validation_set=True uses split model and val data."""
    fit_calls = []

    class TrackingStrategy(NeedsValStrategy):
        def _fit(self, X, y, opt_obj):
            fit_calls.append(len(X))
            return self

    _results = pipeline.run_experiment_pipeline(
        dataset_name="ds",
        X=setup["X"],
        y=np.array([0, 1] * 50),
        random_seed=42,
        models={"m": MockModel()},
        strategies={"s": lambda m: TrackingStrategy(m)},
        objectives=setup["objectives"],
    )
    # Strategy was fitted on val split (~10 samples), not full train
    assert len(fit_calls) > 0
    assert fit_calls[0] < 80  # val set is smaller than full train


def test_run_pipeline_cv_strategy_uses_full_train(setup):
    """A self_tunes_with_cv=True strategy must receive full training data."""
    fit_calls = []

    class TrackingCVStrategy(CVTunesStrategy):
        def _fit(self, X, y, opt_obj):
            fit_calls.append(len(X))
            return self

    _results = pipeline.run_experiment_pipeline(
        dataset_name="ds",
        X=setup["X"],
        y=np.array([0, 1] * 50),
        random_seed=42,
        models={"m": MockModel()},
        strategies={"s": lambda m: TrackingCVStrategy(m)},
        objectives=setup["objectives"],
    )
    assert len(fit_calls) > 0
    assert fit_calls[0] == 80  # full train_full set (100 * 0.8)


def test_run_pipeline_cv_skip_is_loud_and_not_silent(setup, caplog):
    """A CV strategy that cannot run K-fold (InsufficientCVData) must be
    skipped *loudly*: a greppable CV_SKIP warning is logged, the run
    continues (no exception), and that strategy contributes no result
    row instead of vanishing silently.
    """

    class CVImpossibleStrategy(CVTunesStrategy):
        def _fit(self, X, y, opt_obj):
            raise cv.InsufficientCVData("smallest class has 1 member")

    with caplog.at_level(
        logging.WARNING, logger="experiments.common.pipeline"
    ):
        results = pipeline.run_experiment_pipeline(
            dataset_name="ds",
            X=setup["X"],
            y=np.array([0, 1] * 50),
            random_seed=42,
            models={"m": MockModel()},
            strategies={"cv_imp": lambda m: CVImpossibleStrategy(m)},
            objectives=setup["objectives"],
        )

    assert results == []
    cv_skip = [r for r in caplog.records if "CV_SKIP" in r.getMessage()]
    assert cv_skip, "expected a CV_SKIP warning for the skipped strategy"
    assert cv_skip[0].levelno == logging.WARNING


def test_run_pipeline_no_val_no_cv_strategy_receives_full_train(setup):
    """A strategy with needs_val=False and no self_tunes_with_cv override
    still receives the full training set (not an empty sentinel)."""
    fit_calls = []

    class TrackingNoDataStrategy(NoDataStrategy):
        def _fit(self, X, y, opt_obj):
            fit_calls.append(len(X))
            return self

    _results = pipeline.run_experiment_pipeline(
        dataset_name="ds",
        X=setup["X"],
        y=np.array([0, 1] * 50),
        random_seed=42,
        models={"m": MockModel()},
        strategies={"s": lambda m: TrackingNoDataStrategy(m)},
        objectives=setup["objectives"],
    )
    assert len(fit_calls) > 0
    assert fit_calls[0] == 80  # full train_full set (100 * 0.8)


class MockProbaModel(sklearn_base.BaseEstimator):
    """MockModel with predict_proba so the pipeline-level CV precompute
    (which calls fold_model.predict_proba) can exercise its success path."""

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n_classes = len(getattr(self, "classes_", [0, 1]))
        return np.full((len(X), n_classes), 1.0 / n_classes)


def test_run_pipeline_precomputes_cv_folds_once_per_cv_model(setup, caplog):
    """Pipeline precomputes shared fold predictions once per (dataset,
    model) when any CV strategy is registered, and seeds every CV
    strategy's ``_fold_results`` before the inner fit loop."""
    seen_fold_results: list = []

    class TrackingCVStrategy(CVTunesStrategy):
        def __init__(self, model):
            super().__init__(model)
            self._fold_results = None

        def _fit(self, X, y, opt_obj):
            seen_fold_results.append(self._fold_results)
            return self

    caplog.set_level("INFO")
    _results = pipeline.run_experiment_pipeline(
        dataset_name="ds",
        X=setup["X"],
        y=np.array([0, 1] * 50),
        random_seed=42,
        models={"m": MockProbaModel()},
        strategies={
            "CV1": lambda m: TrackingCVStrategy(m),
            "CV2": lambda m: TrackingCVStrategy(m),
        },
        objectives=setup["objectives"],
    )

    precompute_msgs = [
        r.message
        for r in caplog.records
        if "CV fold precompute" in r.message and "failed" not in r.message
    ]
    # One precompute per (dataset, model); both CV strategies consume it.
    assert len(precompute_msgs) == 1, (
        f"expected 1 precompute log, got {len(precompute_msgs)}:"
        f" {precompute_msgs}"
    )
    assert len(seen_fold_results) == 2
    for observed in seen_fold_results:
        assert observed is not None
        assert len(observed) == pipeline.CV_PRECOMPUTE_FOLDS


def test_run_pipeline_skips_cv_precompute_when_no_cv_strategy(setup, caplog):
    """When no registered strategy declares ``self_tunes_with_cv=True``,
    the pipeline must not run the expensive CV precompute."""
    caplog.set_level("INFO")
    _results = pipeline.run_experiment_pipeline(
        dataset_name="ds",
        X=setup["X"],
        y=np.array([0, 1] * 50),
        random_seed=42,
        models={"m": MockProbaModel()},
        strategies={"needs_val": lambda m: NeedsValStrategy(m)},
        objectives=setup["objectives"],
    )

    precompute_msgs = [
        r.message for r in caplog.records if "CV fold precompute" in r.message
    ]
    assert precompute_msgs == [], (
        f"expected no precompute log for non-CV runs, got {precompute_msgs}"
    )


def test_run_pipeline_param_extractor_stored_in_result(setup):
    """param_extractor output should be stored as learned_params."""

    class ParamStrategy(base.DecisionStrategy):
        needs_validation_set = False

        def _fit(self, X, y, opt_obj):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    results = pipeline.run_experiment_pipeline(
        dataset_name="ds",
        X=setup["X"],
        y=np.array([0, 1] * 50),
        random_seed=42,
        models={"m": MockModel()},
        strategies={"s": lambda m: ParamStrategy(m)},
        objectives=setup["objectives"],
        param_extractor=lambda strat: 0.42,
    )
    assert len(results) == 1
    assert results[0].learned_params == pytest.approx(0.42)


# create_run_dir / resumability


def test_create_run_dir_creates_new_directory(tmp_path):
    with mock.patch.object(pipeline, "RESULTS_DIR", tmp_path):
        run_dir = pipeline.create_run_dir("test")

    assert run_dir.is_dir()
    assert run_dir.name.startswith("run_test_")


def test_create_run_dir_resume_reuses_existing(tmp_path):
    existing = tmp_path / "run_test_20260101_000000"
    existing.mkdir()

    run_dir = pipeline.create_run_dir("test", resume=str(existing))

    assert run_dir == existing


def test_create_run_dir_resume_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.create_run_dir("test", resume=str(tmp_path / "nonexistent"))


# _save_task_results / _task_result_path


def test_save_task_results_writes_parquet(tmp_path):
    results = [_make_result(), _make_result(model="m2")]
    path = tmp_path / "ds_seed0.parquet"

    pipeline._save_task_results(results, path)

    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "objective_value" in df.columns


def test_save_task_results_writes_strategy_fit_seconds_column(tmp_path):
    """The strategy_fit_seconds field must be persisted in the parquet so
    per-(dataset, model, strategy, objective) timing is cross-run analysable
    without having to re-parse the run.log."""
    cfg = config.ExperimentConfig(
        dataset_name="ds",
        model_name="m",
        optimization_objective_name="F1",
        strategy_name="s",
        random_seed=0,
        test_size=0.2,
        val_size=0.1,
        noise_level=0.0,
    )
    result_with = config.ExperimentResult(
        config=cfg,
        learned_params=0.5,
        objective_value=0.9,
        strategy_fit_seconds=1.25,
    )
    result_without = config.ExperimentResult(
        config=cfg,
        learned_params=0.5,
        objective_value=0.9,
    )
    path = tmp_path / "ds_seed0.parquet"

    pipeline._save_task_results([result_with, result_without], path)

    df = pd.read_parquet(path)
    assert "strategy_fit_seconds" in df.columns
    assert df["strategy_fit_seconds"].iloc[0] == 1.25
    assert pd.isna(df["strategy_fit_seconds"].iloc[1])


def test_save_task_results_empty_list_does_not_write(tmp_path):
    path = tmp_path / "empty.parquet"
    pipeline._save_task_results([], path)
    assert not path.exists()


def test_save_task_results_custom_param_column(tmp_path):
    results = [_make_result()]
    path = tmp_path / "ds_seed0.parquet"

    pipeline._save_task_results(results, path, "threshold")

    df = pd.read_parquet(path)
    assert "threshold" in df.columns


def test_task_result_path():
    run_dir = pipeline.Path("/some/run_dir")
    path = pipeline._task_result_path(run_dir, "sonar", 2)
    assert path == run_dir / "sonar_seed2.parquet"


# merge_run_results


def test_merge_run_results_combines_parquets(tmp_path):
    run_dir = tmp_path / "run_pfx_20260315_120000"
    run_dir.mkdir()

    # Write two per-task parquets
    for ds, val in [("ds1", 0.9), ("ds2", 0.7)]:
        pipeline._save_task_results(
            [_make_result(dataset=ds, value=val)],
            run_dir / f"{ds}_seed0.parquet",
        )

    out = pipeline.merge_run_results(run_dir, "pfx")

    assert out.exists()
    assert out.parent == run_dir
    assert out.name == "pfx_20260315_120000.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["dataset_name"]) == {"ds1", "ds2"}


def test_merge_run_results_empty_dir_raises(tmp_path):
    run_dir = tmp_path / "run_pfx_20260315_120000"
    run_dir.mkdir()

    with mock.patch.object(pipeline, "RESULTS_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            pipeline.merge_run_results(run_dir, "pfx")


def test_merge_run_results_warns_on_schema_mismatch_but_still_merges(
    tmp_path, caplog
):
    """A per-task parquet whose columns differ (e.g. a schema drift
    between a CPU and a resumed run) must not be NaN-merged silently:
    merge_run_results logs a loud SCHEMA_MISMATCH and still concatenates
    so no rows are lost.
    """
    run_dir = tmp_path / "run_pfx_20260315_120000"
    run_dir.mkdir()

    pipeline._save_task_results(
        [_make_result(dataset="ds1", value=0.9)],
        run_dir / "ds1_seed0.parquet",
    )
    # Divergent schema: missing every ExperimentResult column, plus an
    # unexpected extra one.
    pd.DataFrame({"objective_value": [0.5], "rogue_col": [1]}).to_parquet(
        run_dir / "ds2_seed0.parquet", index=False
    )

    with caplog.at_level(
        logging.WARNING, logger="experiments.common.pipeline"
    ):
        out = pipeline.merge_run_results(run_dir, "pfx")

    mismatch = [
        r for r in caplog.records if "SCHEMA_MISMATCH" in r.getMessage()
    ]
    assert mismatch, "expected a SCHEMA_MISMATCH warning"
    df = pd.read_parquet(out)
    # No rows dropped: both per-task files contributed.
    assert len(df) == 2
    assert "rogue_col" in df.columns
    assert df["rogue_col"].isna().any()  # gap filled with NaN, not dropped


# _run_single_seed — incremental saves & skip logic


def test_run_single_seed_saves_to_run_dir(tmp_path, setup):
    with mock.patch(
        "experiments.common.pipeline.data_loader.load_dataset",
        autospec=True,
        return_value=(setup["X"], np.array([0, 1] * 50)),
    ):
        pipeline._run_single_seed(
            dataset_name="ds",
            random_seed=0,
            models=setup["models"],
            strategies=setup["strategies"],
            objectives=setup["objectives"],
            param_extractor=None,
            run_dir=tmp_path,
        )

    assert (tmp_path / "ds_seed0.parquet").exists()


def test_run_single_seed_skips_completed(tmp_path, setup):
    # Pre-create a *valid* per-task parquet: a completed task is only
    # skipped if its file passes the footer-integrity probe.
    result_path = tmp_path / "ds_seed0.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(result_path)

    with mock.patch(
        "experiments.common.pipeline.data_loader.load_dataset",
        autospec=True,
    ) as mock_load:
        results = pipeline._run_single_seed(
            dataset_name="ds",
            random_seed=0,
            models=setup["models"],
            strategies=setup["strategies"],
            objectives=setup["objectives"],
            param_extractor=None,
            run_dir=tmp_path,
        )

    # Should not have loaded the dataset at all
    mock_load.assert_not_called()
    assert results == []


def test_run_single_seed_reruns_corrupt_parquet(tmp_path, setup):
    # A file that exists but is not a readable parquet (e.g. a write
    # truncated by an interrupted/preempted run) must NOT be treated as
    # a completed task -- the task is re-run instead of silently skipped.
    result_path = tmp_path / "ds_seed0.parquet"
    result_path.write_bytes(b"not a parquet")

    with mock.patch(
        "experiments.common.pipeline.data_loader.load_dataset",
        autospec=True,
        side_effect=RuntimeError("stop after load attempt"),
    ) as mock_load:
        results = pipeline._run_single_seed(
            dataset_name="ds",
            random_seed=0,
            models=setup["models"],
            strategies=setup["strategies"],
            objectives=setup["objectives"],
            param_extractor=None,
            run_dir=tmp_path,
        )

    # The corrupt file did not short-circuit the skip: the task was
    # re-run, which means load_dataset was reached.
    mock_load.assert_called_once()
    assert results == []


# setup_file_logging


def test_setup_file_logging_creates_log_file(tmp_path):
    pipeline.setup_file_logging(tmp_path)

    log_path = tmp_path / "run.log"
    assert log_path.exists()

    # Clean up: remove the handler we just added
    root = logging.getLogger()
    for h in root.handlers[:]:
        if (
            isinstance(h, logging.FileHandler)
            and str(tmp_path) in h.baseFilename
        ):
            root.removeHandler(h)
            h.close()


def test_setup_file_logging_sets_root_logger_to_info(tmp_path):
    """Loky workers start at Python's default WARNING level; without
    setLevel(INFO) here, worker INFO records get dropped before reaching
    any handler and the run.log freezes after the header lines."""
    root = logging.getLogger()
    original_level = root.level
    root.setLevel(logging.WARNING)

    try:
        pipeline.setup_file_logging(tmp_path)
        assert root.level == logging.INFO
    finally:
        root.setLevel(original_level)
        log_path = (tmp_path / "run.log").resolve()
        for h in root.handlers[:]:
            if (
                isinstance(h, logging.FileHandler)
                and Path(h.baseFilename).resolve() == log_path
            ):
                root.removeHandler(h)
                h.close()


def test_setup_file_logging_is_idempotent(tmp_path):
    """Joblib workers call setup_file_logging at task entry; repeated calls
    with the same run_dir must not stack duplicate FileHandlers on the root
    logger."""
    root = logging.getLogger()
    log_path = (tmp_path / "run.log").resolve()

    def count_matching_handlers() -> int:
        return sum(
            1
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_path
        )

    try:
        pipeline.setup_file_logging(tmp_path)
        first_count = count_matching_handlers()
        pipeline.setup_file_logging(tmp_path)
        pipeline.setup_file_logging(tmp_path)

        assert count_matching_handlers() == first_count == 1
    finally:
        for h in root.handlers[:]:
            if (
                isinstance(h, logging.FileHandler)
                and Path(h.baseFilename).resolve() == log_path
            ):
                root.removeHandler(h)
                h.close()


# add_common_cli_args


def test_add_common_cli_args_defaults():
    import argparse

    parser = argparse.ArgumentParser()
    pipeline.add_common_cli_args(parser)
    args = parser.parse_args([])

    assert args.n_jobs == -1
    assert args.n_seeds == 3
    assert args.master_seed == 42
    assert args.resume is None
    assert args.models is None


def test_add_common_cli_args_override():
    import argparse

    parser = argparse.ArgumentParser()
    pipeline.add_common_cli_args(parser)
    args = parser.parse_args(
        [
            "--n-jobs",
            "4",
            "--n-seeds",
            "10",
            "--master-seed",
            "7",
            "--resume",
            "/some/dir",
        ]
    )

    assert args.n_jobs == 4
    assert args.n_seeds == 10
    assert args.master_seed == 7
    assert args.resume == "/some/dir"


def test_add_common_cli_args_models():
    import argparse

    parser = argparse.ArgumentParser()
    pipeline.add_common_cli_args(parser)
    args = parser.parse_args(["--models", "A,B"])

    assert args.models == "A,B"


def test_add_common_cli_args_device_and_models_mutually_exclusive():
    import argparse

    parser = argparse.ArgumentParser()
    pipeline.add_common_cli_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "gpu", "--models", "A"])


# filter_models


def test_filter_models_none_returns_all():
    models = {"A": MockModel(), "B": MockModel()}
    assert pipeline.filter_models(models, None) is models


def test_filter_models_selects_subset():
    models = {"A": MockModel(), "B": MockModel(), "C": MockModel()}
    filtered = pipeline.filter_models(models, "A,C")
    assert list(filtered.keys()) == ["A", "C"]


def test_filter_models_preserves_order():
    models = {"A": MockModel(), "B": MockModel(), "C": MockModel()}
    filtered = pipeline.filter_models(models, "C,A")
    assert list(filtered.keys()) == ["C", "A"]


def test_filter_models_strips_whitespace():
    models = {"A": MockModel(), "B": MockModel()}
    filtered = pipeline.filter_models(models, " A , B ")
    assert list(filtered.keys()) == ["A", "B"]


def test_filter_models_unknown_key_exits():
    models = {"A": MockModel(), "B": MockModel()}
    with pytest.raises(SystemExit, match="Unknown model"):
        pipeline.filter_models(models, "A,Z")


def test_filter_models_with_parens_in_key():
    models = {"LGBMClassifier()": MockModel(), "XGBoost": MockModel()}
    filtered = pipeline.filter_models(models, "LGBMClassifier()")
    assert list(filtered.keys()) == ["LGBMClassifier()"]


# select_models


def test_select_models_no_flags_returns_all():
    all_m = {"A": MockModel(), "B": MockModel(), "C": MockModel()}
    cpu = {"A": all_m["A"], "B": all_m["B"]}
    gpu = {"C": all_m["C"]}
    result = pipeline.select_models(all_m, cpu, gpu, None, None)
    assert list(result.keys()) == ["A", "B", "C"]


def test_select_models_device_cpu():
    all_m = {"A": MockModel(), "B": MockModel()}
    cpu = {"A": all_m["A"]}
    gpu = {"B": all_m["B"]}
    result = pipeline.select_models(all_m, cpu, gpu, "cpu", None)
    assert list(result.keys()) == ["A"]


def test_select_models_device_gpu():
    all_m = {"A": MockModel(), "B": MockModel()}
    cpu = {"A": all_m["A"]}
    gpu = {"B": all_m["B"]}
    result = pipeline.select_models(all_m, cpu, gpu, "gpu", None)
    assert list(result.keys()) == ["B"]


def test_select_models_models_filter_without_device():
    all_m = {"A": MockModel(), "B": MockModel(), "C": MockModel()}
    cpu = {"A": all_m["A"]}
    gpu = {"B": all_m["B"], "C": all_m["C"]}
    result = pipeline.select_models(all_m, cpu, gpu, None, "B,C")
    assert list(result.keys()) == ["B", "C"]
