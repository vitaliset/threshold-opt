"""Lightweight e2e smoke tests for the experiment runners.

These tests exercise the real strategies, models, and data-loading pipeline
with minimal configuration (1 seed, 1 small dataset, few strategies/objectives)
so they finish quickly while still catching regressions in the threshold_opt
library.

Run with:
    uv run pytest experiments/smoke_test.py --integration -v
"""

import math

import numpy as np
import pytest
from sklearn import metrics as sklearn_metrics
from sklearn.linear_model import LogisticRegression

from experiments.common import data_loader, pipeline
from threshold_opt import binary_strategies, metrics, multiclass_strategies

SEED = 42
LOGREG = LogisticRegression(random_state=SEED)


# -- binary -------------------------------------------------------------------

BINARY_STRATEGIES = {
    "Argmax": binary_strategies.ArgmaxThreshold,
    "IsotonicArgmax": binary_strategies.IsotonicArgmax,
    "PlattScalingArgmax": binary_strategies.PlattScalingArgmax,
    "Analytical": binary_strategies.AnalyticalThreshold,
    "Empirical": binary_strategies.EmpiricalThreshold,
    "EmpiricalCV": lambda model: binary_strategies.EmpiricalThreshold(
        model, use_cv=True
    ),
}

BINARY_OBJECTIVES = {
    "Symmetric Cost": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "F1": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.f1_score(
            yt, yp, zero_division=0
        ),
        higher_is_better=True,
        vectorized_key="f1",
    ),
}


@pytest.mark.integration
def test_binary_runner_smoke():
    X, y = data_loader.load_dataset("sonar")
    results = pipeline.run_experiment_pipeline(
        dataset_name="sonar",
        X=X,
        y=y,
        random_seed=SEED,
        models={"LogisticRegression()": LOGREG},
        strategies=BINARY_STRATEGIES,
        objectives=BINARY_OBJECTIVES,
    )

    assert len(results) > 0, "Binary smoke test produced no results"
    for r in results:
        assert r.objective_value is not None, (
            f"{r.config.strategy_name}/{r.config.optimization_objective_name}"
            " returned None"
        )
        assert not math.isnan(r.objective_value), (
            f"{r.config.strategy_name}/{r.config.optimization_objective_name}"
            " returned NaN"
        )


# -- multiclass ---------------------------------------------------------------

MULTICLASS_STRATEGIES = {
    "Argmax": multiclass_strategies.ArgmaxMulticlass,
    "IsotonicArgmax": multiclass_strategies.IsotonicArgmaxMulticlass,
    "Analytical": multiclass_strategies.AnalyticalMulticlass,
    "IsotonicAnalytical": (multiclass_strategies.IsotonicAnalyticalMulticlass),
    "ThresholdRatio": multiclass_strategies.ThresholdRatioMulticlass,
    "SequentialSelection": (
        multiclass_strategies.SequentialSelectionMulticlass
    ),
}

MULTICLASS_OBJECTIVES = {
    "Accuracy": metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score,
        higher_is_better=True,
    ),
    "Symmetric Cost": lambda n: np.ones((n, n)) - np.eye(n),
}


@pytest.mark.integration
def test_multiclass_runner_smoke():
    X, y = data_loader.load_dataset("iris")
    results = pipeline.run_experiment_pipeline(
        dataset_name="iris",
        X=X,
        y=y,
        random_seed=SEED,
        models={"LogisticRegression()": LOGREG},
        strategies=MULTICLASS_STRATEGIES,
        objectives=MULTICLASS_OBJECTIVES,
    )

    assert len(results) > 0, "Multiclass smoke test produced no results"
    for r in results:
        assert r.objective_value is not None, (
            f"{r.config.strategy_name}/{r.config.optimization_objective_name}"
            " returned None"
        )
        assert not math.isnan(r.objective_value), (
            f"{r.config.strategy_name}/{r.config.optimization_objective_name}"
            " returned NaN"
        )
