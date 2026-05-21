import numpy as np
import pytest
from sklearn import base
from sklearn import metrics as sklearn_metrics

from threshold_opt import cv, metrics, multiclass_strategies, optimization


class MockMulticlassModel(base.ClassifierMixin, base.BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if hasattr(self, "classes_"):
            return np.array(X)[:, self.classes_]
        return np.array(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class MockOptimizer(optimization.ParameterOptimizer):
    def __init__(self, best_params):
        self.best_params = np.array(best_params)

    def optimize(
        self,
        objective_fn,
        n_params,
        current_best_params=None,
        current_best_score=-np.inf,
    ):
        score = objective_fn(self.best_params)
        return self.best_params, score


@pytest.fixture
def base_setup():
    model = MockMulticlassModel()
    X_probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([0, 1])
    return model, X_probs, y


def test_argmax_multiclass(base_setup):
    model, X_probs, y = base_setup
    strategy = multiclass_strategies.ArgmaxMulticlass(model)
    strategy.fit(X_probs, y, optimization_objective=[])
    preds = strategy.predict(X_probs)
    np.testing.assert_array_equal([0, 1], preds)


def test_validate_fit_input_accepts_empty_fit_set(base_setup):
    # The pipeline hands strategies that neither need validation nor
    # self-tune with CV (Argmax, Analytical) empty arrays as the
    # fit set. The base validator must not probe the model via
    # predict_proba in that case.
    model, _, _ = base_setup
    metric_objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    empty = np.array([])

    multiclass_strategies.ArgmaxMulticlass(model).fit(
        empty, empty, metric_objective
    )
    multiclass_strategies.AnalyticalMulticlass(model).fit(
        empty, empty, np.eye(3)
    )


def test_calibrated_argmax_multiclass(base_setup):
    model, _X_probs, _y = base_setup
    X_train = np.tile(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), (5, 1)
    )
    y_train = np.tile(np.array([0, 1, 2]), 5)
    model.fit(X_train, y_train)
    strategy = multiclass_strategies.IsotonicArgmaxMulticlass(model)
    strategy.fit(X_train, y_train, optimization_objective=[])
    preds = strategy.predict(X_train)
    np.testing.assert_array_equal(y_train, preds)
    assert strategy.calibrated_model is not None


def test_safe_n_splits_alias_delegates_to_cv_module():
    """multiclass_strategies._safe_n_splits is kept as a thin alias for
    cv.safe_n_splits (unit-tested in cv_test.py); verify the delegation
    so callers importing the old name keep the corrected behaviour.
    """
    assert (
        multiclass_strategies._safe_n_splits(
            np.repeat(np.arange(3), 10), max_splits=5
        )
        == 5
    )
    with pytest.raises(cv.InsufficientCVData):
        multiclass_strategies._safe_n_splits(np.array([0, 1, 2]))


def test_isotonic_calibration_raises_typed_error_on_singleton_class():
    """A class with a single member must raise InsufficientCVData (a
    ValueError subclass) rather than crash deep inside sklearn -- this
    is the signal the experiments pipeline turns into a countable
    CV_SKIP instead of a silently dropped result cell.
    """
    model = MockMulticlassModel()
    X_train = np.tile(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), (5, 1)
    )
    y_train = np.tile(np.array([0, 1, 2]), 5)
    model.fit(X_train, y_train)

    # Class 2 has only one member in this validation split.
    X_small = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    y_small = np.array([0, 0, 1, 1, 2])

    strategy = multiclass_strategies.IsotonicArgmaxMulticlass(model)
    with pytest.raises(cv.InsufficientCVData):
        strategy.fit(X_small, y_small, optimization_objective=[])


def test_calibrated_argmax_survives_small_validation_split():
    """Regression: a validation split where some class has fewer members
    than the default cv=5 used to crash with IndexError inside
    _enforce_prediction_order. _safe_n_splits + explicit StratifiedKFold
    makes the fit succeed instead.
    """
    model = MockMulticlassModel()
    X_train = np.tile(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), (5, 1)
    )
    y_train = np.tile(np.array([0, 1, 2]), 5)
    model.fit(X_train, y_train)

    X_small = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.2, 0.7],
        ]
    )
    y_small = np.array([0, 0, 1, 1, 2, 2])

    strategy = multiclass_strategies.IsotonicArgmaxMulticlass(model)
    strategy.fit(X_small, y_small, optimization_objective=[])
    assert strategy.calibrated_model is not None


@pytest.mark.parametrize(
    "cost_matrix,expected_preds",
    [
        (np.ones((3, 3)) - np.eye(3), np.array([0, 1])),
        (np.array([[0, 1, 1], [100, 0, 1], [100, 1, 0]]), np.array([1, 1])),
    ],
)
def test_analytical_multiclass(base_setup, cost_matrix, expected_preds):
    model, X_probs, y = base_setup
    strategy = multiclass_strategies.AnalyticalMulticlass(model)
    strategy.fit(X_probs, y, cost_matrix)
    preds = strategy.predict(X_probs)
    np.testing.assert_array_equal(expected_preds, preds)


def test_analytical_multiclass_raises_if_not_fit(base_setup):
    model, X_probs, _y = base_setup
    strategy = multiclass_strategies.AnalyticalMulticlass(model)
    with pytest.raises(ValueError, match="Strategy not fit"):
        strategy.predict(X_probs)


def test_analytical_multiclass_raises_on_metric_objective(base_setup):
    model, X_probs, y = base_setup
    strategy = multiclass_strategies.AnalyticalMulticlass(model)
    metric_obj = metrics.MetricObjective(lambda y, p: 0, True)
    with pytest.raises(ValueError, match="requires a cost matrix"):
        strategy.fit(X_probs, y, metric_obj)


@pytest.mark.parametrize(
    "y_invalid,error_regex,cost_matrix",
    [
        (np.array([0, -1]), "Labels must be non-negative", np.eye(3)),
        (np.array([0.5, 1.0]), "integer-valued", np.eye(3)),
        (np.array([0, 3]), "Labels must be in", np.eye(3)),
    ],
)
def test_fit_raises_on_invalid_labels(
    base_setup, y_invalid, error_regex, cost_matrix
):
    model, X_probs, _y = base_setup
    strategy = multiclass_strategies.AnalyticalMulticlass(model)
    with pytest.raises(ValueError, match=error_regex):
        strategy.fit(X_probs, y_invalid, cost_matrix)


class TestIntegerValuedFloatLabels:
    """Validator/coercer contract for float-dtype y.

    astype(int) truncates (0.5 -> 0, 1.9 -> 1); array_equal then compares
    element-wise, so only exactly-integer-valued floats pass.
    """

    @pytest.fixture
    def setup(self, base_setup):
        model, X_probs, _ = base_setup
        strategy = multiclass_strategies.AnalyticalMulticlass(model)
        return model, X_probs, strategy

    def test_accepts_whole_number_floats(self, setup):
        _, X_probs, strategy = setup
        strategy.fit(X_probs, np.array([0.0, 1.0]), np.eye(3))
        preds = strategy.predict(X_probs)
        assert preds.dtype.kind in ("i", "u")

    @pytest.mark.parametrize(
        "y",
        [
            np.array([0.5, 1.0]),
            np.array([0.9, 1.0]),
            np.array([1.5, 2.5]),
            np.array([-0.5, 1.0]),
        ],
    )
    def test_rejects_non_whole_floats(self, setup, y):
        _, X_probs, strategy = setup
        with pytest.raises(ValueError, match="integer-valued"):
            strategy.fit(X_probs, y, np.eye(3))

    def test_rejects_rounding_that_would_collide(self, setup):
        """0.5 must not be silently truncated to 0. A caller that relied on
        rounding would get a 1, not a 0; a caller that relied on truncation
        would get a 0. Both interpretations are footguns, so we reject.
        """
        _, X_probs, strategy = setup
        y_truncates_to_same = np.array([0.0, 0.9])  # both truncate to 0
        with pytest.raises(ValueError, match="integer-valued"):
            strategy.fit(X_probs, y_truncates_to_same, np.eye(3))


class TestCoerceIntegerLabels:
    """base._coerce_integer_labels is the last-mile cast before _fit."""

    def test_passes_through_integer_arrays(self):
        from threshold_opt import base as base_mod

        y = np.array([0, 1, 2], dtype=np.int64)
        out = base_mod._coerce_integer_labels(y)
        assert out.dtype.kind in ("i", "u")
        np.testing.assert_array_equal(out, y)

    def test_casts_integer_valued_float(self):
        from threshold_opt import base as base_mod

        out = base_mod._coerce_integer_labels(np.array([0.0, 1.0, 2.0]))
        assert out.dtype.kind in ("i", "u")
        np.testing.assert_array_equal(out, np.array([0, 1, 2]))

    def test_coercer_alone_truncates_non_whole_floats(self):
        """Documents that the coercer alone would truncate 0.5 -> 0. The
        strategy validator is the gate that prevents this from ever being
        reached with non-integer-valued y; if someone bypasses validation
        they get truncation, not rounding — which is why the validator
        matters.
        """
        from threshold_opt import base as base_mod

        out = base_mod._coerce_integer_labels(np.array([0.5, 1.9]))
        np.testing.assert_array_equal(out, np.array([0, 1]))


def test_threshold_ratio_multiclass():
    probs = np.array([[0.6, 0.4]])
    y = np.array([1])

    class SingleSampleModel:
        def predict_proba(self, X):
            return probs

    model = SingleSampleModel()
    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    optimizer = optimization.RandomSearchOptimizer(n_iter=50)
    strategy = multiclass_strategies.ThresholdRatioMulticlass(
        model, optimizer=optimizer
    )
    strategy.fit(probs, y, objective)

    preds = strategy.predict(probs)
    assert preds[0] == 1
    assert strategy.thresholds is not None
    # To flip argmax from class 0 to class 1 on probs=(0.6, 0.4) via
    # argmax(p / tau), the optimizer must find tau_1/tau_0 < 0.4/0.6.
    assert strategy.thresholds[1] / strategy.thresholds[0] < 0.4 / 0.6


def test_voronoi_tuning_multiclass():
    probs = np.array([[0.6, 0.4]])
    y = np.array([1])

    class SingleSampleModel:
        def predict_proba(self, X):
            return probs

    model = SingleSampleModel()
    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    strategy = multiclass_strategies.VoronoiTuningMulticlass(
        model, n_iter=20, n_jobs=1, seed=42
    )
    strategy.fit(probs, y, objective)
    preds = strategy.predict(probs)
    assert preds[0] == 1


def test_sequential_selection_multiclass():
    probs = np.array([[0.4, 0.6], [0.9, 0.1]])
    y = np.array([0, 0])

    class CustomModel:
        def predict_proba(self, X):
            return probs

    model = CustomModel()
    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    mock_optimizer = MockOptimizer(best_params=[0.4, 0.7])
    strategy = multiclass_strategies.SequentialSelectionMulticlass(
        model, optimizer=mock_optimizer
    )
    strategy.fit(probs, y, objective)

    preds = strategy.predict(probs)
    np.testing.assert_array_equal([0, 0], preds)
    assert strategy.order == (1, 0)
    np.testing.assert_array_equal(strategy.thresholds, [0.4, 0.7])


@pytest.mark.parametrize(
    "objective,probs,y,mock_best_params,expected_preds,expected_biases",
    [
        pytest.param(
            metrics.MetricObjective(
                metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
            ),
            np.array([[0.4, 0.35, 0.25]]),
            np.array([1]),
            [0.1, -0.1],
            [1],
            [0.0, 0.1, -0.1],
            id="metric",
        ),
        pytest.param(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.6, 0.4]]),
            np.array([1]),
            [0.3],
            [1],
            [0.0, 0.3],
            id="cost",
        ),
    ],
)
def test_multidimensional_threshold(
    objective, probs, y, mock_best_params, expected_preds, expected_biases
):
    class SingleSampleModel:
        def predict_proba(self, X):
            return probs

    model = SingleSampleModel()
    mock_optimizer = MockOptimizer(best_params=mock_best_params)
    strategy = multiclass_strategies.MultidimensionalThresholdStrategy(
        model, optimizer=mock_optimizer
    )
    strategy.fit(probs, y, objective)
    preds = strategy.predict(probs)
    assert preds[0] == expected_preds[0]
    np.testing.assert_allclose(strategy.biases, expected_biases)


def test_multidimensional_threshold_cv():
    class SingleSampleModel(MockMulticlassModel):
        def predict_proba(self, X):
            return np.tile(np.array([[0.4, 0.35, 0.25]]), (X.shape[0], 1))

    model = SingleSampleModel()
    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    mock_optimizer = MockOptimizer(best_params=[0.1, -0.1])
    strategy = multiclass_strategies.MultidimensionalThresholdStrategy(
        model, optimizer=mock_optimizer, use_cv=True, cv_folds=2
    )
    # 4 samples across 2 classes so real 2-fold StratifiedKFold works.
    X_large = np.tile(np.array([[0.4, 0.35, 0.25]]), (4, 1))
    y_large = np.array([0, 0, 1, 1])

    strategy.fit(X_large, y_large, objective)

    preds = strategy.predict(X_large)
    assert preds[0] == 1
    np.testing.assert_allclose(strategy.biases, [0.0, 0.1, -0.1])


def test_risk_partitioning_consistency():
    cost_matrix = np.array([[0, 10], [1, 0]])
    probs = np.array([[0.5, 0.5]])
    y = np.array([0])

    class SingleSampleModel:
        def predict_proba(self, X):
            return probs

    model = SingleSampleModel()

    ref_strategy = multiclass_strategies.AnalyticalMulticlass(model)
    ref_strategy.fit(probs, y, cost_matrix)
    ref_preds = ref_strategy.predict(probs)
    assert ref_preds[0] == 0

    mock_optimizer = MockOptimizer(best_params=[0.0])
    strategy = multiclass_strategies.RiskPartitioningStrategy(
        model, optimizer=mock_optimizer
    )
    strategy.fit(probs, y, cost_matrix)
    preds = strategy.predict(probs)
    assert preds[0] == 0


def test_calibrated_analytical_multiclass():
    cost_matrix = np.ones((3, 3)) - np.eye(3)
    model = MockMulticlassModel()

    X_train = np.tile(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        (5, 1),
    )
    y_train = np.tile(np.array([0, 1, 2]), 5)
    model.fit(X_train, y_train)

    strategy = multiclass_strategies.IsotonicAnalyticalMulticlass(model)
    strategy.fit(X_train, y_train, cost_matrix)
    preds = strategy.predict(X_train)

    np.testing.assert_array_equal(y_train, preds)
    assert strategy.calibrated_model is not None


def test_calibrated_analytical_pads_when_calibrator_sees_fewer_classes():
    """If the calibrator was fit on a subset of classes (e.g. a rare class
    absent from a stratified val split), predict_proba returns fewer
    columns than the cost matrix — the strategy must pad them to match
    instead of letting ``probs @ cost_matrix`` raise a shape mismatch."""

    # 4x4 cost matrix, uniform off-diagonal
    cost_matrix = np.ones((4, 4)) - np.eye(4)

    class MockPartialCalibrator:
        # Calibrator only saw classes [0, 1, 2]; class 3 is absent.
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, X):
            return np.tile(np.array([[0.5, 0.3, 0.2]]), (len(X), 1))

    strategy = multiclass_strategies.IsotonicAnalyticalMulticlass(
        MockMulticlassModel()
    )
    strategy.calibrated_model = MockPartialCalibrator()
    strategy.cost_matrix = cost_matrix

    preds = strategy.predict(np.zeros((3, 5)))
    # Expected risks given padded probs = [0.5, 0.3, 0.2, 0.0]:
    #   risks[j=0] = 0.0 + 0.3 + 0.2 + 0.0 = 0.5   ← argmin
    #   risks[j=1] = 0.5 + 0.0 + 0.2 + 0.0 = 0.7
    #   risks[j=2] = 0.5 + 0.3 + 0.0 + 0.0 = 0.8
    #   risks[j=3] = 0.5 + 0.3 + 0.2 + 0.0 = 1.0
    assert preds.shape == (3,)
    np.testing.assert_array_equal(preds, np.array([0, 0, 0]))


# predict-before-fit guards


def test_threshold_ratio_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.ThresholdRatioMulticlass(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


def test_voronoi_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.VoronoiTuningMulticlass(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


def test_sequential_selection_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.SequentialSelectionMulticlass(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


def test_multidimensional_threshold_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.MultidimensionalThresholdStrategy(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


def test_risk_partitioning_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.RiskPartitioningStrategy(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


# RiskPartitioningStrategy - validation and solver paths


def test_risk_partitioning_metric_objective_raises(base_setup):
    model, X_probs, y = base_setup
    strategy = multiclass_strategies.RiskPartitioningStrategy(model)
    metric_obj = metrics.MetricObjective(lambda yt, yp: 0.0, True)
    with pytest.raises(ValueError, match="requires a cost matrix"):
        strategy.fit(X_probs, y, metric_obj)


def test_risk_partitioning_greedy_solver_with_cv_raises(base_setup):
    model, _X_probs, _y = base_setup
    with pytest.raises(ValueError, match="use_cv=True"):
        multiclass_strategies.RiskPartitioningStrategy(
            model, solver="greedy", use_cv=True
        )


def test_risk_partitioning_unknown_solver_raises(base_setup):
    model, _X_probs, _y = base_setup
    with pytest.raises(ValueError, match="Unknown solver"):
        multiclass_strategies.RiskPartitioningStrategy(
            model, solver="bad_solver"
        )


def test_risk_partitioning_greedy_solver_correct_predictions():
    """Greedy solver predicts class 1 when it's clearly cheaper."""
    cost_matrix = np.array([[0, 10], [1, 0]])
    # prob=[0.05, 0.95]: P(1)>10/11 so class 1 preferred despite high FP cost
    probs = np.array([[0.05, 0.95]])
    y = np.array([1])

    class FixedModel:
        def predict_proba(self, X):
            return probs

    model = FixedModel()
    strategy = multiclass_strategies.RiskPartitioningStrategy(
        model, solver="greedy"
    )
    strategy.fit(probs, y, cost_matrix)
    preds = strategy.predict(probs)
    assert preds[0] == 1


def test_risk_partitioning_random_search_solver(base_setup):
    model, _, _ = base_setup
    cost_matrix = np.ones((3, 3)) - np.eye(3)
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    y = np.array([0, 1, 2])

    mock_optimizer = MockOptimizer(best_params=[0.0, 0.0])
    strategy = multiclass_strategies.RiskPartitioningStrategy(
        model, optimizer=mock_optimizer, solver="random_search"
    )
    strategy.fit(probs, y, cost_matrix)
    preds = strategy.predict(probs)
    np.testing.assert_array_equal([0, 1, 2], preds)
    assert strategy.biases is not None


# LinearScalingMulticlass


def test_linear_scaling_flips_prediction():
    """beta down-weighting truth class 1 flips the analytical choice."""
    cost_matrix = np.array([[0.0, 1.0], [4.0, 0.0]])
    probs = np.array([[0.7, 0.3]])
    y = np.array([0])

    class SingleSampleModel:
        def predict_proba(self, X):
            return probs

    model = SingleSampleModel()

    ref = multiclass_strategies.AnalyticalMulticlass(model)
    ref.fit(probs, y, cost_matrix)
    assert ref.predict(probs)[0] == 1

    # beta_1/beta_0 = 0.4 < 7/12, so the boundary rotates past the sample.
    strategy = multiclass_strategies.LinearScalingMulticlass(
        model, optimizer=MockOptimizer(best_params=[1.0, 0.4])
    )
    strategy.fit(probs, y, cost_matrix)
    assert strategy.predict(probs)[0] == 0
    np.testing.assert_allclose(strategy.scaling_factors, [1.0, 0.4])


def test_linear_scaling_consistency_with_analytical():
    """beta = 1 reproduces the bare analytical rule (the warm start)."""
    cost_matrix = np.array([[0, 10], [1, 0]])
    probs = np.array([[0.5, 0.5]])
    y = np.array([0])

    class SingleSampleModel:
        def predict_proba(self, X):
            return probs

    model = SingleSampleModel()

    ref = multiclass_strategies.AnalyticalMulticlass(model)
    ref.fit(probs, y, cost_matrix)

    strategy = multiclass_strategies.LinearScalingMulticlass(
        model, optimizer=MockOptimizer(best_params=[1.0, 1.0])
    )
    strategy.fit(probs, y, cost_matrix)
    np.testing.assert_array_equal(strategy.predict(probs), ref.predict(probs))


def test_linear_scaling_scale_invariance():
    """argmin is invariant to scaling all beta by a positive constant."""
    cost_matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    y = np.array([0, 2])

    class FixedModel:
        def predict_proba(self, X):
            return probs

    model = FixedModel()

    s1 = multiclass_strategies.LinearScalingMulticlass(
        model, optimizer=MockOptimizer(best_params=[2.0, 1.0, 0.5])
    )
    s1.fit(probs, y, cost_matrix)
    s2 = multiclass_strategies.LinearScalingMulticlass(
        model, optimizer=MockOptimizer(best_params=[20.0, 10.0, 5.0])
    )
    s2.fit(probs, y, cost_matrix)
    np.testing.assert_array_equal(s1.predict(probs), s2.predict(probs))


def test_linear_scaling_metric_objective_raises(base_setup):
    model, X_probs, y = base_setup
    strategy = multiclass_strategies.LinearScalingMulticlass(model)
    metric_obj = metrics.MetricObjective(lambda yt, yp: 0.0, True)
    with pytest.raises(ValueError, match="requires a cost matrix"):
        strategy.fit(X_probs, y, metric_obj)


def test_linear_scaling_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.LinearScalingMulticlass(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


def test_linear_scaling_cv():
    class SimpleModel(MockMulticlassModel):
        def predict_proba(self, X):
            return np.tile(np.array([[0.3, 0.7]]), (len(X), 1))

    model = SimpleModel()
    X = np.tile(np.array([[0.3, 0.7]]), (6, 1))
    y = np.array([1, 1, 1, 0, 0, 0])
    model.fit(X, y)

    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    strategy = multiclass_strategies.LinearScalingMulticlass(
        model, use_cv=True, cv_folds=2
    )
    strategy.fit(X, y, cost_matrix)

    assert strategy.scaling_factors is not None
    assert len(strategy.scaling_factors) == 2


# ThresholdRatioMulticlass - CV mode


def test_threshold_ratio_cv():
    class SimpleModel(MockMulticlassModel):
        def predict_proba(self, X):
            return np.tile(np.array([[0.3, 0.7]]), (len(X), 1))

    model = SimpleModel()
    X = np.tile(np.array([[0.3, 0.7]]), (6, 1))
    y = np.array([1, 1, 1, 0, 0, 0])
    model.fit(X, y)

    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    strategy = multiclass_strategies.ThresholdRatioMulticlass(
        model, use_cv=True, cv_folds=2
    )
    strategy.fit(X, y, objective)

    assert strategy.thresholds is not None
    assert len(strategy.thresholds) == 2


def test_cv_strategy_caches_fold_results_across_objectives():
    """A single CV strategy instance fit with two different objectives
    must refit the K folds exactly once; the second fit reuses the
    cached ``_fold_results``."""
    fit_calls: list[int] = []

    class CountingModel(MockMulticlassModel):
        # Class-level tracker so sklearn.base.clone() (which makes a fresh
        # instance) still appends here.
        def fit(self, X, y):
            fit_calls.append(len(X))
            return super().fit(X, y)

    model = CountingModel()
    X = np.tile(np.array([[0.3, 0.7]]), (6, 1))
    y = np.array([1, 1, 1, 0, 0, 0])
    model.fit(X, y)
    calls_after_training = len(fit_calls)

    accuracy = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    f1_macro = metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.f1_score(
            yt, yp, average="macro", zero_division=0
        ),
        higher_is_better=True,
    )
    strategy = multiclass_strategies.ThresholdRatioMulticlass(
        model, use_cv=True, cv_folds=2
    )
    strategy.fit(X, y, accuracy)
    calls_after_first = len(fit_calls)
    strategy.fit(X, y, f1_macro)
    calls_after_second = len(fit_calls)

    # First fit trains 2 fold clones; second fit must reuse them.
    assert calls_after_first == calls_after_training + 2
    assert calls_after_second == calls_after_first
    assert strategy._fold_results is not None
    assert len(strategy._fold_results) == 2


def test_cv_strategy_accepts_preseeded_fold_results():
    """When the pipeline seeds ``_fold_results`` before the first fit,
    the strategy must not run its own K-fold refit loop."""
    from threshold_opt import base as base_mod

    class CountingModel(MockMulticlassModel):
        def __init__(self):
            super().__init__()
            self.fit_calls = 0

        def fit(self, X, y):
            self.fit_calls += 1
            return super().fit(X, y)

    model = CountingModel()
    X = np.tile(np.array([[0.3, 0.7]]), (6, 1))
    y = np.array([1, 1, 1, 0, 0, 0])
    model.fit(X, y)
    calls_after_training = model.fit_calls

    preseeded = [
        base_mod._CVFoldPredictions(
            y=np.array([0, 1, 1]),
            probs=np.array([[0.5, 0.5], [0.2, 0.8], [0.3, 0.7]]),
        ),
        base_mod._CVFoldPredictions(
            y=np.array([0, 0, 1]),
            probs=np.array([[0.6, 0.4], [0.7, 0.3], [0.1, 0.9]]),
        ),
    ]

    strategy = multiclass_strategies.ThresholdRatioMulticlass(
        model, use_cv=True, cv_folds=2
    )
    strategy._fold_results = preseeded
    strategy.fit(
        X,
        y,
        metrics.MetricObjective(
            metric_fn=sklearn_metrics.accuracy_score,
            higher_is_better=True,
        ),
    )

    # No K-fold refits happened; the preseed was used as-is.
    assert model.fit_calls == calls_after_training
    assert strategy._fold_results is preseeded


# SequentialSelectionMulticlass - cost matrix objective


def test_sequential_selection_with_cost_matrix():
    probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    y = np.array([0, 1])
    cost_matrix = np.array([[0, 1], [1, 0]])

    class FixedModel:
        def predict_proba(self, X):
            return probs

    model = FixedModel()
    mock_optimizer = MockOptimizer(best_params=[0.5, 0.5])
    strategy = multiclass_strategies.SequentialSelectionMulticlass(
        model, optimizer=mock_optimizer
    )
    strategy.fit(probs, y, cost_matrix)
    preds = strategy.predict(probs)
    np.testing.assert_array_equal([0, 1], preds)


# VoronoiCentroidOptimizer - non-perfect-square raises


def test_voronoi_centroid_optimizer_non_perfect_square_raises():
    optimizer = multiclass_strategies.VoronoiCentroidOptimizer(
        n_iter=5, seed=0
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="perfect square"):
        optimizer._sample_params(rng, n_params=5)


@pytest.mark.parametrize("n_dim", [2, 3, 4, 5, 8, 10, 20])
def test_voronoi_sampler_vertex_self_maps_to_own_class(n_dim):
    """Sequential-d² sampler preserves the pure-class → own-class
    invariant by construction: for every draw, the nearest centroid to
    vertex e_k is c_k for all k. Regression against the Hungarian sampler
    which only enforced bijection and let ~99% of vertices swap owners at
    K=26.
    """
    optimizer = multiclass_strategies.VoronoiCentroidOptimizer(
        n_iter=1, seed=0
    )
    rng = np.random.default_rng(42)
    edges = np.eye(n_dim)
    expected = np.arange(n_dim)
    for _ in range(25):
        params = optimizer._sample_params(rng, n_params=n_dim * n_dim)
        centroids = params.reshape(n_dim, n_dim)
        assert centroids.shape == (n_dim, n_dim)
        np.testing.assert_allclose(centroids.sum(axis=1), 1.0, atol=1e-9)
        preds = multiclass_strategies._assign_nearest_centroid(
            edges, centroids
        )
        np.testing.assert_array_equal(preds, expected)


def _direct_sq_dist(points, centroids):
    """Direct ``sum((p - c)^2)`` distance matrix -- the correctness oracle.

    This is the pre-GEMM implementation's arithmetic. The production
    `_assign_nearest_centroid` uses the algebraically-identical GEMM
    expansion for speed; it must keep returning a *valid* nearest
    centroid w.r.t. these direct distances.
    """
    diff = points[:, None, :] - centroids[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


@pytest.mark.parametrize("k", [2, 3, 10, 26, 100])
@pytest.mark.parametrize("n", [1, 7, 503])
def test_assign_nearest_centroid_is_valid_argmin(k, n):
    """GEMM assignment is always a (numerically) valid nearest centroid.

    `_assign_nearest_centroid` is results-affecting, so the GEMM
    expansion must not regress the assignment. The principled contract
    (robust to a rare, benign floating-point near-tie) is: the centroid
    it picks is, by the *direct* squared distance, within 1e-9 of the
    true row minimum -- i.e. it never returns a meaningfully-farther
    centroid (the symptom of the ``->j`` axis bug, which gave excess
    ~0.2). Empirically the assignment is in fact bit-identical to the
    direct argmin across K up to 100, so we also assert exact agreement
    at a fixed seed; the tolerance bound is the hard guarantee.
    """
    rng = np.random.default_rng(20260517 + k * 1000 + n)
    for _ in range(20):
        points = rng.dirichlet(np.ones(k), size=n)
        centroids = rng.dirichlet(np.ones(k), size=k)
        assign = multiclass_strategies._assign_nearest_centroid(
            points, centroids
        )
        d2 = _direct_sq_dist(points, centroids)
        excess = d2[np.arange(n), assign] - d2.min(axis=1)
        assert excess.max() <= 1e-9, (k, n, float(excess.max()))
        np.testing.assert_array_equal(assign, np.argmin(d2, axis=1))
        assert assign.dtype == np.intp
    # Structured invariant (margin >> FP): simplex vertices assign to
    # their own centroid under an identity centroid set, every k.
    eye = np.eye(k)
    np.testing.assert_array_equal(
        multiclass_strategies._assign_nearest_centroid(eye, eye),
        np.arange(k),
    )


def test_swap_max_counterexample_violates_dlc_at_k_equals_10():
    """Numerical witness for the swap-max footnote in tex/main.tex.

    The dissertation claims that the swap-max sampler at K=10 can produce a
    configuration in which every centroid satisfies the swap-max
    construction (``argmax_j mu_i[j] == i``) yet fails the DLC
    invariant ``argmin_j d²(e_k, mu_j) == {k}``. This test fixes the
    explicit counterexample given in the footnote so that any later
    rewrite of the proof is checked against an executable witness.
    """
    K = 10
    centroids = np.zeros((K, K))

    mu_9 = np.array(
        [0.0, 0.195, 0.139, 0.0, 0.014, 0.270, 0.052, 0.0, 0.0, 0.330]
    )
    np.testing.assert_allclose(mu_9.sum(), 1.0, atol=1e-9)
    assert int(mu_9.argmax()) == 9
    assert abs(float(mu_9 @ mu_9) - 0.242) < 5e-3

    # mu_4 as described in the footnote: 0.4 at position 4, 0.379 at
    # position 9, the remaining 0.221 split evenly over the eight other
    # coordinates so the squared norm stays below 0.340.
    mu_4 = np.full(K, 0.221 / 8)
    mu_4[4] = 0.4
    mu_4[9] = 0.379
    np.testing.assert_allclose(mu_4.sum(), 1.0, atol=1e-9)
    assert int(mu_4.argmax()) == 4
    assert float(mu_4 @ mu_4) < 0.340

    centroids[9] = mu_9
    centroids[4] = mu_4

    def dsq(vertex_idx: int, c: np.ndarray) -> float:
        return 1.0 - 2.0 * c[vertex_idx] + float(c @ c)

    # DLC at k=9 requires mu_9 to be the unique argmin of d²(e_9, mu_j).
    # The counterexample shows mu_4 is strictly closer to e_9 than mu_9 is.
    assert dsq(9, mu_4) < dsq(9, mu_9)


def test_voronoi_sampler_emits_no_runtime_warning():
    """Sequential-d² has no silent fallback path; a successful draw
    emits no warnings.
    """
    optimizer = multiclass_strategies.VoronoiCentroidOptimizer(
        n_iter=1, seed=0
    )
    rng = np.random.default_rng(0)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        for _ in range(50):
            optimizer._sample_params(rng, n_params=16)  # K=4 (the worst case)


def test_voronoi_sampler_raises_on_pathological_rng():
    """If the RNG keeps returning a draw that cannot satisfy the d²
    constraints, the sampler exhausts its retry budget and raises —
    rather than silently falling back to the identity strategy.
    """
    optimizer = multiclass_strategies.VoronoiCentroidOptimizer(
        n_iter=1, seed=0
    )

    class DegenerateRng:
        """Always returns a two-tied draw. After swap-max, subsequent
        classes cannot beat the first centroid's d² at their own vertex
        because d² ties are rejected with strict inequality."""

        def dirichlet(self, alpha):
            out = np.zeros(len(alpha))
            out[0] = 0.5
            out[1] = 0.5
            return out

    with pytest.raises(RuntimeError, match="sequential-d"):
        optimizer._sample_params(DegenerateRng(), n_params=9)


# AnalyticalMulticlass - expected risk computation


def test_analytical_multiclass_risk_computation_correct():
    """Verify that argmin(probs @ cost_matrix) is used correctly."""
    # For probs=[0.6, 0.4], cost_matrix=[[0,1],[5,0]]:
    # risk[0] = 0.6*0 + 0.4*5 = 2.0
    # risk[1] = 0.6*1 + 0.4*0 = 0.6
    # → predict 1
    cost_matrix = np.array([[0, 1], [5, 0]])
    probs = np.array([[0.6, 0.4]])
    y = np.array([1])

    class FixedModel:
        def predict_proba(self, X):
            return probs

    model = FixedModel()
    strategy = multiclass_strategies.AnalyticalMulticlass(model)
    strategy.fit(probs, y, cost_matrix)
    preds = strategy.predict(probs)
    assert preds[0] == 1


def test_analytical_multiclass_uniform_cost_equals_argmax():
    """With uniform off-diagonal costs, argmin risk == argmax prob."""
    cost_matrix = np.ones((3, 3)) - np.eye(3)
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    y = np.array([1, 0])

    class FixedModel:
        def predict_proba(self, X):
            return probs

    model = FixedModel()
    strategy = multiclass_strategies.AnalyticalMulticlass(model)
    strategy.fit(probs, y, cost_matrix)
    preds = strategy.predict(probs)
    # With uniform off-diagonal cost, min risk ↔ max prob
    np.testing.assert_array_equal([1, 0], preds)


# needs_validation_set


def test_multiclass_needs_validation_set_properties(base_setup):
    model, _, _ = base_setup
    assert (
        multiclass_strategies.ArgmaxMulticlass(model).needs_validation_set
        is False
    )
    assert (
        multiclass_strategies.AnalyticalMulticlass(model).needs_validation_set
        is False
    )
    assert (
        multiclass_strategies.IsotonicArgmaxMulticlass(
            model
        ).needs_validation_set
        is True
    )
    assert (
        multiclass_strategies.ThresholdRatioMulticlass(
            model, use_cv=False
        ).needs_validation_set
        is True
    )
    assert (
        multiclass_strategies.ThresholdRatioMulticlass(
            model, use_cv=True
        ).needs_validation_set
        is False
    )
    assert (
        multiclass_strategies.PlattScalingArgmaxMulticlass(
            model
        ).needs_validation_set
        is True
    )
    assert (
        multiclass_strategies.TemperatureScalingArgmaxMulticlass(
            model
        ).needs_validation_set
        is True
    )
    assert (
        multiclass_strategies.DirichletCalibrationMulticlass(
            model
        ).needs_validation_set
        is True
    )


# Calibration strategies (Platt, Temperature, Dirichlet) - multiclass

_TRAIN_PROBS = np.tile(
    np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), (5, 1)
)
_TRAIN_Y = np.tile(np.array([0, 1, 2]), 5)


@pytest.mark.parametrize(
    "strategy_cls",
    [
        multiclass_strategies.PlattScalingArgmaxMulticlass,
        multiclass_strategies.TemperatureScalingArgmaxMulticlass,
        multiclass_strategies.DirichletCalibrationMulticlass,
    ],
)
def test_calibration_strategy_fits_and_predicts(base_setup, strategy_cls):
    model, _, _ = base_setup
    model.fit(_TRAIN_PROBS, _TRAIN_Y)
    strategy = strategy_cls(model)
    strategy.fit(_TRAIN_PROBS, _TRAIN_Y, optimization_objective=[])
    preds = strategy.predict(_TRAIN_PROBS)
    assert preds.shape == (len(_TRAIN_Y),)
    assert set(preds).issubset({0, 1, 2})


@pytest.mark.parametrize(
    "strategy_cls",
    [
        multiclass_strategies.PlattScalingArgmaxMulticlass,
        multiclass_strategies.TemperatureScalingArgmaxMulticlass,
        multiclass_strategies.DirichletCalibrationMulticlass,
    ],
)
def test_calibration_strategy_predict_before_fit_raises(
    base_setup, strategy_cls
):
    model, X_probs, _ = base_setup
    strategy = strategy_cls(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


# Calibrate-then-Analytical strategies - multiclass

_CALIB_ANALYTICAL_MULTICLASS = [
    multiclass_strategies.PlattScalingAnalyticalMulticlass,
    multiclass_strategies.TemperatureScalingAnalyticalMulticlass,
    multiclass_strategies.DirichletCalibrationAnalyticalMulticlass,
]


@pytest.mark.parametrize("strategy_cls", _CALIB_ANALYTICAL_MULTICLASS)
def test_calib_analytical_multiclass_fits_and_predicts(strategy_cls):
    # Uniform off-diagonal cost matrix, so argmin risk == argmax prob;
    # the separable mock then yields exact label recovery, mirroring
    # test_calibrated_analytical_multiclass (IsotonicAnalytical).
    cost_matrix = np.ones((3, 3)) - np.eye(3)
    model = MockMulticlassModel()
    model.fit(_TRAIN_PROBS, _TRAIN_Y)

    strategy = strategy_cls(model)
    strategy.fit(_TRAIN_PROBS, _TRAIN_Y, cost_matrix)
    preds = strategy.predict(_TRAIN_PROBS)

    np.testing.assert_array_equal(_TRAIN_Y, preds)


@pytest.mark.parametrize("strategy_cls", _CALIB_ANALYTICAL_MULTICLASS)
def test_calib_analytical_multiclass_needs_validation_set(
    base_setup, strategy_cls
):
    model, _, _ = base_setup
    assert strategy_cls(model).needs_validation_set is True


@pytest.mark.parametrize("strategy_cls", _CALIB_ANALYTICAL_MULTICLASS)
def test_calib_analytical_multiclass_raises_on_metric_objective(
    base_setup, strategy_cls
):
    model, X_probs, y = base_setup
    strategy = strategy_cls(model)
    metric_obj = metrics.MetricObjective(lambda y, p: 0, True)
    with pytest.raises(ValueError, match="requires a cost matrix"):
        strategy.fit(X_probs, y, metric_obj)


@pytest.mark.parametrize("strategy_cls", _CALIB_ANALYTICAL_MULTICLASS)
def test_calib_analytical_multiclass_predict_before_fit_raises(
    base_setup, strategy_cls
):
    model, X_probs, _ = base_setup
    strategy = strategy_cls(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


@pytest.mark.parametrize(
    "strategy_cls",
    [
        multiclass_strategies.PlattScalingAnalyticalMulticlass,
        multiclass_strategies.TemperatureScalingAnalyticalMulticlass,
    ],
)
def test_calib_analytical_multiclass_pads_missing_classes(strategy_cls):
    """The CalibratedClassifierCV-backed analytical strategies inherit
    the missing-class padding from _CalibratedAnalyticalMulticlass: if
    the calibrator saw only a subset of classes its predict_proba is
    narrower than the cost matrix and must be zero-padded before the
    risk matmul. Dirichlet is excluded (full-width by construction)."""

    cost_matrix = np.ones((4, 4)) - np.eye(4)

    class MockPartialCalibrator:
        # Calibrator only saw classes [0, 1, 2]; class 3 is absent.
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, X):
            return np.tile(np.array([[0.5, 0.3, 0.2]]), (len(X), 1))

    strategy = strategy_cls(MockMulticlassModel())
    strategy.calibrated_model = MockPartialCalibrator()
    strategy.cost_matrix = cost_matrix

    preds = strategy.predict(np.zeros((3, 5)))
    # Padded probs = [0.5, 0.3, 0.2, 0.0]; risks argmin at class 0
    # (same arithmetic as the IsotonicAnalytical padding test).
    assert preds.shape == (3,)
    np.testing.assert_array_equal(preds, np.array([0, 0, 0]))


def test_dirichlet_calibration_multiclass_reduces_nll(base_setup):
    """DirichletCalibrator minimises NLL: train <= raw."""
    model, _, _ = base_setup
    model.fit(_TRAIN_PROBS, _TRAIN_Y)

    probs = model.predict_proba(_TRAIN_PROBS)
    nll_before = -np.mean(
        np.log(np.clip(probs[np.arange(len(_TRAIN_Y)), _TRAIN_Y], 1e-12, 1.0))
    )

    strategy = multiclass_strategies.DirichletCalibrationMulticlass(model)
    strategy.fit(_TRAIN_PROBS, _TRAIN_Y, optimization_objective=[])

    calibrated = strategy.calibrator.predict_proba(probs)
    nll_after = -np.mean(
        np.log(
            np.clip(calibrated[np.arange(len(_TRAIN_Y)), _TRAIN_Y], 1e-12, 1.0)
        )
    )

    assert nll_after <= nll_before + 1e-6


# SequentialSelectionMulticlass._order_predict semantics


def _make_sequential_strategy():
    return multiclass_strategies.SequentialSelectionMulticlass(
        MockMulticlassModel()
    )


def test_order_predict_multiple_pass_earliest_in_order_wins():
    strategy = _make_sequential_strategy()
    order = (2, 0, 1)  # class 2 is tried first
    probs = np.array(
        [[0.6, 0.2, 0.7]]
    )  # classes 0 and 2 pass their thresholds
    thresholds = np.array([0.5, 0.5, 0.5])

    preds = strategy._order_predict(order, probs, thresholds)
    # Class 2 is earlier in `order`, so it should win over class 0.
    np.testing.assert_array_equal(preds, [2])


def test_order_predict_no_class_passes_falls_back_to_argmax():
    strategy = _make_sequential_strategy()
    order = (0, 1, 2)
    probs = np.array([[0.3, 0.5, 0.2]])  # nothing >= 0.9
    thresholds = np.array([0.9, 0.9, 0.9])

    preds = strategy._order_predict(order, probs, thresholds)
    np.testing.assert_array_equal(preds, [1])  # argmax fallback


def test_order_predict_reversed_order_changes_winner():
    strategy = _make_sequential_strategy()
    probs = np.array([[0.8, 0.7, 0.6]])
    thresholds = np.array([0.5, 0.5, 0.5])

    preds_fwd = strategy._order_predict((0, 1, 2), probs, thresholds)
    preds_rev = strategy._order_predict((2, 1, 0), probs, thresholds)
    np.testing.assert_array_equal(preds_fwd, [0])
    np.testing.assert_array_equal(preds_rev, [2])


def test_order_predict_per_sample_independence():
    strategy = _make_sequential_strategy()
    order = (1, 0, 2)
    probs = np.array(
        [
            [0.2, 0.9, 0.1],  # class 1 passes first in order -> 1
            [0.8, 0.1, 0.2],  # class 1 fails, class 0 passes -> 0
            [0.1, 0.1, 0.1],  # nothing passes -> argmax fallback (tie on 0)
        ]
    )
    thresholds = np.array([0.5, 0.5, 0.5])

    preds = strategy._order_predict(order, probs, thresholds)
    np.testing.assert_array_equal(preds, [1, 0, 0])


# FullMatrixPartitioningStrategy


class _RecordingOptimizer(optimization.ParameterOptimizer):
    """No-op optimizer that records its call signature and returns the
    supplied current_best_params unchanged.

    Used to inspect what FullMatrixPartitioningStrategy.fit passes to the
    optimizer and to force a deterministic Delta = 0 outcome so warm-start
    invariants can be checked exactly.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def optimize(
        self,
        objective_fn,
        n_params,
        current_best_params=None,
        current_best_score=-np.inf,
    ):
        params = (
            np.zeros(n_params)
            if current_best_params is None
            else np.asarray(current_best_params)
        )
        self.calls.append(
            {
                "n_params": n_params,
                "current_best_params": params.copy(),
                "current_best_score": current_best_score,
            }
        )
        return params, current_best_score


def _fixed_probs_model(probs: np.ndarray):
    class FixedModel:
        def predict_proba(self, X):
            n = len(np.asarray(X))
            if probs.shape[0] == 1 and n != 1:
                return np.tile(probs, (n, 1))
            return probs

    return FixedModel()


def _balanced_dataset(K: int, n_per_class: int = 4) -> tuple:
    """Construct a small balanced dataset with predict_proba = identity-ish
    soft labels. Each row of X is the one-hot encoded class with the rest
    of the mass spread uniformly, so the base model below produces
    sensible probabilities and StratifiedKFold can split evenly.
    """
    soft = np.full((K, K), 0.05 / max(1, K - 1))
    np.fill_diagonal(soft, 0.95)
    X = np.tile(soft, (n_per_class, 1))
    y = np.tile(np.arange(K), n_per_class)

    class IdentityProbaModel(MockMulticlassModel):
        def predict_proba(self, X_in):
            arr = np.asarray(X_in)
            return arr[:, :K]

    model = IdentityProbaModel()
    model.fit(X, y)
    return model, X, y


def test_full_matrix_initial_state(base_setup):
    model, _, _ = base_setup
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(model)
    assert strategy.partition_matrix is None
    assert strategy._fold_results is None


def test_full_matrix_predict_before_fit_raises(base_setup):
    model, X_probs, _ = base_setup
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(model)
    with pytest.raises(ValueError, match="not fit"):
        strategy.predict(X_probs)


def test_full_matrix_properties_no_cv(base_setup):
    model, _, _ = base_setup
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, use_cv=False
    )
    assert strategy.needs_validation_set is True
    assert strategy.self_tunes_with_cv is False


def test_full_matrix_properties_cv(base_setup):
    model, _, _ = base_setup
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, use_cv=True
    )
    assert strategy.needs_validation_set is False
    assert strategy.self_tunes_with_cv is True


@pytest.mark.parametrize("K", [2, 3, 5, 10])
def test_full_matrix_shape_after_fit(K):
    model, X, y = _balanced_dataset(K, n_per_class=3)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, cost_matrix)
    assert strategy.partition_matrix.shape == (K, K)


def test_full_matrix_predict_shape_and_dtype():
    K = 3
    model, X, y = _balanced_dataset(K, n_per_class=3)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, cost_matrix)
    preds = strategy.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.issubdtype(preds.dtype, np.integer)
    assert preds.min() >= 0
    assert preds.max() < K


def test_full_matrix_predict_deterministic():
    K = 3
    model, X, y = _balanced_dataset(K, n_per_class=3)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, cost_matrix)
    np.testing.assert_array_equal(strategy.predict(X), strategy.predict(X))


def test_full_matrix_warm_start_cost_matrix_equals_analytical():
    """At Delta=0 the strategy must reduce to AnalyticalMulticlass."""
    cost_matrix = np.array([[0, 1, 2], [3, 0, 1], [2, 1, 0]], dtype=float)
    probs = np.array(
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float
    )
    y = np.array([0, 1, 2])
    model = _fixed_probs_model(probs)

    full = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    full.fit(probs, y, cost_matrix)

    np.testing.assert_allclose(full.partition_matrix, cost_matrix)

    analytical = multiclass_strategies.AnalyticalMulticlass(model)
    analytical.fit(probs, y, cost_matrix)
    np.testing.assert_array_equal(
        full.predict(probs), analytical.predict(probs)
    )


def test_full_matrix_warm_start_metric_equals_argmax():
    """For a MetricObjective the anchor is the symmetric 0/1 cost matrix,
    so Delta=0 yields the Argmax rule."""
    K = 3
    probs = np.array(
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float
    )
    y = np.array([0, 1, 2])
    model = _fixed_probs_model(probs)
    metric_obj = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )

    full = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    full.fit(probs, y, metric_obj)

    expected_anchor = np.ones((K, K)) - np.eye(K)
    np.testing.assert_allclose(full.partition_matrix, expected_anchor)

    argmax = multiclass_strategies.ArgmaxMulticlass(model)
    argmax.fit(probs, y, metric_obj)
    np.testing.assert_array_equal(full.predict(probs), argmax.predict(probs))


def test_full_matrix_optimizer_call_signature():
    K = 4
    model, X, y = _balanced_dataset(K, n_per_class=2)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    optimizer = _RecordingOptimizer()
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=optimizer
    )
    strategy.fit(X, y, cost_matrix)

    assert len(optimizer.calls) == 1
    call = optimizer.calls[0]
    assert call["n_params"] == K * K
    np.testing.assert_array_equal(call["current_best_params"], np.zeros(K * K))
    # current_best_score equals objective evaluated at Delta=0; for this
    # data the warm-start matches the labels exactly (cost = 0).
    assert call["current_best_score"] == 0.0


def test_full_matrix_random_search_can_improve_over_warm_start():
    """With miscalibrated probabilities and an asymmetric cost matrix, a
    real RandomSearchOptimizer should learn a Delta that strictly improves
    over the analytical warm start.
    """
    cost_matrix = np.array([[0.0, 10.0], [1.0, 0.0]])
    probs = np.array(
        [
            [0.55, 0.45],
            [0.55, 0.45],
            [0.55, 0.45],
            [0.55, 0.45],
            [0.55, 0.45],
            [0.55, 0.45],
            [0.55, 0.45],
        ]
    )
    # Truth is class 1 for all rows: predicting class 1 costs 0; predicting
    # class 0 costs 10 per row. Analytical plug-in picks class 0 because
    # 0.55*0 + 0.45*1 = 0.45 < 0.45*10 = 4.5.
    y = np.array([1, 1, 1, 1, 1, 1, 1])
    model = _fixed_probs_model(probs[:1])

    analytical = multiclass_strategies.AnalyticalMulticlass(model)
    analytical.fit(probs, y, cost_matrix)
    analytical_cost = cost_matrix[y, analytical.predict(probs)].mean()
    assert analytical_cost > 0  # warm start is wrong here

    full = multiclass_strategies.FullMatrixPartitioningStrategy(
        model,
        optimizer=optimization.RandomSearchOptimizer(
            bounds=(-5.0, 5.0), n_iter=200, n_jobs=1
        ),
    )
    full.fit(probs, y, cost_matrix)
    learned_cost = cost_matrix[y, full.predict(probs)].mean()
    assert learned_cost < analytical_cost


def test_full_matrix_fits_with_cost_matrix():
    K = 3
    model, X, y = _balanced_dataset(K, n_per_class=3)
    cost_matrix = np.array([[0, 1, 5], [1, 0, 1], [5, 1, 0]], dtype=float)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, cost_matrix)
    preds = strategy.predict(X)
    assert preds.shape == (X.shape[0],)


def test_full_matrix_fits_with_metric_objective():
    """FullMatrix lifts the cost-matrix-only restriction of
    RiskPartitioning by routing all objectives through random search."""
    K = 3
    model, X, y = _balanced_dataset(K, n_per_class=3)
    f1_macro = metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.f1_score(
            yt, yp, average="macro", zero_division=0
        ),
        higher_is_better=True,
    )
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, f1_macro)
    preds = strategy.predict(X)
    assert preds.shape == (X.shape[0],)
    assert strategy.partition_matrix.shape == (K, K)


def test_full_matrix_cv_populates_fold_results():
    K = 2
    model, X, y = _balanced_dataset(K, n_per_class=4)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer(), use_cv=True, cv_folds=2
    )
    strategy.fit(X, y, cost_matrix)
    assert strategy._fold_results is not None
    assert len(strategy._fold_results) == 2


@pytest.mark.parametrize("cv_folds", [2, 3])
def test_full_matrix_cv_respects_n_folds(cv_folds):
    K = 2
    model, X, y = _balanced_dataset(K, n_per_class=cv_folds * 2)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model,
        optimizer=_RecordingOptimizer(),
        use_cv=True,
        cv_folds=cv_folds,
    )
    strategy.fit(X, y, cost_matrix)
    assert len(strategy._fold_results) == cv_folds


def test_full_matrix_cv_reuses_fold_cache_across_objectives():
    """A single instance fit twice (different objectives) must refit the
    K folds exactly once."""
    fit_calls: list[int] = []

    class CountingModel(MockMulticlassModel):
        def fit(self, X, y):
            fit_calls.append(len(X))
            return super().fit(X, y)

        def predict_proba(self, X):
            arr = np.asarray(X)
            return arr[:, :2]

    model = CountingModel()
    X = np.tile(np.array([[0.7, 0.3], [0.3, 0.7]]), (4, 1))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    model.fit(X, y)
    calls_after_training = len(fit_calls)

    accuracy = metrics.MetricObjective(
        metric_fn=sklearn_metrics.accuracy_score, higher_is_better=True
    )
    f1_macro = metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.f1_score(
            yt, yp, average="macro", zero_division=0
        ),
        higher_is_better=True,
    )

    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer(), use_cv=True, cv_folds=2
    )
    strategy.fit(X, y, accuracy)
    calls_after_first = len(fit_calls)
    strategy.fit(X, y, f1_macro)
    calls_after_second = len(fit_calls)

    assert calls_after_first == calls_after_training + 2
    assert calls_after_second == calls_after_first
    assert strategy._fold_results is not None
    assert len(strategy._fold_results) == 2


def test_full_matrix_uses_default_random_search_optimizer(base_setup):
    model, _, _ = base_setup
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(model)
    assert isinstance(strategy.optimizer, optimization.RandomSearchOptimizer)


def test_full_matrix_accepts_custom_optimizer(base_setup):
    model, _, _ = base_setup
    custom = _RecordingOptimizer()
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=custom
    )
    assert strategy.optimizer is custom


def test_full_matrix_param_extractor_returns_flat_list():
    """The extractor must flatten the K x K matrix to a 1-D list of length
    K*K. A nested list would make the learned_params parquet column mix
    list and list-of-list values, which pyarrow rejects at write time.
    Same flat convention as VoronoiTuning's centroids."""
    pytest.importorskip("experiments")
    from experiments.multiclass.runner import _multiclass_param_extractor

    K = 3
    model, X, y = _balanced_dataset(K, n_per_class=2)
    cost_matrix = np.ones((K, K)) - np.eye(K)
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, cost_matrix)
    extracted = _multiclass_param_extractor(strategy)

    assert isinstance(extracted, list)
    assert len(extracted) == K * K
    assert all(isinstance(v, float) for v in extracted)


def test_full_matrix_learned_params_column_is_arrow_convertible():
    """Regression: a learned_params column mixing FullMatrix output with a
    shorter affine vector and a None must convert to an Arrow table. A
    nested K x K list here triggers
    'cannot mix list and non-list values' at parquet write, which crashed
    the multiclass run end-to-end while the unit tests still passed."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("experiments")
    import pandas as pd

    from experiments.multiclass.runner import _multiclass_param_extractor

    K = 3
    model, X, y = _balanced_dataset(K, n_per_class=2)
    cost_matrix = np.ones((K, K)) - np.eye(K)

    full = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    full.fit(X, y, cost_matrix)
    affine = multiclass_strategies.RiskPartitioningStrategy(
        model, optimizer=MockOptimizer(best_params=[0.0, 0.0])
    )
    affine.fit(X, y, cost_matrix)

    df = pd.DataFrame(
        {
            "learned_params": [
                _multiclass_param_extractor(full),
                _multiclass_param_extractor(affine),
                None,
            ]
        }
    )
    # Must not raise ArrowInvalid.
    table = pa.Table.from_pandas(df, preserve_index=False)
    assert table.num_rows == 3


def test_full_matrix_k_equals_2():
    """K=2 is an edge case for the K x K reshape and the metric anchor."""
    K = 2
    model, X, y = _balanced_dataset(K, n_per_class=3)
    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(
        model, optimizer=_RecordingOptimizer()
    )
    strategy.fit(X, y, cost_matrix)
    assert strategy.partition_matrix.shape == (2, 2)
    preds = strategy.predict(X)
    assert set(np.unique(preds).tolist()).issubset({0, 1})


def test_full_matrix_label_validation(base_setup):
    model, _, _ = base_setup
    strategy = multiclass_strategies.FullMatrixPartitioningStrategy(model)
    # Label out of range: cost matrix is 2x2 but y has a 5.
    cost_matrix = np.eye(2)
    X = np.array([[0.5, 0.5], [0.4, 0.6]])
    y = np.array([0, 5])
    with pytest.raises(ValueError, match="Labels must be in"):
        strategy.fit(X, y, cost_matrix)
