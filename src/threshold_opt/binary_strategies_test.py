import numpy as np
import pytest
from sklearn import base
from sklearn import metrics as sklearn_metrics

from threshold_opt import binary_strategies, metrics, optimization


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


class MockModel(base.ClassifierMixin, base.BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.array([[1 - x[0], x[0]] for x in X])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


@pytest.fixture
def setup():
    model = MockModel()
    X = np.array([[0.1], [0.6], [0.4], [0.8]])
    y = np.array([0, 1, 0, 1])
    cost_matrix = [[0, 1], [10, 0]]
    return model, X, y, cost_matrix


def test_predict_raises_value_error_if_not_fit(setup):
    model, X, _y, _cost_matrix = setup
    strategy = binary_strategies.ArgmaxThreshold(model)
    with pytest.raises(ValueError, match="Threshold not determined"):
        strategy.predict(X)


def test_fit_raises_value_error_for_invalid_labels(setup):
    model, X, _y, cost_matrix = setup
    strategy = binary_strategies.ArgmaxThreshold(model)
    with pytest.raises(
        ValueError, match="Binary strategies require labels to be 0 or 1"
    ):
        strategy.fit(X, np.array([0, 2]), cost_matrix)


def test_argmax_threshold(setup):
    model, X, y, cost_matrix = setup
    strategy = binary_strategies.ArgmaxThreshold(model)
    strategy.fit(X, y, cost_matrix)
    assert strategy.threshold == 0.5
    np.testing.assert_array_equal([0, 1, 0, 1], strategy.predict(X))


def test_calibrated_argmax(setup):
    model, X, y, cost_matrix = setup
    model.fit(X, y)
    strategy = binary_strategies.IsotonicArgmax(model)
    X_large = np.tile(X, (5, 1))
    y_large = np.tile(y, 5)
    strategy.fit(X_large, y_large, cost_matrix)
    assert strategy.threshold == 0.5
    assert strategy.calibrated_model is not None


def test_analytical_threshold(setup):
    model, X, y, cost_matrix = setup
    strategy = binary_strategies.AnalyticalThreshold(model)
    strategy.fit(X, y, cost_matrix)
    assert strategy.threshold == pytest.approx(1 / 11)


def test_analytical_threshold_symmetric_cost_equals_argmax(setup):
    """Symmetric 0/1 cost: argmin risk == argmax prob, i.e. threshold 0.5."""
    model, X, y, _ = setup
    strategy = binary_strategies.AnalyticalThreshold(model)
    strategy.fit(X, y, [[0, 1], [1, 0]])
    assert strategy.threshold == pytest.approx(0.5)
    np.testing.assert_array_equal(strategy.predict(X), model.predict(X))


def test_analytical_threshold_with_metric_objective_raises_error(setup):
    model, X, y, _cost_matrix = setup
    strategy = binary_strategies.AnalyticalThreshold(model)
    metric_objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.f1_score, higher_is_better=True
    )
    with pytest.raises(ValueError):
        strategy.fit(X, y, metric_objective)


@pytest.mark.parametrize(
    "objective,expected_range",
    [
        pytest.param([[0, 1], [10, 0]], (0.4, 0.6), id="cost_matrix"),
        pytest.param(
            metrics.MetricObjective(
                metric_fn=sklearn_metrics.f1_score, higher_is_better=True
            ),
            (0.4, 0.6),
            id="f1_score",
        ),
        pytest.param(
            metrics.MetricObjective(
                metric_fn=sklearn_metrics.log_loss, higher_is_better=False
            ),
            (0.0, 1.0),
            id="log_loss",
        ),
    ],
)
def test_empirical_threshold(setup, objective, expected_range):
    model, X, y, _cost_matrix = setup
    strategy = binary_strategies.EmpiricalThreshold(model)
    strategy.fit(X, y, objective)
    assert expected_range[0] <= strategy.threshold <= expected_range[1]


def test_empirical_threshold_stability(setup):
    model, X, y, _cost_matrix = setup
    mock_optimizer = MockOptimizer(best_params=[0.123])
    strategy = binary_strategies.EmpiricalThreshold(
        model, optimizer=mock_optimizer
    )
    metric_objective = metrics.MetricObjective(
        metric_fn=lambda y_t, y_p: 0.5, higher_is_better=True
    )
    strategy.fit(X, y, metric_objective)
    assert strategy.threshold == pytest.approx(0.123)


def test_empirical_threshold_cv():
    model = MockModel()
    strategy = binary_strategies.EmpiricalThreshold(
        model, use_cv=True, cv_folds=2
    )
    metric_objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.f1_score, higher_is_better=True
    )
    # 8 samples (4 per class) so 2-fold StratifiedKFold has a real fold on
    # each side; X values are cleanly separated at 0.5, so the learned
    # threshold must land in [0.4, 0.6] regardless of the fold split.
    X_large = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]]
    )
    y_large = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    strategy.fit(X_large, y_large, metric_objective)

    assert 0.4 <= strategy.threshold <= 0.6


def test_calibrated_analytical(setup):
    model, X, y, cost_matrix = setup
    model.fit(X, y)
    strategy = binary_strategies.IsotonicAnalytical(model)
    X_large = np.tile(X, (5, 1))
    y_large = np.tile(y, 5)
    strategy.fit(X_large, y_large, cost_matrix)
    assert strategy.threshold == pytest.approx(1 / 11)
    assert strategy.calibrated_model is not None


def test_calibrated_analytical_calibration_effect():
    class InvertedModel(base.ClassifierMixin, base.BaseEstimator):
        _estimator_type = "classifier"

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            n = len(X)
            return np.tile(
                np.array([[0.8, 0.2], [0.2, 0.8]]), (n // 2 + 1, 1)
            )[:n]

        def predict(self, X):
            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(int)

    model = InvertedModel()
    n_repeats = 5
    X_fit = np.tile(np.array([[0], [1]]), (n_repeats, 1))
    y_fit = np.tile(np.array([1, 0]), n_repeats)
    model.fit(X_fit, y_fit)

    strategy = binary_strategies.IsotonicAnalytical(model)
    cost_matrix = [[0, 2], [3, 0]]
    strategy.fit(X_fit, y_fit, cost_matrix)
    assert strategy.threshold == pytest.approx(0.4)

    X_test = np.array([[0], [1]])
    preds = strategy.predict(X_test)
    np.testing.assert_array_equal([1, 1], preds)


# needs_validation_set property


def test_needs_validation_set_properties(setup):
    model, _, _, _ = setup
    assert (
        binary_strategies.ArgmaxThreshold(model).needs_validation_set is False
    )
    assert (
        binary_strategies.AnalyticalThreshold(model).needs_validation_set
        is False
    )
    assert binary_strategies.IsotonicArgmax(model).needs_validation_set is True
    assert (
        binary_strategies.IsotonicAnalytical(model).needs_validation_set
        is True
    )
    assert (
        binary_strategies.EmpiricalThreshold(
            model, use_cv=False
        ).needs_validation_set
        is True
    )
    assert (
        binary_strategies.EmpiricalThreshold(
            model, use_cv=True
        ).needs_validation_set
        is False
    )


# IsotonicArgmax


def test_calibrated_argmax_predict_before_fit_raises(setup):
    model, X, _, _ = setup
    strategy = binary_strategies.IsotonicArgmax(model)
    with pytest.raises(ValueError):
        strategy.predict(X)


# IsotonicAnalytical


def test_calibrated_analytical_raises_on_metric_objective(setup):
    model, X, y, _ = setup
    model.fit(X, y)
    strategy = binary_strategies.IsotonicAnalytical(model)
    metric_obj = metrics.MetricObjective(
        metric_fn=lambda yt, yp: 0.0, higher_is_better=True
    )
    with pytest.raises(ValueError, match="requires a cost matrix"):
        strategy.fit(X, y, metric_obj)


# AnalyticalThreshold - extreme cost matrices


def test_analytical_threshold_extreme_costs_below_zero():
    """Cost matrix where formula yields threshold < 0 → always predict 1."""

    class ConstantModel(base.ClassifierMixin, base.BaseEstimator):
        _estimator_type = "classifier"

        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return np.column_stack(
                [np.full(len(X), 0.3), np.full(len(X), 0.7)]
            )

        def predict(self, X):
            return np.ones(len(X), dtype=int)

    model = ConstantModel()
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    # C_01=1, C_00=0, C_10=0, C_11=0: numerator=1, denominator=1+0-0=1 → 1.0
    # C_01=-5, C_10=1: numerator=-5, denominator=-5+1=-4 → 1.25
    # For threshold > 1: all probs < threshold → predict 0
    cost_matrix = [[0, -5], [1, 0]]
    strategy = binary_strategies.AnalyticalThreshold(model)
    model.fit(X, y)
    strategy.fit(X, y, cost_matrix)
    threshold = strategy.threshold
    assert threshold > 1.0
    preds = strategy.predict(X)
    np.testing.assert_array_equal([0, 0], preds)


def test_analytical_threshold_extreme_costs_above_one():
    """Cost matrix yielding threshold > 1: all probs below it, predict 0."""

    class ConstantModel(base.ClassifierMixin, base.BaseEstimator):
        _estimator_type = "classifier"

        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return np.column_stack(
                [np.full(len(X), 0.7), np.full(len(X), 0.3)]
            )

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    model = ConstantModel()
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    # C_01=1, C_10=-5: numerator=1, denominator=1+(-5)=-4 → -0.25
    # Threshold < 0: all probs >= threshold → predict 1
    cost_matrix = [[0, 1], [-5, 0]]
    strategy = binary_strategies.AnalyticalThreshold(model)
    model.fit(X, y)
    strategy.fit(X, y, cost_matrix)
    threshold = strategy.threshold
    assert threshold < 0.0
    preds = strategy.predict(X)
    np.testing.assert_array_equal([1, 1], preds)


# EmpiricalThreshold - CV behavior


def test_empirical_threshold_cv_does_not_mutate_model():
    """CV mode must clone per fold, not mutate self.model."""

    class TrackingModel(base.ClassifierMixin, base.BaseEstimator):
        _estimator_type = "classifier"

        def __init__(self):
            self.fit_calls = []

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.fit_calls.append(len(X))
            return self

        def predict_proba(self, X):
            return np.array([[1 - x[0], x[0]] for x in X])

        def predict(self, X):
            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(int)

    model = TrackingModel()
    X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model.fit(X, y)
    calls_before = len(model.fit_calls)

    strategy = binary_strategies.EmpiricalThreshold(
        model, use_cv=True, cv_folds=2
    )
    metric_objective = metrics.MetricObjective(
        metric_fn=lambda yt, yp: float(np.mean(yt == yp)),
        higher_is_better=True,
    )
    strategy.fit(X, y, metric_objective)

    # Original model should not have been re-fitted by CV folds
    assert len(model.fit_calls) == calls_before, (
        "EmpiricalThreshold CV mode must not re-fit self.model; "
        "use clone(self.model) per fold instead"
    )


def test_empirical_threshold_accepts_preseeded_fold_results():
    """Pipeline-level CV precompute can seed ``_fold_results``; when it
    does, ``_fit`` must not refit the underlying model itself."""

    class TrackingModel(base.ClassifierMixin, base.BaseEstimator):
        _estimator_type = "classifier"

        def __init__(self):
            self.fit_calls = []

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.fit_calls.append(len(X))
            return self

        def predict_proba(self, X):
            return np.array([[1 - x[0], x[0]] for x in X])

    from threshold_opt import base as base_mod

    model = TrackingModel()
    X = np.array([[0.1], [0.3], [0.5], [0.7]])
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    calls_after_training = len(model.fit_calls)

    preseeded = [
        base_mod._CVFoldPredictions(
            y=np.array([0, 1]),
            probs=np.array([[0.8, 0.2], [0.1, 0.9]]),
        ),
        base_mod._CVFoldPredictions(
            y=np.array([0, 1]),
            probs=np.array([[0.6, 0.4], [0.2, 0.8]]),
        ),
    ]

    strategy = binary_strategies.EmpiricalThreshold(
        model, use_cv=True, cv_folds=2
    )
    strategy._fold_results = preseeded
    metric_objective = metrics.MetricObjective(
        metric_fn=lambda yt, yp: float(np.mean(yt == yp)),
        higher_is_better=True,
    )
    strategy.fit(X, y, metric_objective)

    # No additional fit calls on top of the initial training.
    assert len(model.fit_calls) == calls_after_training
    assert strategy._fold_results is preseeded


def test_empirical_threshold_cv_requires_grid_search_optimizer(setup):
    model, X, y, _ = setup
    random_opt = optimization.RandomSearchOptimizer(n_iter=10)
    strategy = binary_strategies.EmpiricalThreshold(
        model, optimizer=random_opt, use_cv=True
    )
    metric_obj = metrics.MetricObjective(
        metric_fn=lambda yt, yp: float(np.mean(yt == yp)),
        higher_is_better=True,
    )
    with pytest.raises(ValueError, match="GridSearchOptimizer"):
        strategy.fit(X, y, metric_obj)


def test_empirical_threshold_cost_matrix_selects_low_threshold():
    """Asymmetric costs shift the optimal threshold in the expected direction.

    The data is non-separable (y=1 at x=0.4 and y=0 at x=0.6), so every
    threshold creates either an FP or an FN.  The optimal threshold trades
    these off according to the cost matrix.
    """
    model = MockModel()
    # predict_proba returns [[1-x, x]] per sample, so prob(class=1) = x[0]
    X = np.array([[0.2], [0.4], [0.6], [0.8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)

    # FN is very expensive: lower threshold to capture borderline positives
    # t≈0.3: 0.4 classified pos (TP), 0.6 classified pos (FP cost=1) → cheap
    high_fn_cost = [[0, 1], [100, 0]]
    strategy = binary_strategies.EmpiricalThreshold(model)
    strategy.fit(X, y, high_fn_cost)
    assert strategy.threshold < 0.5

    # FP is very expensive: raise threshold to avoid false alarms
    # t≈0.7: 0.4 classified neg (FN cost=1), 0.6 classified neg (TN) → cheap
    high_fp_cost = [[0, 100], [1, 0]]
    strategy2 = binary_strategies.EmpiricalThreshold(model)
    strategy2.fit(X, y, high_fp_cost)
    assert strategy2.threshold > 0.5


# _get_positive_proba edge cases


def test_get_positive_proba_single_class_model():
    """When model only saw class 1, probabilities are extracted correctly."""

    class SingleClassModel(base.ClassifierMixin, base.BaseEstimator):
        _estimator_type = "classifier"

        def fit(self, X, y):
            self.classes_ = np.array([1])
            return self

        def predict_proba(self, X):
            return np.ones((len(X), 1))

        def predict(self, X):
            return np.ones(len(X), dtype=int)

    model = SingleClassModel()
    X = np.array([[0.1], [0.5]])
    y = np.array([1, 1])
    model.fit(X, y)

    strategy = binary_strategies.ArgmaxThreshold(model)
    strategy.fit(X, y, [[0, 1], [1, 0]])
    preds = strategy.predict(X)
    np.testing.assert_array_equal([1, 1], preds)


# Calibration strategies (Platt, Temperature, Dirichlet) - binary

_BINARY_X = np.tile(np.array([[0.1], [0.6], [0.4], [0.8]]), (5, 1))
_BINARY_Y = np.tile(np.array([0, 1, 0, 1]), 5)


@pytest.mark.parametrize(
    "strategy_cls",
    [
        binary_strategies.PlattScalingArgmax,
        binary_strategies.TemperatureScalingArgmax,
        binary_strategies.DirichletCalibrationArgmax,
    ],
)
def test_calibration_binary_fits_and_predicts(strategy_cls):
    model = MockModel()
    model.fit(_BINARY_X, _BINARY_Y)
    strategy = strategy_cls(model)
    cost_matrix = [[0, 1], [1, 0]]
    strategy.fit(_BINARY_X, _BINARY_Y, cost_matrix)
    assert strategy.threshold == pytest.approx(0.5)
    preds = strategy.predict(_BINARY_X)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize(
    "strategy_cls",
    [
        binary_strategies.PlattScalingArgmax,
        binary_strategies.TemperatureScalingArgmax,
        binary_strategies.DirichletCalibrationArgmax,
        binary_strategies.PlattScalingAnalytical,
        binary_strategies.TemperatureScalingAnalytical,
        binary_strategies.DirichletCalibrationAnalytical,
    ],
)
def test_calibration_binary_needs_validation_set(strategy_cls):
    model = MockModel()
    assert strategy_cls(model).needs_validation_set is True


@pytest.mark.parametrize(
    "strategy_cls",
    [
        binary_strategies.PlattScalingArgmax,
        binary_strategies.TemperatureScalingArgmax,
        binary_strategies.DirichletCalibrationArgmax,
        binary_strategies.PlattScalingAnalytical,
        binary_strategies.TemperatureScalingAnalytical,
        binary_strategies.DirichletCalibrationAnalytical,
    ],
)
def test_calibration_binary_predict_before_fit_raises(strategy_cls):
    model = MockModel()
    X = np.array([[0.1], [0.6]])
    strategy = strategy_cls(model)
    with pytest.raises((ValueError, AttributeError)):
        strategy.predict(X)


@pytest.mark.parametrize(
    "strategy_cls",
    [
        binary_strategies.PlattScalingAnalytical,
        binary_strategies.TemperatureScalingAnalytical,
        binary_strategies.DirichletCalibrationAnalytical,
    ],
)
def test_calibration_analytical_fits_and_predicts(strategy_cls):
    model = MockModel()
    model.fit(_BINARY_X, _BINARY_Y)
    strategy = strategy_cls(model)
    cost_matrix = [[0, 1], [10, 0]]
    strategy.fit(_BINARY_X, _BINARY_Y, cost_matrix)
    assert strategy.threshold == pytest.approx(1 / 11)
    preds = strategy.predict(_BINARY_X)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize(
    "strategy_cls",
    [
        binary_strategies.PlattScalingAnalytical,
        binary_strategies.TemperatureScalingAnalytical,
        binary_strategies.DirichletCalibrationAnalytical,
    ],
)
def test_calibration_analytical_raises_on_metric_objective(strategy_cls):
    model = MockModel()
    model.fit(_BINARY_X, _BINARY_Y)
    strategy = strategy_cls(model)
    metric_obj = metrics.MetricObjective(
        metric_fn=sklearn_metrics.f1_score, higher_is_better=True
    )
    with pytest.raises(ValueError, match="requires a cost matrix"):
        strategy.fit(_BINARY_X, _BINARY_Y, metric_obj)


def test_dirichlet_calibration_binary_reduces_nll():
    """DirichletCalibrator minimises NLL: train NLL <= raw."""
    model = MockModel()
    model.fit(_BINARY_X, _BINARY_Y)

    probs = model.predict_proba(_BINARY_X)
    y = _BINARY_Y
    nll_before = -np.mean(
        np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
    )

    strategy = binary_strategies.DirichletCalibrationArgmax(model)
    strategy.fit(_BINARY_X, _BINARY_Y, [[0, 1], [1, 0]])

    calibrated = strategy.calibrator.predict_proba(probs)
    nll_after = -np.mean(
        np.log(np.clip(calibrated[np.arange(len(y)), y], 1e-12, 1.0))
    )

    assert nll_after <= nll_before + 1e-6


# Vectorized grid search path


def test_empirical_vectorized_cost_matrix():
    """Vectorized path produces valid threshold for cost."""
    model = MockModel()
    X = np.array([[0.2], [0.4], [0.6], [0.8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)

    cost_matrix = [[0, 1], [100, 0]]
    strategy = binary_strategies.EmpiricalThreshold(model)
    strategy.fit(X, y, cost_matrix)
    assert 0.0 <= strategy.threshold <= 1.0
    assert strategy.threshold < 0.5  # FN expensive → low threshold


def test_empirical_vectorized_metric_objective():
    """Vectorized path with vectorized_key produces a valid threshold."""
    model = MockModel()
    X = np.array([[0.2], [0.4], [0.6], [0.8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)

    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.f1_score,
        higher_is_better=True,
        vectorized_key="f1",
    )
    strategy = binary_strategies.EmpiricalThreshold(model)
    strategy.fit(X, y, objective)
    assert 0.0 <= strategy.threshold <= 1.0


def test_empirical_fallback_custom_metric():
    """MetricObjective without vectorized_key falls back to scalar path."""
    model = MockModel()
    X = np.array([[0.2], [0.4], [0.6], [0.8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)

    objective = metrics.MetricObjective(
        metric_fn=lambda yt, yp: float(np.mean(yt == yp)),
        higher_is_better=True,
        # no vectorized_key → fallback
    )
    strategy = binary_strategies.EmpiricalThreshold(model)
    strategy.fit(X, y, objective)
    assert 0.0 <= strategy.threshold <= 1.0


def test_empirical_cv_vectorized_metric_objective():
    """CV path with vectorized_key produces a valid threshold."""
    model = MockModel()
    X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model.fit(X, y)

    objective = metrics.MetricObjective(
        metric_fn=sklearn_metrics.f1_score,
        higher_is_better=True,
        vectorized_key="f1",
    )
    strategy = binary_strategies.EmpiricalThreshold(
        model, use_cv=True, cv_folds=2
    )
    strategy.fit(X, y, objective)
    assert 0.0 <= strategy.threshold <= 1.0
