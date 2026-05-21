import dataclasses
import typing

import numpy as np
import pytest
from sklearn import metrics as sklearn_metrics

from threshold_opt import metrics


@pytest.mark.parametrize(
    "cost_matrix,expected_threshold",
    [
        ([[0, 1], [1, 0]], 0.5),
        ([[0, 1], [10, 0]], 1 / 11),
        ([[0, 1], [2, 0]], 1 / 3),
        ([[0, 1], [1, 2]], 0.5),  # zero_denominator
    ],
)
def test_calculate_optimal_threshold(cost_matrix, expected_threshold):
    assert metrics.calculate_optimal_threshold(cost_matrix) == pytest.approx(
        expected_threshold
    )


def test_calculate_optimal_threshold_invalid_shape():
    cost_matrix = [[0, 1, 2], [1, 2, 3], [3, 4, 5]]
    with pytest.raises(ValueError, match="requires 2x2 matrix"):
        metrics.calculate_optimal_threshold(cost_matrix)


@pytest.mark.parametrize(
    "y_true,y_pred,cost_matrix,expected_risk",
    [
        (
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1]),
            [[0, 1], [1, 0]],
            0.0,
        ),
        (
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            [[0, 1], [10, 0]],
            2.75,
        ),
    ],
)
def test_calculate_empirical_risk(y_true, y_pred, cost_matrix, expected_risk):
    assert metrics.calculate_empirical_risk(
        y_true, y_pred, cost_matrix
    ) == pytest.approx(expected_risk)


def test_calculate_empirical_risk_invalid_cost_matrix():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    cost_matrix = [[0, 1, 2], [1, 0, 0]]  # Not square
    with pytest.raises(ValueError, match="Cost matrix must be square"):
        metrics.calculate_empirical_risk(y_true, y_pred, cost_matrix)


@pytest.mark.parametrize(
    "cost_matrix,expected_threshold",
    [
        # Asymmetric: FP much more expensive than FN → high threshold
        ([[0, 10], [1, 0]], 10 / 11),
        # FP cost equals FN cost → 0.5
        ([[0, 3], [3, 0]], 0.5),
        # Non-zero diagonal (rewards for correct prediction)
        ([[0, 2], [4, -1]], pytest.approx(2 / (2 + 5))),
    ],
)
def test_calculate_optimal_threshold_asymmetric(
    cost_matrix, expected_threshold
):
    result = metrics.calculate_optimal_threshold(cost_matrix)
    assert result == expected_threshold


def test_calculate_optimal_threshold_returns_value_outside_unit_interval():
    """Formula can yield threshold outside [0, 1] for unusual cost matrices."""
    # C_01=-5, C_10=1: numerator=-5, denominator=-5+1=-4 → 1.25
    cost_matrix = [[0, -5], [1, 0]]
    result = metrics.calculate_optimal_threshold(cost_matrix)
    assert result > 1.0

    # C_01=1, C_10=-5: numerator=1, denominator=1+(-5)=-4 → -0.25
    cost_matrix2 = [[0, 1], [-5, 0]]
    result2 = metrics.calculate_optimal_threshold(cost_matrix2)
    assert result2 < 0.0


def test_calculate_empirical_risk_multiclass_3class():
    """Correct risk calculation for 3-class case."""
    # Predict everything as class 0; class 1 and 2 are mispredicted
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 0])
    cost_matrix = np.array(
        [
            [0, 1, 2],
            [3, 0, 1],
            [4, 2, 0],
        ]
    )
    # true 0 pred 0: cost 0; true 1 pred 0: cost 3; true 2 pred 0: cost 4
    # total = 0 + 3 + 4 = 7; avg = 7/3
    result = metrics.calculate_empirical_risk(y_true, y_pred, cost_matrix)
    assert result == pytest.approx(7 / 3)


def test_calculate_empirical_risk_all_misclassified():
    """All wrong predictions → risk equals average FN/FP cost."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    cost_matrix = [[0, 5], [2, 0]]
    # 2 FP (cost 5 each), 2 FN (cost 2 each)
    # total = 2*5 + 2*2 = 14; avg = 14/4 = 3.5
    result = metrics.calculate_empirical_risk(y_true, y_pred, cost_matrix)
    assert result == pytest.approx(3.5)


# Vectorized binary confusion counts


def test_binary_confusion_counts_vectorized_basic():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.2, 0.6, 0.4, 0.8])
    thresholds = np.array([0.3, 0.5, 0.7])

    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        y_true, probs, thresholds
    )

    # t=0.3: preds=[0,1,1,1] → TP=2, FP=1, FN=0, TN=1
    assert (tp[0], fp[0], fn[0], tn[0]) == (2, 1, 0, 1)
    # t=0.5: preds=[0,1,0,1] → TP=2, FP=1, FN=0, TN=1  (0.6>=0.5)
    # Wait: 0.2<0.5→0, 0.6>=0.5→1, 0.4<0.5→0, 0.8>=0.5→1
    assert (tp[1], fp[1], fn[1], tn[1]) == (1, 1, 1, 1)
    # t=0.7: preds=[0,0,0,1] → TP=1, FP=0, FN=1, TN=2
    assert (tp[2], fp[2], fn[2], tn[2]) == (1, 0, 1, 2)


def test_binary_confusion_counts_all_positive():
    y_true = np.array([0, 1])
    probs = np.array([0.9, 0.9])
    thresholds = np.array([0.5])
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        y_true, probs, thresholds
    )
    assert (tp[0], fp[0], fn[0], tn[0]) == (1, 1, 0, 0)


def test_binary_confusion_counts_all_negative():
    y_true = np.array([0, 1])
    probs = np.array([0.1, 0.1])
    thresholds = np.array([0.5])
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        y_true, probs, thresholds
    )
    assert (tp[0], fp[0], fn[0], tn[0]) == (0, 0, 1, 1)


# Vectorized metric scores vs sklearn

_Y_TRUE = np.array([0, 0, 0, 1, 1, 1, 0, 1])
_PROBS = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9, 0.8, 0.5])
_THRESHOLDS = np.array([0.2, 0.5, 0.8])


def _make_preds(probs, threshold):
    return (probs >= threshold).astype(int)


def test_vectorized_f1_matches_sklearn():
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        _Y_TRUE, _PROBS, _THRESHOLDS
    )
    scores = metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "f1")
    for i, t in enumerate(_THRESHOLDS):
        expected = sklearn_metrics.f1_score(
            _Y_TRUE, _make_preds(_PROBS, t), zero_division=0
        )
        assert scores[i] == pytest.approx(expected, abs=1e-10)


def test_vectorized_f_beta_matches_sklearn():
    for beta in [0.5, 2.0]:
        tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
            _Y_TRUE, _PROBS, _THRESHOLDS
        )
        scores = metrics.vectorized_scores_from_counts(
            tp, fp, fn, tn, "f_beta", beta=beta
        )
        for i, t in enumerate(_THRESHOLDS):
            expected = sklearn_metrics.fbeta_score(
                _Y_TRUE, _make_preds(_PROBS, t), beta=beta, zero_division=0
            )
            assert scores[i] == pytest.approx(expected, abs=1e-10)


def test_vectorized_mcc_matches_sklearn():
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        _Y_TRUE, _PROBS, _THRESHOLDS
    )
    scores = metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "mcc")
    for i, t in enumerate(_THRESHOLDS):
        expected = sklearn_metrics.matthews_corrcoef(
            _Y_TRUE, _make_preds(_PROBS, t)
        )
        assert scores[i] == pytest.approx(expected, abs=1e-10)


def test_vectorized_balanced_accuracy_matches_sklearn():
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        _Y_TRUE, _PROBS, _THRESHOLDS
    )
    scores = metrics.vectorized_scores_from_counts(
        tp, fp, fn, tn, "balanced_accuracy"
    )
    for i, t in enumerate(_THRESHOLDS):
        expected = sklearn_metrics.balanced_accuracy_score(
            _Y_TRUE, _make_preds(_PROBS, t)
        )
        assert scores[i] == pytest.approx(expected, abs=1e-10)


def test_vectorized_balanced_error_matches_one_minus_balanced_accuracy():
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        _Y_TRUE, _PROBS, _THRESHOLDS
    )
    ba_scores = metrics.vectorized_scores_from_counts(
        tp, fp, fn, tn, "balanced_accuracy"
    )
    be_scores = metrics.vectorized_scores_from_counts(
        tp, fp, fn, tn, "balanced_error"
    )
    for i in range(len(_THRESHOLDS)):
        assert be_scores[i] == pytest.approx(1.0 - ba_scores[i], abs=1e-10)


def test_vectorized_gmean_matches_manual():
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        _Y_TRUE, _PROBS, _THRESHOLDS
    )
    scores = metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "gmean")
    for i, t in enumerate(_THRESHOLDS):
        y_pred = _make_preds(_PROBS, t)
        tp_s = ((y_pred == 1) & (_Y_TRUE == 1)).sum()
        fn_s = ((y_pred == 0) & (_Y_TRUE == 1)).sum()
        fp_s = ((y_pred == 1) & (_Y_TRUE == 0)).sum()
        tn_s = ((y_pred == 0) & (_Y_TRUE == 0)).sum()
        tpr = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        tnr = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
        expected = np.sqrt(tpr * tnr)
        assert scores[i] == pytest.approx(expected, abs=1e-10)


def test_vectorized_cost_scores_matches_empirical_risk():
    cost_matrix = np.array([[0.0, 1.0], [10.0, 0.0]])
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        _Y_TRUE, _PROBS, _THRESHOLDS
    )
    scores = metrics.vectorized_cost_scores(tp, fp, fn, tn, cost_matrix)
    for i, t in enumerate(_THRESHOLDS):
        expected = -metrics.calculate_empirical_risk(
            _Y_TRUE, _make_preds(_PROBS, t), cost_matrix
        )
        assert scores[i] == pytest.approx(expected, abs=1e-10)


def test_vectorized_scores_unknown_key_raises():
    tp = fp = fn = tn = np.array([1])
    with pytest.raises(ValueError, match="Unknown vectorized metric key"):
        metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "nonexistent")


def test_vectorized_scores_nan_counts_raises():
    """NaN in confusion counts must raise immediately."""
    tp = np.array([1.0, np.nan])
    fp = np.array([0.0, 1.0])
    fn = np.array([0.0, 0.0])
    tn = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="contain NaN"):
        metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "f1")


def test_mcc_zero_denominator_returns_zero():
    """MCC when all predictions are the same class (denominator=0)."""
    # All predicted positive, all actually positive: fp=0, tn=0
    # den = sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    #     = sqrt(2 * 2 * 0 * 0) = 0
    tp = np.array([2.0])
    fp = np.array([0.0])
    fn = np.array([0.0])
    tn = np.array([0.0])
    scores = metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "mcc")
    assert scores[0] == 0.0


def test_f1_zero_denominator_returns_zero():
    """F1 with tp=fp=fn=0 (all predicted negative, no positives)."""
    zeros = np.array([0.0])
    tn = np.array([5.0])
    scores = metrics.vectorized_scores_from_counts(
        zeros, zeros, zeros, tn, "f1"
    )
    assert scores[0] == 0.0


def test_f_beta_zero_denominator_returns_zero():
    """F_beta with tp=fp=fn=0 returns 0 for any beta."""
    zeros = np.array([0.0])
    tn = np.array([5.0])
    for beta in (0.5, 1.0, 2.0):
        scores = metrics.vectorized_scores_from_counts(
            zeros, zeros, zeros, tn, "f_beta", beta=beta
        )
        assert scores[0] == 0.0


def test_balanced_accuracy_zero_denominator_returns_zero():
    """Balanced accuracy when a class is absent (tp+fn=0 or tn+fp=0)."""
    # Only negatives in y_true: tp=0, fn=0 → tpr branch takes the zero path.
    tp = np.array([0.0])
    fp = np.array([1.0])
    fn = np.array([0.0])
    tn = np.array([4.0])
    scores = metrics.vectorized_scores_from_counts(
        tp, fp, fn, tn, "balanced_accuracy"
    )
    # tpr = 0 (no positives), tnr = 4/5 = 0.8
    assert scores[0] == pytest.approx(0.4)


def test_gmean_zero_denominator_returns_zero():
    """Gmean when a class is absent: sqrt(0 * anything) = 0."""
    tp = np.array([0.0])
    fp = np.array([1.0])
    fn = np.array([0.0])
    tn = np.array([4.0])
    scores = metrics.vectorized_scores_from_counts(tp, fp, fn, tn, "gmean")
    assert scores[0] == 0.0


# balanced_error_cost: data-dependent cost matrix equivalent to BER


class TestBalancedErrorCost:
    def test_uniform_binary_is_symmetric_cost(self):
        """50/50 binary prior reduces to the standard symmetric cost
        matrix; balanced error then equals plain misclassification."""
        y = np.array([0, 1] * 100)
        M = metrics.balanced_error_cost(y)
        np.testing.assert_allclose(M, [[0.0, 1.0], [1.0, 0.0]])

    def test_imbalanced_binary_matches_closed_form(self):
        """For pi=(0.75, 0.25), cost(y, a) = 1/(2 pi(y)) gives
        [[0, 2/3], [2, 0]]."""
        y = np.array([0, 0, 0, 1])
        M = metrics.balanced_error_cost(y)
        np.testing.assert_allclose(M, [[0.0, 2 / 3], [2.0, 0.0]])

    def test_diagonal_is_zero(self):
        """No cost on correct predictions, regardless of priors."""
        y = np.array([0, 0, 1, 2, 2, 2])
        M = metrics.balanced_error_cost(y)
        np.testing.assert_allclose(np.diag(M), 0.0)

    def test_uniform_multiclass_matches_one_off_diagonal(self):
        """For uniform K-class priors, cost(y, a) = 1 off-diagonal."""
        y = np.array([0, 1, 2, 3] * 50)
        M = metrics.balanced_error_cost(y)
        expected = 1.0 - np.eye(4)
        np.testing.assert_allclose(M, expected)

    def test_off_diagonal_within_class_is_constant(self):
        """Per row y, every wrong prediction costs the same 1/(K pi(y))."""
        y = np.array([0, 0, 0, 1, 1, 2])  # priors [3/6, 2/6, 1/6]
        K = 3
        priors = np.array([3, 2, 1]) / 6
        M = metrics.balanced_error_cost(y)
        for i in range(K):
            row_off_diag = np.delete(M[i, :], i)
            np.testing.assert_allclose(
                row_off_diag, 1.0 / (K * priors[i]), atol=1e-12
            )

    def test_absent_class_yields_zero_row(self):
        """A class with no training samples gets a zero row: the matrix
        cannot charge cost when ``pi(y) = 0``."""
        y = np.array([0, 0, 2, 2, 2])  # class 1 absent
        M = metrics.balanced_error_cost(y)
        assert M.shape == (3, 3)
        np.testing.assert_allclose(M[1, :], 0.0)

    @pytest.mark.parametrize("p_pos", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_optimal_threshold_equals_positive_prior(self, p_pos):
        """For balanced-error cost in binary, the analytical optimal
        threshold equals ``P(y=1)``: the cost-matrix layout makes
        ``p* = c_01 / (c_01 + c_10) = (0.5/pi_0) / (0.5/pi_0 + 0.5/pi_1)
        = pi_1``."""
        n = 1000
        n_pos = round(p_pos * n)
        y = np.array([0] * (n - n_pos) + [1] * n_pos)
        M = metrics.balanced_error_cost(y)
        assert metrics.calculate_optimal_threshold(M) == pytest.approx(
            p_pos, abs=1e-9
        )

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_empirical_risk_matches_balanced_error_binary(self, seed):
        """Headline equivalence: on matched train/test priors, the cost
        matrix's empirical risk equals ``1 - balanced_accuracy_score``."""
        rng = np.random.default_rng(seed)
        y_true = rng.choice([0, 1], size=2_000, p=[0.7, 0.3])
        y_pred = rng.choice([0, 1], size=2_000, p=[0.6, 0.4])
        M = metrics.balanced_error_cost(y_true)
        er = metrics.calculate_empirical_risk(y_true, y_pred, M)
        ber = 1.0 - sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
        assert er == pytest.approx(ber, abs=1e-12)

    @pytest.mark.parametrize("K", [3, 4, 6])
    def test_empirical_risk_matches_balanced_error_multiclass(self, K):
        """Same equivalence for multiclass: macro-balanced error rate."""
        rng = np.random.default_rng(13 * K)
        y_true = rng.integers(0, K, size=3_000)
        y_pred = rng.integers(0, K, size=3_000)
        M = metrics.balanced_error_cost(y_true)
        er = metrics.calculate_empirical_risk(y_true, y_pred, M)
        ber = 1.0 - sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
        assert er == pytest.approx(ber, abs=1e-12)

    def test_perfect_prediction_zero_risk(self):
        """If predictions match truth everywhere, empirical risk is 0."""
        y = np.array([0, 0, 1, 2, 2, 2])
        M = metrics.balanced_error_cost(y)
        assert metrics.calculate_empirical_risk(y, y, M) == 0.0

    def test_constant_wrong_prediction_recovers_imbalance(self):
        """Always predicting class 0 in a 75/25 binary problem charges
        the cost on every positive: cost ~ 0.25 * 2 = 0.5 = BER."""
        y = np.array([0, 0, 0, 1])
        y_pred = np.zeros_like(y)
        M = metrics.balanced_error_cost(y)
        er = metrics.calculate_empirical_risk(y, y_pred, M)
        assert er == pytest.approx(0.5)


# DataDependentCostBuilder: dataclass wrapper + pipeline integration contract


class TestDataDependentCostBuilder:
    def test_is_dataclass(self):
        """The builder is a dataclass so it carries its build_fn as a
        field rather than a callable instance — this is what lets the
        pipeline distinguish it from CostMatrixFactory via isinstance."""
        assert dataclasses.is_dataclass(metrics.DataDependentCostBuilder)

    def test_not_callable(self):
        """Crucial contract: the builder must NOT be a bare callable,
        otherwise the pipeline's ``callable(opt)`` branch (used for
        CostMatrixFactory) would incorrectly try to call it with
        ``n_classes``. The pipeline must reach the ``isinstance``
        branch first."""
        assert not callable(metrics.BALANCED_ERROR_COST)

    def test_not_metric_objective(self):
        """The builder must not be confused with a MetricObjective —
        AnalyticalThreshold raises for MetricObjective but accepts
        materialized cost matrices."""
        assert not isinstance(
            metrics.BALANCED_ERROR_COST, metrics.MetricObjective
        )

    def test_carries_user_build_fn(self):
        """The build_fn field stores the supplied callable."""
        called_with: list[np.ndarray] = []

        def fn(y: np.ndarray) -> np.ndarray:
            called_with.append(np.asarray(y))
            return np.array([[0.0, 5.0], [7.0, 0.0]])

        builder = metrics.DataDependentCostBuilder(build_fn=fn)
        out = builder.build_fn(np.array([0, 1, 1]))
        assert len(called_with) == 1
        np.testing.assert_array_equal(called_with[0], [0, 1, 1])
        np.testing.assert_allclose(out, [[0.0, 5.0], [7.0, 0.0]])

    def test_balanced_error_cost_constant_is_a_builder(self):
        """The exported convenience constant is the right type."""
        assert isinstance(
            metrics.BALANCED_ERROR_COST, metrics.DataDependentCostBuilder
        )

    def test_balanced_error_cost_constant_materializes_correctly(self):
        """Calling its build_fn matches the standalone helper."""
        y = np.array([0, 0, 0, 0, 1])  # 80/20 prior
        np.testing.assert_allclose(
            metrics.BALANCED_ERROR_COST.build_fn(y),
            metrics.balanced_error_cost(y),
        )

    def test_in_optimization_objective_union(self):
        """The type union exposes the new builder so callers can type
        their objectives Mapping[str, OptimizationObjective] without
        the new entry being a Union widening surprise."""
        args = typing.get_args(metrics.OptimizationObjective)
        assert metrics.DataDependentCostBuilder in args
