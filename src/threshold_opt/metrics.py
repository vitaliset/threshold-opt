"""Core mathematical functions for risk estimation and threshold derivation."""

import dataclasses
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

CostMatrix = npt.ArrayLike
CostMatrixFactory = Callable[[int], CostMatrix]


@dataclasses.dataclass
class MetricObjective:
    metric_fn: Callable[[npt.ArrayLike, npt.ArrayLike], float]
    higher_is_better: bool
    vectorized_key: str | None = None
    beta: float = 1.0


@dataclasses.dataclass
class DataDependentCostBuilder:
    """Cost matrix that depends on the training-set class priors.

    The pipeline materializes the builder once per task by calling
    ``build_fn(y_train)`` and then passes the resulting cost matrix to
    strategies just like any other ``CostMatrix`` ndarray. This lets a
    single objective definition adapt to each dataset's empirical
    class distribution without duplicating cost-matrix entries.
    """

    build_fn: Callable[[npt.NDArray], np.ndarray]


def balanced_error_cost(y: npt.ArrayLike) -> np.ndarray:
    """Cost matrix whose expected value equals macro balanced error.

    Macro balanced error rate equals
    ``(1/K) * sum_y (1 - recall_y) = E_{(X,Y)}[cost(Y, A(X))]`` for
    ``cost(y, a) = 1 / (K * pi(y))`` when ``a != y`` and 0 on the
    diagonal, where ``pi(y)`` is the marginal class prior. Strategies
    that minimize this empirical risk on a held-out set therefore
    minimize balanced error directly.

    Empty classes (no samples in ``y``) get cost 0; this avoids
    division by zero but means strategies cannot be punished for
    failing to predict classes absent at fit time.
    """
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    K = int(classes.max()) + 1 if len(classes) else 0
    priors = np.zeros(K)
    priors[classes.astype(int)] = counts / counts.sum()
    M = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        if priors[i] <= 0:
            continue
        M[i, :] = 1.0 / (K * priors[i])
        M[i, i] = 0.0
    return M


BALANCED_ERROR_COST = DataDependentCostBuilder(build_fn=balanced_error_cost)


OptimizationObjective = Union[
    CostMatrix, MetricObjective, CostMatrixFactory, DataDependentCostBuilder
]


def get_objective_score(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    optimization_objective: "OptimizationObjective",
) -> float:
    """Score a prediction under an OptimizationObjective.

    Higher is always better: metric signs are flipped to minimization
    form, and cost-matrix objectives return negative empirical risk.
    """
    if isinstance(optimization_objective, MetricObjective):
        sign = 1.0 if optimization_objective.higher_is_better else -1.0
        return sign * optimization_objective.metric_fn(y_true, y_pred)

    cost_matrix = np.asarray(optimization_objective)
    return -calculate_empirical_risk(y_true, y_pred, cost_matrix)


def calculate_optimal_threshold(cost_matrix: CostMatrix) -> float:
    """Derives the theoretical optimal threshold p* for a binary cost matrix.

    The optimal threshold is derived by minimizing the expected cost.
    The formula is:
        p* = (FP - TN) / ( (FP - TN) + (FN - TP) )
           = (C_01 - C_00) / ( (C_01 - C_00) + (C_10 - C_11) )

    Args:
        cost_matrix: 2x2 matrix [[TN_cost, FP_cost], [FN_cost, TP_cost]].
                     Note: This follows the standard confusion matrix
                     layout where
                       C_ij is the cost of predicting class j when truth is i.

    Returns:
        float: The probability threshold in [0, 1].

    Raises:
        ValueError: If the cost matrix is not 2x2.
    """
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.shape != (2, 2):
        raise ValueError(
            f"Binary threshold requires 2x2 matrix, got {cost_matrix.shape}"
        )

    c_00, c_01 = cost_matrix[0]
    c_10, c_11 = cost_matrix[1]

    numerator = c_01 - c_00
    denominator = (c_01 - c_00) + (c_10 - c_11)

    if np.isclose(denominator, 0):
        return 0.5

    return float(numerator / denominator)


def calculate_empirical_risk(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    cost_matrix: CostMatrix,
) -> float:
    """Computes average realized cost (Empirical Risk).

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        cost_matrix: The cost matrix where `cost_matrix[i, j]`
          represents the cost
          of predicting class `j` when the true class is `i`.

    Returns:
        The average empirical risk.

    Raises:
        ValueError: If the cost matrix is not square.
    """
    cost_matrix = np.asarray(cost_matrix)

    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("Cost matrix must be square.")

    y_true = _as_integer_labels(y_true, "y_true")
    y_pred = _as_integer_labels(y_pred, "y_pred")

    n = y_true.shape[0]
    if n == 0:
        return 0.0

    if y_pred.shape[0] != n:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {n} and {y_pred.shape[0]}."
        )

    n_classes = cost_matrix.shape[0]
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.min() < 0 or arr.max() >= n_classes:
            raise ValueError(
                f"{name} contains labels outside [0, {n_classes - 1}]."
            )

    total_cost = cost_matrix[y_true, y_pred].sum()
    return float(total_cost / n)


def _as_integer_labels(
    labels: npt.ArrayLike, name: str
) -> npt.NDArray[np.intp]:
    arr = np.asarray(labels)
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        return arr.astype(np.intp, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        cast = arr.astype(np.intp)
        if not np.array_equal(cast, arr):
            raise ValueError(
                f"{name} must be integer-valued, got non-integer floats."
            )
        return cast
    raise ValueError(f"{name} must be integer labels, got dtype {arr.dtype}.")


# ---------------------------------------------------------------------------
# Vectorized scoring for binary threshold grid search
# ---------------------------------------------------------------------------


def binary_confusion_counts_vectorized(
    y_true: npt.NDArray[np.int_],
    probs: npt.NDArray[np.float64],
    thresholds: npt.NDArray[np.float64],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute TP, FP, FN, TN for all thresholds simultaneously.

    Uses numpy broadcasting to avoid per-threshold Python loops.

    Args:
        y_true: shape (n_samples,), values in {0, 1}.
        probs: shape (n_samples,), predicted probability of class 1.
        thresholds: shape (n_thresholds,).

    Returns:
        (tp, fp, fn, tn) each of shape (n_thresholds,).
    """
    # pred_positive has shape (n_thresholds, n_samples)
    pred_positive = probs[np.newaxis, :] >= thresholds[:, np.newaxis]
    positives = y_true.astype(bool)
    negatives = ~positives

    tp = (pred_positive & positives).sum(axis=1)
    fp = (pred_positive & negatives).sum(axis=1)
    fn = (~pred_positive & positives).sum(axis=1)
    tn = (~pred_positive & negatives).sum(axis=1)
    return tp, fp, fn, tn


def vectorized_scores_from_counts(
    tp: npt.NDArray,
    fp: npt.NDArray,
    fn: npt.NDArray,
    tn: npt.NDArray,
    metric_key: str,
    *,
    beta: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Compute metric scores for all thresholds from confusion counts.

    All inputs have shape (n_thresholds,). Returns shape (n_thresholds,)
    holding the metric's natural value (e.g. F1 or balanced accuracy
    higher-is-better, balanced error lower-is-better); the caller is
    responsible for sign conventions via ``MetricObjective.higher_is_better``.
    """
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)
    tn = tn.astype(np.float64)

    if np.any(np.isnan(tp) | np.isnan(fp) | np.isnan(fn) | np.isnan(tn)):
        raise ValueError(
            "Confusion counts contain NaN values. Check input arrays for NaN."
        )

    if metric_key == "f1":
        num = 2.0 * tp
        den = 2.0 * tp + fp + fn
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    if metric_key == "f_beta":
        beta2 = beta**2
        num = (1.0 + beta2) * tp
        den = (1.0 + beta2) * tp + beta2 * fn + fp
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    if metric_key == "mcc":
        num = tp * tn - fp * fn
        product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        den = np.sqrt(np.maximum(product, 0.0))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    if metric_key == "balanced_accuracy":
        tpr = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
        )
        tnr = np.divide(
            tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0
        )
        return (tpr + tnr) / 2.0

    if metric_key == "balanced_error":
        # 1 - balanced accuracy = (FPR + FNR) / 2; the natural cost form
        # of BA under prior-weighted misclassification costs (the BER /
        # balanced error rate of Menon et al. 2013). Returned as-is so the
        # caller can negate via ``higher_is_better=False``.
        tpr = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
        )
        tnr = np.divide(
            tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0
        )
        return 1.0 - (tpr + tnr) / 2.0

    if metric_key == "gmean":
        tpr = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
        )
        tnr = np.divide(
            tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0
        )
        return np.sqrt(tpr * tnr)

    raise ValueError(f"Unknown vectorized metric key: {metric_key!r}")


def vectorized_cost_scores(
    tp: npt.NDArray,
    fp: npt.NDArray,
    fn: npt.NDArray,
    tn: npt.NDArray,
    cost_matrix: npt.NDArray,
) -> npt.NDArray[np.float64]:
    """Compute negative empirical risk for all thresholds at once.

    Returns scores where higher is better (negative cost).
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    n = (tp + fp + fn + tn).astype(np.float64)
    total_cost = (
        tn * cost_matrix[0, 0]
        + fp * cost_matrix[0, 1]
        + fn * cost_matrix[1, 0]
        + tp * cost_matrix[1, 1]
    )
    return -(total_cost / n)
