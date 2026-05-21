"""Multiclass Decision Strategies."""

import collections
import math
from typing import Optional, Self, Tuple

import numpy as np
import numpy.typing as npt
from sklearn import calibration as sklearn_calibration
from sklearn import model_selection
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator

from threshold_opt import base, calibration, cv, metrics, optimization


def _safe_n_splits(y: np.ndarray, max_splits: int = 5) -> int:
    """Pick an n_splits that StratifiedKFold can honour on *y*.

    Thin alias for :func:`threshold_opt.cv.safe_n_splits`. Returns
    ``min(max_splits, smallest_class_count)`` and raises
    :class:`cv.InsufficientCVData` (a ``ValueError`` subclass) when the
    smallest class has fewer than 2 members -- the old ``max(2, ...)``
    clamp returned 2 there and then crashed deep inside sklearn.
    """
    return cv.safe_n_splits(y, max_splits)


def _build_cv_objective_fn(
    strategy: "MulticlassDecisionStrategy",
    X: npt.ArrayLike,
    y: np.ndarray,
    optimization_objective: metrics.OptimizationObjective,
    predict_from_probs,
):
    """Build an ``objective_fn(params)`` for CV or single-split tuning.

    ``predict_from_probs(probs, params) -> preds`` is the only per-strategy
    callable. Higher returned scores are better, matching
    ``metrics.get_objective_score``. When ``strategy.use_cv`` is True,
    K models are refit on each training fold and scores are averaged
    across folds.

    Fold predictions cache on ``strategy._fold_results`` so the K refits
    run once per strategy instance regardless of how many objectives
    the pipeline evaluates, and so the pipeline can hoist the refits
    further by pre-populating the slot before the first ``fit`` call.
    """
    y = np.asarray(y)

    if strategy.use_cv:
        if strategy._fold_results is None:
            kf = model_selection.StratifiedKFold(
                n_splits=cv.safe_n_splits(y, strategy.cv_folds),
                shuffle=True,
                random_state=base.CV_RANDOM_STATE,
            )
            fold_results: list[base._CVFoldPredictions] = []
            for train_idx, val_idx in kf.split(X, y):
                fold_model = clone(strategy.model)
                fold_model.fit(X[train_idx], y[train_idx])
                fold_results.append(
                    base._CVFoldPredictions(
                        y=y[val_idx],
                        probs=fold_model.predict_proba(X[val_idx]),
                    )
                )
            strategy._fold_results = fold_results

        def objective_fn(params):
            fold_scores = [
                metrics.get_objective_score(
                    fold.y,
                    predict_from_probs(fold.probs, params),
                    optimization_objective,
                )
                for fold in strategy._fold_results
            ]
            return float(np.mean(fold_scores))

        return objective_fn

    probs = strategy.model.predict_proba(X)

    def objective_fn(params):
        return metrics.get_objective_score(
            y, predict_from_probs(probs, params), optimization_objective
        )

    return objective_fn


def _assign_nearest_centroid(
    points: npt.NDArray[np.float64],
    centroids: npt.NDArray[np.float64],
) -> npt.NDArray[np.int_]:
    """Assigns each row of `points` to its nearest row of `centroids`.

    Squared-Euclidean argmin via the GEMM expansion
    ``||p - c||^2 = ||p||^2 - 2 p.c^T + ||c||^2``. This routes the
    O(N*K^2) work through BLAS instead of a Python-level ``(N, K, K)``
    broadcast: ~3x faster at K=10 and ~24x at K=100. Decisive because
    VoronoiTuning's random search invokes this once per sampled
    configuration, so it dominated the multiclass campaign.

    Algebraically identical to the direct ``sum((p - c)^2)`` form. The
    argmin is empirically bit-identical to the direct form across K up
    to 100 on simplex data (zero disagreements over 1e4+ rows); the only
    theoretically possible divergence is a sub-1e-9 near-tie, which is
    still a valid nearest centroid. The ``||p||^2`` term does not affect
    the argmin but is kept so the returned distances equal the true
    squared distance. Guarded by
    ``test_assign_nearest_centroid_is_valid_argmin`` (note the
    ``einsum("ij,ij->i", ...)`` axis: the centroid norm is per-centroid,
    i.e. summed over features -- a ``->j`` here is a silent bug because
    the centroid array is square).
    """
    points = np.asarray(points, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    p_sq = np.einsum("ij,ij->i", points, points)[:, None]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    sq_dist = p_sq - 2.0 * (points @ centroids.T) + c_sq
    return np.argmin(sq_dist, axis=1).astype(np.intp)


class MulticlassDecisionStrategy(base.DecisionStrategy):
    """Abstract base class for multiclass decision strategies."""

    def _validate_fit_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> None:
        y = np.asarray(y)
        if y.size == 0:
            # The pipeline passes empty arrays to strategies that neither
            # need a validation set nor self-tune with CV (Argmax and
            # Analytical). Label validation is vacuous and the n_classes
            # probe via predict_proba(X[:1]) would raise on empty X.
            return

        if not np.issubdtype(y.dtype, np.integer):
            # Accept integer-valued floats (OpenML hands us 0.0, 1.0, 2.0
            # for some datasets). Reject genuine non-integers like 0.5.
            y_int = y.astype(int)
            if not np.array_equal(y, y_int):
                raise ValueError(
                    f"Labels must be integer-valued. Got {y.dtype}"
                    f" with non-integer values."
                )
            y = y_int

        if np.any(y < 0):
            raise ValueError("Labels must be non-negative.")

        if isinstance(optimization_objective, np.ndarray):
            n_classes = optimization_objective.shape[0]
        else:
            n_classes = self.model.predict_proba(np.asarray(X)[:1]).shape[1]

        if np.any(y >= n_classes):
            raise ValueError(
                f"Labels must be in [0, {n_classes - 1}]."
                f" Got max label {y.max()}."
            )


class ArgmaxMulticlass(MulticlassDecisionStrategy):
    """Standard strategy that predicts the class with the highest
    probability."""

    @property
    def needs_validation_set(self) -> bool:
        return False

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        probs = self.model.predict_proba(X)
        return np.argmax(probs, axis=1)


class _CalibratedArgmaxMulticlass(ArgmaxMulticlass):
    """Fits sklearn calibration then uses argmax.

    Subclasses set ``_calibration_method`` to choose the sklearn backend.
    """

    _calibration_method: str = ""

    @property
    def needs_validation_set(self) -> bool:
        return True

    def __init__(self, model):
        super().__init__(model)
        self.calibrated_model: Optional[
            sklearn_calibration.CalibratedClassifierCV
        ] = None

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        if self.calibrated_model is None:
            y_arr = np.asarray(y)
            n_splits = _safe_n_splits(y_arr)
            cv = model_selection.StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=base.CV_RANDOM_STATE,
            )
            self.calibrated_model = sklearn_calibration.CalibratedClassifierCV(
                estimator=FrozenEstimator(self.model),
                method=self._calibration_method,
                cv=cv,
            )
            self.calibrated_model.fit(X, y_arr)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.calibrated_model is None:
            raise ValueError("Strategy not fit.")
        probs = self.calibrated_model.predict_proba(X)
        return np.argmax(probs, axis=1)


class IsotonicArgmaxMulticlass(_CalibratedArgmaxMulticlass):
    """Applies Isotonic Calibration then uses Argmax."""

    _calibration_method = "isotonic"


class AnalyticalMulticlass(MulticlassDecisionStrategy):
    """Strategy that minimizes the expected risk given a cost matrix."""

    def __init__(self, model):
        super().__init__(model)
        self.cost_matrix: Optional[np.ndarray] = None

    @property
    def needs_validation_set(self) -> bool:
        return False

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        if isinstance(optimization_objective, metrics.MetricObjective):
            raise ValueError("AnalyticalMulticlass requires a cost matrix.")

        self.cost_matrix = np.asarray(optimization_objective)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.cost_matrix is None:
            raise ValueError("Strategy not fit. Cost matrix missing.")

        probs = self.model.predict_proba(X)
        risks = probs @ self.cost_matrix

        return np.argmin(risks, axis=1)


class _CalibratedAnalyticalMulticlass(AnalyticalMulticlass):
    """Fits sklearn calibration then minimizes expected risk.

    The analytical-rule counterpart of ``_CalibratedArgmaxMulticlass``:
    subclasses set ``_calibration_method`` to choose the sklearn
    backend, calibration uses the same robust ``_safe_n_splits``
    StratifiedKFold, and the calibrated posteriors feed the cost-matrix
    argmin instead of a plain argmax.
    """

    _calibration_method: str = ""

    @property
    def needs_validation_set(self) -> bool:
        return True

    def __init__(self, model):
        super().__init__(model)
        self.calibrated_model: Optional[
            sklearn_calibration.CalibratedClassifierCV
        ] = None

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        # AnalyticalMulticlass._fit rejects a MetricObjective and stores
        # the cost matrix; run it first so a misuse fails before any
        # (wasted) calibrator fit.
        super()._fit(X, y, optimization_objective)
        if self.calibrated_model is None:
            y_arr = np.asarray(y)
            n_splits = _safe_n_splits(y_arr)
            cv = model_selection.StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=base.CV_RANDOM_STATE,
            )
            self.calibrated_model = sklearn_calibration.CalibratedClassifierCV(
                estimator=FrozenEstimator(self.model),
                method=self._calibration_method,
                cv=cv,
            )
            self.calibrated_model.fit(X, y_arr)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.cost_matrix is None or self.calibrated_model is None:
            raise ValueError("Strategy not fit.")

        probs = self.calibrated_model.predict_proba(X)
        n_classes_total = self.cost_matrix.shape[0]
        if probs.shape[1] != n_classes_total:
            # The calibrator saw only a subset of classes — e.g. a
            # stratified validation split that happens to drop a rare
            # class. ``probs`` is then narrower than ``cost_matrix`` and
            # the matmul below would raise. Pad with zero columns in the
            # slots for missing classes so the (N, K) x (K, K) shape is
            # restored; zero posterior is a reasonable fallback for a
            # class the calibrator had no data on.
            full = np.zeros((probs.shape[0], n_classes_total))
            cal_classes = np.asarray(self.calibrated_model.classes_, dtype=int)
            full[:, cal_classes] = probs
            probs = full

        risks = probs @ self.cost_matrix

        return np.argmin(risks, axis=1)


class IsotonicAnalyticalMulticlass(_CalibratedAnalyticalMulticlass):
    """Applies Isotonic Calibration then minimizes expected risk."""

    _calibration_method = "isotonic"


class PlattScalingAnalyticalMulticlass(_CalibratedAnalyticalMulticlass):
    """Applies Platt Scaling (Sigmoid Calibration) then minimizes
    expected risk.

    Reference: Platt, J. (1999). "Probabilistic Outputs for Support
    Vector Machines."
    """

    _calibration_method = "sigmoid"


class TemperatureScalingAnalyticalMulticlass(_CalibratedAnalyticalMulticlass):
    """Applies Temperature Scaling then minimizes expected risk.

    Learns a single scalar T and applies softmax(logits / T).

    Reference: Guo, C. et al. (2017). "On Calibration of Modern Neural
    Networks." ICML 2017.
    """

    _calibration_method = "temperature"


class PlattScalingArgmaxMulticlass(_CalibratedArgmaxMulticlass):
    """Applies Platt Scaling (Sigmoid Calibration) then uses Argmax.

    Reference: Platt, J. (1999). "Probabilistic Outputs for Support
    Vector Machines."
    """

    _calibration_method = "sigmoid"


class TemperatureScalingArgmaxMulticlass(_CalibratedArgmaxMulticlass):
    """Applies Temperature Scaling then uses Argmax.

    Learns a single scalar T and applies softmax(logits / T).

    Reference: Guo, C. et al. (2017). "On Calibration of Modern Neural
    Networks." ICML 2017.
    """

    _calibration_method = "temperature"


class DirichletCalibrationMulticlass(MulticlassDecisionStrategy):
    """Applies Dirichlet Calibration then uses Argmax.

    Learns a full linear mapping in log-probability space, preserving
    the joint structure of the probability vector on the simplex.

    Reference: Kull, M. et al. (2019). "Beyond temperature scaling:
    Obtaining well-calibrated multiclass probabilities with Dirichlet
    calibration." NeurIPS 2019.
    """

    @property
    def needs_validation_set(self) -> bool:
        return True

    def __init__(
        self,
        model,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
    ):
        super().__init__(model)
        self.calibrator: Optional[calibration.DirichletCalibrator] = None
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        if self.calibrator is None:
            probs = self.model.predict_proba(X)
            self.calibrator = calibration.DirichletCalibrator(
                reg_lambda=self.reg_lambda, reg_mu=self.reg_mu
            )
            self.calibrator.fit(probs, np.asarray(y))
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.calibrator is None:
            raise ValueError("Strategy not fit.")
        probs = self.model.predict_proba(X)
        calibrated_probs = self.calibrator.predict_proba(probs)
        return np.argmax(calibrated_probs, axis=1)


class DirichletCalibrationAnalyticalMulticlass(AnalyticalMulticlass):
    """Applies Dirichlet Calibration then minimizes expected risk.

    The analytical-rule counterpart of
    ``DirichletCalibrationMulticlass``. Like the argmax-side variant it
    uses ``calibration.DirichletCalibrator`` rather than sklearn's
    ``CalibratedClassifierCV``; the calibrator is fed the full-width
    base posteriors so the calibrated output is always K-wide and no
    missing-class padding is needed before the cost-matrix argmin.

    Reference: Kull, M. et al. (2019). "Beyond temperature scaling:
    Obtaining well-calibrated multiclass probabilities with Dirichlet
    calibration." NeurIPS 2019.
    """

    @property
    def needs_validation_set(self) -> bool:
        return True

    def __init__(
        self,
        model,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
    ):
        super().__init__(model)
        self.calibrator: Optional[calibration.DirichletCalibrator] = None
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        # AnalyticalMulticlass._fit rejects a MetricObjective and stores
        # the cost matrix; run it first so a misuse fails before any
        # (wasted) calibrator fit.
        super()._fit(X, y, optimization_objective)
        if self.calibrator is None:
            probs = self.model.predict_proba(X)
            self.calibrator = calibration.DirichletCalibrator(
                reg_lambda=self.reg_lambda, reg_mu=self.reg_mu
            )
            self.calibrator.fit(probs, np.asarray(y))
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.cost_matrix is None or self.calibrator is None:
            raise ValueError("Strategy not fit.")
        probs = self.model.predict_proba(X)
        calibrated_probs = self.calibrator.predict_proba(probs)
        risks = calibrated_probs @ self.cost_matrix
        return np.argmin(risks, axis=1)


class ThresholdRatioMulticlass(MulticlassDecisionStrategy):
    """Strategy that predicts argmax(probability / threshold) for each
    class."""

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or optimization.RandomSearchOptimizer(
            bounds=(0.01, 1.5)
        )
        self.thresholds: Optional[np.ndarray] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        n_classes = self.model.predict_proba(X[:1]).shape[1]

        def predict_from_probs(probs, params):
            return np.argmax(probs / params, axis=1)

        objective_fn = _build_cv_objective_fn(
            self, X, y, optimization_objective, predict_from_probs
        )

        initial_params = np.ones(n_classes)
        initial_score = objective_fn(initial_params)

        best_params, _ = self.optimizer.optimize(
            objective_fn,
            n_params=n_classes,
            current_best_params=initial_params,
            current_best_score=initial_score,
        )

        self.thresholds = best_params
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.thresholds is None:
            raise ValueError("Strategy not fit. Thresholds missing.")

        probs = self.model.predict_proba(X)
        return np.argmax(probs / self.thresholds, axis=1)


class VoronoiCentroidOptimizer(optimization.RandomSearchOptimizer):
    """Optimizer that samples valid centroids on the probability simplex."""

    def __init__(
        self,
        n_iter: int = 100,
        seed: Optional[int] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            n_iter=n_iter, bounds=(0.0, 1.0), seed=seed, n_jobs=n_jobs
        )

    _MAX_RETRIES = 500

    def _sample_params(
        self, rng: np.random.Generator, n_params: int
    ) -> npt.NDArray[np.float64]:
        n_dim = math.isqrt(n_params)
        if n_dim * n_dim != n_params:
            raise ValueError(f"n_params {n_params} must be a perfect square.")

        # Sequential d² sampler. For each class i in order:
        #   1. draw c ~ Dir(1) and swap c[argmax], c[i] so the max sits at i;
        #   2. accept only if c beats every already-placed c_k on e_i
        #      (d²(e_i, c) < d²(e_i, c_k)) AND fails to beat c_k on e_k
        #      (d²(e_k, c) > d²(e_k, c_k)).
        # Preserves both sampler invariants by construction:
        #   - bijective class→centroid labelling (trivially: one centroid
        #     per class by enumeration);
        #   - pure-class vertex → own class: argmin_j d²(e_k, c_j) = k.
        # Empirically the retry cap is never hit at K <= 26; a RuntimeError
        # on exhaustion is informative rather than a silent fallback.
        alpha = np.ones(n_dim)
        centroids = np.empty((n_dim, n_dim))

        def _dsq(vertex_idx: int, c: np.ndarray) -> float:
            return 1.0 - 2.0 * c[vertex_idx] + float(c @ c)

        for i in range(n_dim):
            for _attempt in range(self._MAX_RETRIES):
                c = rng.dirichlet(alpha)
                j = int(c.argmax())
                c[[i, j]] = c[[j, i]]
                accept = True
                for k in range(i):
                    if _dsq(k, c) <= _dsq(k, centroids[k]):
                        accept = False
                        break
                    if _dsq(i, c) >= _dsq(i, centroids[k]):
                        accept = False
                        break
                if accept:
                    centroids[i] = c
                    break
            else:
                raise RuntimeError(
                    f"VoronoiCentroidOptimizer sequential-d² sampler failed"
                    f" to place centroid {i} after {self._MAX_RETRIES}"
                    f" retries (K={n_dim})."
                )

        return centroids.flatten()


class VoronoiTuningMulticlass(MulticlassDecisionStrategy):
    """Strategy using Voronoi Tessellation on the probability simplex."""

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        n_iter: int = 100,
        n_jobs: int = 1,
        seed: Optional[int] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or VoronoiCentroidOptimizer(
            n_iter=n_iter, n_jobs=n_jobs, seed=seed
        )
        self.best_centroids: Optional[np.ndarray] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        n_classes = self.model.predict_proba(X[:1]).shape[1]

        def predict_from_probs(probs, flat_centroids):
            centroids = flat_centroids.reshape((n_classes, n_classes))
            return _assign_nearest_centroid(probs, centroids)

        objective_fn = _build_cv_objective_fn(
            self, X, y, optimization_objective, predict_from_probs
        )

        initial_centroids = np.identity(n_classes).flatten()
        initial_score = objective_fn(initial_centroids)

        best_flat_centroids, _ = self.optimizer.optimize(
            objective_fn,
            n_params=n_classes * n_classes,
            current_best_params=initial_centroids,
            current_best_score=initial_score,
        )

        self.best_centroids = best_flat_centroids.reshape(
            (n_classes, n_classes)
        )

        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.best_centroids is None:
            raise ValueError("Strategy not fit.")

        probs = self.model.predict_proba(X)
        return _assign_nearest_centroid(probs, self.best_centroids)


class SequentialSelectionMulticlass(MulticlassDecisionStrategy):
    """Applies per-class binary thresholds in a sequential order."""

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or optimization.RandomSearchOptimizer(
            bounds=(0.0, 1.0)
        )
        self.thresholds: Optional[np.ndarray] = None
        self.order: Optional[Tuple[int, ...]] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _order_predict(
        self,
        order: Tuple[int, ...],
        probs: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        conditions = np.zeros_like(probs, dtype=bool)

        for i, class_idx in enumerate(order):
            conditions[:, i] = probs[:, class_idx] >= thresholds[class_idx]

        default_predictions = np.argmax(probs, axis=1)

        return np.select(conditions.T, order, default=default_predictions)

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        y = np.asarray(y)
        n_classes = self.model.predict_proba(X[:1]).shape[1]

        counts = collections.Counter(y)
        self.order = tuple(
            sorted(range(n_classes), key=lambda k: counts.get(k, 0))
        )

        def predict_from_probs(probs, thresholds):
            return self._order_predict(self.order, probs, thresholds)

        objective_fn = _build_cv_objective_fn(
            self, X, y, optimization_objective, predict_from_probs
        )

        initial_params = np.full(n_classes, 0.5)
        initial_score = objective_fn(initial_params)

        best_params, _ = self.optimizer.optimize(
            objective_fn,
            n_params=n_classes,
            current_best_params=initial_params,
            current_best_score=initial_score,
        )

        self.thresholds = best_params
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.thresholds is None or self.order is None:
            raise ValueError("Strategy not fit.")

        probs = self.model.predict_proba(X)
        return self._order_predict(self.order, probs, self.thresholds)


class RiskPartitioningStrategy(MulticlassDecisionStrategy):
    """Minimizes expected risk by learning a partition matrix (alpha)."""

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        solver: str = "greedy",
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        if solver not in ("greedy", "random_search"):
            raise ValueError(
                f"Unknown solver '{solver}'. Use 'greedy' or 'random_search'."
            )
        if use_cv and solver == "greedy":
            raise ValueError(
                'use_cv=True is only supported with solver="random_search"'
                " for RiskPartitioningStrategy."
            )
        super().__init__(model)
        self.optimizer = optimizer
        self.solver = solver
        self.biases: Optional[np.ndarray] = None
        self.cost_matrix: Optional[np.ndarray] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        if not isinstance(optimization_objective, np.ndarray):
            raise ValueError(
                "RiskPartitioningStrategy requires a cost matrix."
            )

        self.cost_matrix = np.asarray(optimization_objective)
        y = np.asarray(y)

        if self.solver == "greedy":
            probs = self.model.predict_proba(X)
            base_risks = probs @ self.cost_matrix
            self.biases = optimization.solve_obrien_changepoints(
                base_risks, y, self.cost_matrix
            )
        else:
            self.biases = self._solve_random_search(X, y)

        return self

    def _solve_random_search(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_classes = self.model.predict_proba(X[:1]).shape[1]
        optimizer = self.optimizer or optimization.RandomSearchOptimizer(
            bounds=(-1.0, 1.0)
        )

        cost_matrix = self.cost_matrix

        def predict_from_probs(probs, params):
            full_biases = np.concatenate(([0.0], params))
            return np.argmin(probs @ cost_matrix + full_biases, axis=1)

        objective_fn = _build_cv_objective_fn(
            self, X, y, cost_matrix, predict_from_probs
        )

        initial_params = np.zeros(n_classes - 1)

        best_params, _ = optimizer.optimize(
            objective_fn,
            n_params=n_classes - 1,
            current_best_params=initial_params,
            current_best_score=objective_fn(initial_params),
        )

        return np.concatenate(([0.0], best_params))

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.biases is None or self.cost_matrix is None:
            raise ValueError("Strategy not fit.")

        probs = self.model.predict_proba(X)
        risks = probs @ self.cost_matrix
        return np.argmin(risks + self.biases, axis=1)


class LinearScalingMulticlass(MulticlassDecisionStrategy):
    """Minimizes expected risk under a per-truth-class scaled cost matrix.

    Learns a non-negative weight vector beta and predicts under the
    row-scaled cost matrix A_{ij} = C_{ij} * beta_i (equivalently
    A = diag(beta) @ C). This is O'Brien's linear scaling, the
    "Adjusted Priors" reweighting of Lachiche and Flach (2003). At
    beta = 1 it reduces to AnalyticalMulticlass (the Bayes-optimal rule),
    which is the search warm start, so the optimizer can only improve on
    it. The argmin is invariant to scaling all beta by a positive
    constant; the K-parameter search carries one redundant scale degree
    of freedom, which is harmless for the gradient-free optimizer given
    the analytical warm start.

    Cost-matrix objectives only. A metrics.MetricObjective has no cost
    matrix to scale (and under 0/1 loss linear scaling coincides with
    ThresholdRatioMulticlass), so _fit raises in that case, mirroring
    RiskPartitioningStrategy.
    """

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or optimization.RandomSearchOptimizer(
            bounds=(0.0, 5.0)
        )
        self.scaling_factors: Optional[np.ndarray] = None
        self.cost_matrix: Optional[np.ndarray] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        if not isinstance(optimization_objective, np.ndarray):
            raise ValueError("LinearScalingMulticlass requires a cost matrix.")

        self.cost_matrix = np.asarray(optimization_objective, dtype=float)
        y = np.asarray(y)
        n_classes = self.cost_matrix.shape[0]
        cost_matrix = self.cost_matrix

        def predict_from_probs(probs, beta):
            return np.argmin(
                probs @ (np.asarray(beta)[:, None] * cost_matrix), axis=1
            )

        objective_fn = _build_cv_objective_fn(
            self, X, y, optimization_objective, predict_from_probs
        )

        initial_params = np.ones(n_classes)
        initial_score = objective_fn(initial_params)

        best_params, _ = self.optimizer.optimize(
            objective_fn,
            n_params=n_classes,
            current_best_params=initial_params,
            current_best_score=initial_score,
        )

        self.scaling_factors = best_params
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.scaling_factors is None or self.cost_matrix is None:
            raise ValueError("Strategy not fit.")

        probs = self.model.predict_proba(X)
        return np.argmin(
            probs @ (self.scaling_factors[:, None] * self.cost_matrix),
            axis=1,
        )


class FullMatrixPartitioningStrategy(MulticlassDecisionStrategy):
    """Learns the unconstrained K x K partition matrix A (O'Brien 2008).

    Decision rule: argmin_i (p^T A)_i with A in R^{K x K}. Search variable
    is Delta = A - A_0 in R^{K x K}, anchored on A_0 = C for cost-matrix
    objectives (warm start at the analytical/Bayes-optimal rule) and on
    A_0 = J - I (the symmetric 0/1 cost matrix) for MetricObjective
    targets (warm start at argmax). Search is unconstrained; the matrix
    analogue of the Dominant Label Constraint (A_jj <= A_ji for all
    i != j) is intentionally not enforced.
    """

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or optimization.RandomSearchOptimizer(
            bounds=(-1.0, 1.0)
        )
        self.partition_matrix: Optional[np.ndarray] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        n_classes = self.model.predict_proba(np.asarray(X)[:1]).shape[1]

        if isinstance(optimization_objective, np.ndarray):
            anchor = np.asarray(optimization_objective, dtype=float)
        else:
            anchor = np.ones((n_classes, n_classes)) - np.eye(n_classes)

        def predict_from_probs(probs, params):
            A = anchor + np.asarray(params).reshape(n_classes, n_classes)
            return np.argmin(probs @ A, axis=1)

        objective_fn = _build_cv_objective_fn(
            self, X, y, optimization_objective, predict_from_probs
        )

        initial_params = np.zeros(n_classes * n_classes)
        initial_score = objective_fn(initial_params)

        best_params, _ = self.optimizer.optimize(
            objective_fn,
            n_params=n_classes * n_classes,
            current_best_params=initial_params,
            current_best_score=initial_score,
        )

        self.partition_matrix = anchor + np.asarray(best_params).reshape(
            n_classes, n_classes
        )
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.partition_matrix is None:
            raise ValueError("Strategy not fit. Partition matrix missing.")

        probs = self.model.predict_proba(X)
        return np.argmin(probs @ self.partition_matrix, axis=1)


class MultidimensionalThresholdStrategy(MulticlassDecisionStrategy):
    """Maximizes a score by learning multidimensional thresholds."""

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or optimization.RandomSearchOptimizer(
            bounds=(-1.0, 1.0)
        )
        self.biases: Optional[np.ndarray] = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self._fold_results: Optional[list[base._CVFoldPredictions]] = None

    @property
    def self_tunes_with_cv(self) -> bool:
        return self.use_cv

    @property
    def needs_validation_set(self) -> bool:
        return not self.use_cv

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        n_classes = self.model.predict_proba(X[:1]).shape[1]

        def predict_from_probs(probs, params):
            full_biases = np.concatenate(([0.0], params))
            return np.argmax(probs + full_biases, axis=1)

        objective_fn = _build_cv_objective_fn(
            self, X, y, optimization_objective, predict_from_probs
        )

        initial_params = np.zeros(n_classes - 1)
        initial_score = objective_fn(initial_params)

        best_params, _ = self.optimizer.optimize(
            objective_fn,
            n_params=n_classes - 1,
            current_best_params=initial_params,
            current_best_score=initial_score,
        )

        self.biases = np.concatenate(([0.0], best_params))
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int_]:
        if self.biases is None:
            raise ValueError("Strategy not fit. Biases missing.")

        probs = self.model.predict_proba(X)
        return np.argmax(probs + self.biases, axis=1)
