"""Decision Strategies implementing the inference-to-decision mapping."""

import warnings
from typing import Optional, Self

import numpy as np
import numpy.typing as npt
from sklearn import calibration as sklearn_calibration
from sklearn import model_selection
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator

from threshold_opt import base, calibration, cv, metrics, optimization


def _positive_class_probs(probs: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from a 1D or 2D ``predict_proba``
    output. Shared-fold caches (see pipeline._precompute_cv_fold_results)
    always store 2D probs; legacy callers may pass the already-extracted
    1D vector.
    """
    if probs.ndim == 1:
        return probs
    if probs.shape[1] >= 2:
        return probs[:, 1]
    # Degenerate single-class fold; probs[:, 0] is the only column. The
    # binary strategies treat that as "no positive signal" and fall back
    # to zero probabilities downstream.
    warnings.warn(
        "_positive_class_probs received a single-column predict_proba"
        " output and is returning zeros; the fold lost the positive class"
        " entirely and downstream threshold selection will be biased toward"
        " predicting only negatives.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.zeros(probs.shape[0])


def _score_all_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray,
    optimization_objective: metrics.OptimizationObjective,
) -> np.ndarray:
    """Score all thresholds at once using vectorized numpy operations.

    Returns an array of shape ``(len(grid),)`` where higher is better.
    Falls back to a per-threshold loop for :class:`MetricObjective` instances
    that do not provide a ``vectorized_key``.
    """
    tp, fp, fn, tn = metrics.binary_confusion_counts_vectorized(
        y_true, probs, grid
    )

    if isinstance(optimization_objective, metrics.MetricObjective):
        obj = optimization_objective
        if obj.vectorized_key is not None:
            scores = metrics.vectorized_scores_from_counts(
                tp, fp, fn, tn, obj.vectorized_key, beta=obj.beta
            )
            if not obj.higher_is_better:
                scores = -scores
            return scores
        return np.array(
            [
                metrics.get_objective_score(
                    y_true, (probs >= t).astype(int), optimization_objective
                )
                for t in grid
            ]
        )

    cost_matrix = np.asarray(optimization_objective)
    return metrics.vectorized_cost_scores(tp, fp, fn, tn, cost_matrix)


class BinaryDecisionStrategy(base.DecisionStrategy):
    """Abstract base class for threshold determination strategies."""

    def __init__(self, model):
        super().__init__(model)
        self._threshold: float | None = None

    def _validate_fit_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> None:
        y = np.asarray(y)
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                "Binary strategies require labels to be 0 or 1."
                f" Got {unique_labels}."
            )

    def _get_positive_proba(self, model, X: npt.ArrayLike) -> npt.ArrayLike:
        """Safely extracts probabilities for the positive class (1)."""
        probs = model.predict_proba(X)
        if probs.shape[1] >= 2:
            return probs[:, 1]

        if hasattr(model, "classes_"):
            if len(model.classes_) == 1 and model.classes_[0] == 1:
                return probs[:, 0]
        # Degenerate fit: model.classes_ does not include the positive
        # class, so the single column we have represents the negative
        # class. Returning zeros reports "no positive signal", which
        # collapses downstream threshold selection toward predicting only
        # negatives — surface that explicitly so callers can audit.
        warnings.warn(
            "_get_positive_proba: model has no positive-class column; "
            "returning zeros. Threshold selection will be biased toward "
            "predicting only negatives.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros(probs.shape[0])

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Predict class labels for X using the determined threshold."""
        probs = self._get_positive_proba(self.model, X)
        return (probs >= self.threshold).astype(int)

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ValueError(
                "Threshold not determined. Please call `fit` first."
            )
        return self._threshold


class ArgmaxThreshold(BinaryDecisionStrategy):
    """Argmax threshold derivation."""

    @property
    def needs_validation_set(self) -> bool:
        return False

    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        self._threshold = 0.5
        return self


class _CalibratedArgmaxThreshold(BinaryDecisionStrategy):
    """Fits sklearn calibration then thresholds at 0.5.

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
            self.calibrated_model = sklearn_calibration.CalibratedClassifierCV(
                estimator=FrozenEstimator(self.model),
                method=self._calibration_method,
            )
            self.calibrated_model.fit(X, y)
        self._threshold = 0.5
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        if self.calibrated_model is None:
            raise ValueError("Strategy not fit.")

        probs_calibrated = self._get_positive_proba(self.calibrated_model, X)
        return (probs_calibrated >= self.threshold).astype(int)


class _CalibratedAnalyticalThreshold(BinaryDecisionStrategy):
    """Fits sklearn calibration then uses the analytical optimal threshold."""

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
        if isinstance(optimization_objective, metrics.MetricObjective):
            raise ValueError(
                f"{type(self).__name__} requires a cost matrix for"
                " optimization, got MetricObjective."
            )
        if self.calibrated_model is None:
            self.calibrated_model = sklearn_calibration.CalibratedClassifierCV(
                estimator=FrozenEstimator(self.model),
                method=self._calibration_method,
            )
            self.calibrated_model.fit(X, y)
        self._threshold = metrics.calculate_optimal_threshold(
            optimization_objective
        )
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        if self.calibrated_model is None:
            raise ValueError("Strategy not fit.")

        probs_calibrated = self._get_positive_proba(self.calibrated_model, X)
        return (probs_calibrated >= self.threshold).astype(int)


class IsotonicArgmax(_CalibratedArgmaxThreshold):
    """Isotonic Regression + argmax (threshold 0.5)."""

    _calibration_method = "isotonic"


class PlattScalingArgmax(_CalibratedArgmaxThreshold):
    """Platt Scaling + argmax (threshold 0.5).

    Reference: Platt, J. (1999).
    """

    _calibration_method = "sigmoid"


class TemperatureScalingArgmax(_CalibratedArgmaxThreshold):
    """Temperature Scaling + argmax (threshold 0.5).

    Reference: Guo, C. et al. (2017). "On Calibration of Modern Neural
    Networks." ICML 2017.
    """

    _calibration_method = "temperature"


class IsotonicAnalytical(_CalibratedAnalyticalThreshold):
    """Isotonic Regression + analytical optimal threshold."""

    _calibration_method = "isotonic"


class PlattScalingAnalytical(_CalibratedAnalyticalThreshold):
    """Platt Scaling + analytical optimal threshold.

    Reference: Platt, J. (1999).
    """

    _calibration_method = "sigmoid"


class TemperatureScalingAnalytical(_CalibratedAnalyticalThreshold):
    """Temperature Scaling + analytical optimal threshold.

    Reference: Guo, C. et al. (2017). "On Calibration of Modern Neural
    Networks." ICML 2017.
    """

    _calibration_method = "temperature"


class AnalyticalThreshold(BinaryDecisionStrategy):
    """Analytical threshold derivation."""

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
            raise ValueError(
                "AnalyticalThreshold requires a cost matrix for optimization,"
                " got metrics.MetricObjective."
            )
        self._threshold = metrics.calculate_optimal_threshold(
            optimization_objective
        )
        return self


class EmpiricalThreshold(BinaryDecisionStrategy):
    """Post-hoc optimization via Optimizer on Validation set."""

    def __init__(
        self,
        model,
        optimizer: Optional[optimization.ParameterOptimizer] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.optimizer = optimizer or optimization.GridSearchOptimizer(
            bounds=(0.00, 1.00), steps=301
        )
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
        if self.use_cv:
            if not isinstance(
                self.optimizer, optimization.GridSearchOptimizer
            ):
                raise ValueError(
                    "CV tuning only supported with GridSearchOptimizer for"
                    " EmpiricalThreshold."
                )

            if self._fold_results is None:
                kf = model_selection.StratifiedKFold(
                    n_splits=cv.safe_n_splits(y, self.cv_folds),
                    shuffle=True,
                    random_state=base.CV_RANDOM_STATE,
                )

                self._fold_results = []
                for train_idx, val_idx in kf.split(X, y):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_v, y_v = X[val_idx], y[val_idx]
                    model = clone(self.model)
                    model.fit(X_tr, y_tr)
                    self._fold_results.append(
                        base._CVFoldPredictions(
                            y=y_v, probs=model.predict_proba(X_v)
                        )
                    )

            grid = np.linspace(
                self.optimizer.bounds[0],
                self.optimizer.bounds[1],
                self.optimizer.steps,
            )

            scores_per_t = np.zeros(len(grid))
            for fold in self._fold_results:
                scores_per_t += _score_all_thresholds(
                    fold.y,
                    _positive_class_probs(fold.probs),
                    grid,
                    optimization_objective,
                )
            scores_per_t /= len(self._fold_results)

            self._threshold = grid[np.argmax(scores_per_t)]
            return self
        else:
            probs = self._get_positive_proba(self.model, X)

            if isinstance(self.optimizer, optimization.GridSearchOptimizer):
                grid = np.linspace(
                    self.optimizer.bounds[0],
                    self.optimizer.bounds[1],
                    self.optimizer.steps,
                )
                scores = _score_all_thresholds(
                    y, probs, grid, optimization_objective
                )
                self._threshold = grid[np.argmax(scores)]
            else:

                def objective_fn(params):
                    threshold = params[0]
                    y_pred = (probs >= threshold).astype(int)
                    return metrics.get_objective_score(
                        y, y_pred, optimization_objective
                    )

                initial_params = np.array([0.5])
                initial_score = objective_fn(initial_params)

                best_params, _ = self.optimizer.optimize(
                    objective_fn,
                    n_params=1,
                    current_best_params=initial_params,
                    current_best_score=initial_score,
                )

                self._threshold = best_params[0]
            return self


class DirichletCalibrationArgmax(BinaryDecisionStrategy):
    """Applies Dirichlet Calibration then uses argmax (threshold 0.5).

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
        self._threshold = 0.5
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        if self.calibrator is None:
            raise ValueError("Strategy not fit.")
        probs = self.model.predict_proba(X)
        calibrated_probs = self.calibrator.predict_proba(probs)
        return (calibrated_probs[:, 1] >= self.threshold).astype(int)


class DirichletCalibrationAnalytical(BinaryDecisionStrategy):
    """Applies Dirichlet Calibration then uses Analytical Threshold.

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
        if isinstance(optimization_objective, metrics.MetricObjective):
            raise ValueError(
                "DirichletCalibrationAnalytical requires"
                " a cost matrix for optimization,"
                " got MetricObjective."
            )
        if self.calibrator is None:
            probs = self.model.predict_proba(X)
            self.calibrator = calibration.DirichletCalibrator(
                reg_lambda=self.reg_lambda, reg_mu=self.reg_mu
            )
            self.calibrator.fit(probs, np.asarray(y))
        self._threshold = metrics.calculate_optimal_threshold(
            optimization_objective
        )
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        if self.calibrator is None:
            raise ValueError("Strategy not fit.")
        probs = self.model.predict_proba(X)
        calibrated_probs = self.calibrator.predict_proba(probs)
        return (calibrated_probs[:, 1] >= self.threshold).astype(int)
