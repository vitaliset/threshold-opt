"""Base classes for decision strategies."""

import abc
import dataclasses
from typing import Self

import numpy as np
import numpy.typing as npt

from threshold_opt import metrics

# Seed shared by every K-fold loop in the library and by the experiments
# pipeline's precompute (see ``experiments.common.pipeline``). Both call
# sites must agree on this value; otherwise the precompute cache desyncs
# from the strategy's own lazy refit and the experiment is broken in a
# hard-to-diagnose way.
CV_RANDOM_STATE = 42


@dataclasses.dataclass(frozen=True)
class _CVFoldPredictions:
    """Labels and predicted probabilities for a single CV validation fold."""

    y: np.ndarray
    probs: np.ndarray


def _coerce_integer_labels(y: npt.ArrayLike) -> np.ndarray:
    """Cast *y* to int when it's already integer-valued (even if float dtype).

    OpenML can return class labels encoded as float64 (values like 0.0, 1.0,
    2.0). Downstream multiclass strategies use labels as array indices, so
    we canonicalise here. Non-integer-valued floats (e.g. 0.5, regression
    targets) pass through untouched; the strategy-level validator rejects
    them before this runs.
    """
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    return arr.astype(int)


class DecisionStrategy(abc.ABC):
    """Abstract base class for decision strategies."""

    def __init__(self, model):
        self.model = model

    @property
    def needs_validation_set(self) -> bool:
        """Whether the strategy requires a validation set for calibration."""
        return True

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        """Fit the strategy using validation data."""
        self._validate_fit_input(X, y, optimization_objective)
        return self._fit(X, _coerce_integer_labels(y), optimization_objective)

    def _validate_fit_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> None:
        """Validates input data before fitting."""
        pass

    @abc.abstractmethod
    def _fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        optimization_objective: metrics.OptimizationObjective,
    ) -> Self:
        """Implementation of the fit logic."""

    @abc.abstractmethod
    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Predict class labels for X using the fitted strategy."""
