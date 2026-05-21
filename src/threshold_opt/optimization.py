"""Parameter optimization strategies."""

import abc
import itertools
from typing import Callable, Optional, Tuple

import joblib
import numpy as np
import numpy.typing as npt

# Fixed fallback seed for random search when no explicit seed is given.
# Without this, ``np.random.default_rng(None)`` draws fresh OS entropy on
# every call, so results are not reproducible run to run and the
# predict-cache decoupling (Phase A vs Phase B) can never be bit-exact.
# The experiment's statistical variability comes from the data-resampling
# seeds (train/test splits), not from the search RNG, so a deterministic
# search budget is the intended design; an explicit ``seed`` still
# overrides this default.
DEFAULT_RANDOM_SEARCH_SEED = 42


class ParameterOptimizer(abc.ABC):
    """Abstract base class for parameter optimization strategies."""

    @abc.abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[npt.ArrayLike], float],
        n_params: int,
        current_best_params: Optional[npt.ArrayLike] = None,
        current_best_score: float = -np.inf,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Finds parameters that maximize the objective function."""


class RandomSearchOptimizer(ParameterOptimizer):
    """Optimizes parameters via Random Search within specified bounds."""

    def __init__(
        self,
        n_iter: int = 100,
        bounds: Tuple[float, float] = (0.01, 1.0),
        seed: Optional[int] = None,
        n_jobs: int = 1,
    ):
        self.n_iter = n_iter
        self.bounds = bounds
        self.seed = seed
        self.n_jobs = n_jobs

    def _sample_params(
        self, rng: np.random.Generator, n_params: int
    ) -> npt.NDArray[np.float64]:
        return rng.uniform(self.bounds[0], self.bounds[1], size=n_params)

    def _optimize_batch(
        self,
        seed: int,
        n_iter: int,
        objective_fn: Callable[[npt.ArrayLike], float],
        n_params: int,
    ) -> Tuple[Optional[npt.NDArray[np.float64]], float]:
        rng = np.random.default_rng(seed)
        local_best_score = -np.inf
        local_best_params = None

        for _ in range(n_iter):
            params = self._sample_params(rng, n_params)
            score = objective_fn(params)
            if score > local_best_score:
                local_best_score = score
                local_best_params = params

        return local_best_params, local_best_score

    def optimize(
        self,
        objective_fn: Callable[[npt.ArrayLike], float],
        n_params: int,
        current_best_params: Optional[npt.ArrayLike] = None,
        current_best_score: float = -np.inf,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        best_params = (
            np.array(current_best_params)
            if current_best_params is not None
            else np.zeros(n_params)
        )
        best_score = current_best_score

        master_seed = (
            self.seed if self.seed is not None else DEFAULT_RANDOM_SEARCH_SEED
        )
        rng = np.random.default_rng(master_seed)
        seeds = rng.integers(0, 2**32, size=self.n_jobs)

        iters_per_job = [self.n_iter // self.n_jobs] * self.n_jobs
        for i in range(self.n_iter % self.n_jobs):
            iters_per_job[i] += 1

        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._optimize_batch)(
                seeds[i], iters_per_job[i], objective_fn, n_params
            )
            for i in range(self.n_jobs)
        )

        for params, score in results:
            if params is not None and score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score


class GridSearchOptimizer(ParameterOptimizer):
    """Optimizes parameters via Grid Search."""

    def __init__(
        self,
        steps: int = 10,
        bounds: Tuple[float, float] = (0.01, 1.0),
        n_jobs: int = 1,
    ):
        self.steps = steps
        self.bounds = bounds
        self.n_jobs = n_jobs

    def _optimize_chunk(
        self,
        p0_values: npt.NDArray[np.float64],
        other_param_grids: Tuple[npt.NDArray[np.float64], ...],
        objective_fn: Callable[[npt.ArrayLike], float],
    ) -> Tuple[Optional[npt.NDArray[np.float64]], float]:
        local_best_score = -np.inf
        local_best_params = None

        for p0 in p0_values:
            for rest in itertools.product(*other_param_grids):
                params = np.array((p0, *rest))
                score = objective_fn(params)
                if score > local_best_score:
                    local_best_score = score
                    local_best_params = params

        return local_best_params, local_best_score

    def optimize(
        self,
        objective_fn: Callable[[npt.ArrayLike], float],
        n_params: int,
        current_best_params: Optional[npt.ArrayLike] = None,
        current_best_score: float = -np.inf,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        best_params = (
            np.array(current_best_params)
            if current_best_params is not None
            else np.zeros(n_params)
        )
        best_score = current_best_score

        grid_1d = np.linspace(self.bounds[0], self.bounds[1], self.steps)

        if n_params == 0:
            return best_params, best_score

        p0_chunks = np.array_split(grid_1d, self.n_jobs)
        other_param_grids = (grid_1d,) * (n_params - 1)

        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._optimize_chunk)(
                chunk, other_param_grids, objective_fn
            )
            for chunk in p0_chunks
            if chunk.size > 0
        )

        for params, score in results:
            if params is not None and score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score


def solve_obrien_changepoints(
    base_risks: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int_],
    cost_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves the partition shift using the Greedy Changepoint algorithm.

    Implements the method described in O'Brien et al. (2008), Section 5.1.
    Per-coordinate optimal but not globally optimal: classes are processed
    in descending count order and one ``alpha_i`` is updated at a time with
    the others held fixed, which is greedy coordinate descent over a
    non-convex piecewise-constant objective. ``optimization_test`` includes
    a brute-force comparison that bounds the gap on small instances.

    Convention note: this file uses ``cost_matrix[u, v]`` = cost of
    predicting ``v`` when the true class is ``u`` (row=true, col=pred),
    matching ``metrics.calculate_empirical_risk``. The paper uses the
    opposite convention ``c_{i|j}`` (row=pred, col=true), so a paper
    element ``a_{i|j}`` corresponds to ``cost_matrix[j, i]`` here. The
    partition element in this convention is
    ``a[u, v] = cost[u, v] + alpha[v] - alpha[u]``.
    """
    n_samples, n_classes = base_risks.shape
    alphas = np.zeros(n_classes)

    counts = np.bincount(y_true, minlength=n_classes)
    class_order = np.argsort(counts)[::-1]

    for i in class_order:
        temp_risks = base_risks + alphas
        temp_risks[:, i] = np.inf

        best_alts = np.argmin(temp_risks, axis=1)
        risk_alts = temp_risks[np.arange(n_samples), best_alts]

        risk_i = base_risks[:, i]
        changepoints = risk_alts - risk_i

        sorted_indices = np.argsort(changepoints)
        sorted_changepoints = changepoints[sorted_indices]

        cost_if_i = cost_matrix[y_true, i]
        cost_if_alt = cost_matrix[y_true, best_alts]

        deltas = cost_if_alt[sorted_indices] - cost_if_i[sorted_indices]

        current_total_cost = np.sum(cost_if_i)
        min_total_cost = current_total_cost

        best_alpha_i = sorted_changepoints[0] - 1.0

        cumulative_deltas = np.cumsum(deltas)
        total_costs = current_total_cost + cumulative_deltas

        min_idx = np.argmin(total_costs)
        if total_costs[min_idx] < min_total_cost:
            if min_idx < n_samples - 1:
                low = sorted_changepoints[min_idx]
                high = sorted_changepoints[min_idx + 1]
                # Paper specifies the geometric mean of the changepoint
                # interval; we use the arithmetic midpoint because any
                # interior point of the open interval induces the same
                # partition and changepoints can be negative, which makes
                # the geometric mean ill-defined.
                best_alpha_i = (low + high) / 2.0

                if low < 0 < high:
                    # Snap to exact zero whenever zero is interior to the
                    # admissible interval. Both endpoints of the open
                    # interval induce the same partition cost (any choice
                    # in the open interval is a global minimizer of the
                    # 1D coordinate update), so picking 0 keeps the
                    # analytical baseline as a fixed point of the
                    # algorithm. The tradeoff is documented in §6.3 of the
                    # dissertation: when many coordinates' admissible intervals
                    # straddle zero, the algorithm's output is biased
                    # toward the analytical baseline rather than the
                    # arithmetic midpoint of the changepoint interval.
                    best_alpha_i = 0.0
            else:
                best_alpha_i = sorted_changepoints[-1] + 1.0

        other_idx = np.where(np.arange(n_classes) != i)[0]

        # Upper bound: a[i,k] = cost[i,k] + alpha_k - alpha_i
        # must exceed diagonal a[i,i] = cost[i,i].
        # alpha_i < cost[i,k] + alpha_k - cost[i,i] for k!=i
        upper_bound = (
            np.min(cost_matrix[i, other_idx] + alphas[other_idx])
            - cost_matrix[i, i]
            - 1e-6
        )

        # Lower bound: a[j,i] = cost[j,i] + alpha_i - alpha_j
        # must exceed diagonal a[j,j] = cost[j,j].
        # alpha_i > cost[j,j] - cost[j,i] + alpha_j for j!=i
        lower_bound = (
            np.max(
                cost_matrix[other_idx, other_idx]
                - cost_matrix[other_idx, i]
                + alphas[other_idx]
            )
            + 1e-6
        )

        # An inverted interval means no admissible alpha_i exists: the
        # cost matrix violates the dominant-label constraint the bounds
        # encode. np.clip would silently return upper_bound here and
        # emit a quietly-wrong partition, so fail loudly instead. The
        # +1e-6 tolerance absorbs the two 1e-6 bound epsilons (plus
        # float noise) so a feasible-but-razor-thin interval is kept.
        if lower_bound > upper_bound + 1e-6:
            raise ValueError(
                f"O'Brien partition infeasible for class {int(i)}: lower"
                f" bound {lower_bound:.6g} exceeds upper bound"
                f" {upper_bound:.6g}. The cost matrix violates the"
                f" dominant-label admissibility constraint; the"
                f" changepoint update would be silently clipped to a"
                f" wrong alpha. Check the cost matrix."
            )

        best_alpha_i = np.clip(best_alpha_i, lower_bound, upper_bound)

        alphas[i] = best_alpha_i

    return alphas
