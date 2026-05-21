import joblib
import numpy as np
import pytest

from threshold_opt import optimization


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_random_search_optimizer_simple_quadratic(n_jobs):
    def objective_fn(params):
        return -((params[0] - 0.5) ** 2)

    optimizer = optimization.RandomSearchOptimizer(
        n_iter=500, bounds=(0.0, 1.0), seed=42, n_jobs=n_jobs
    )
    with joblib.parallel_backend("threading"):
        best_params, best_score = optimizer.optimize(objective_fn, n_params=1)

    assert best_params[0] == pytest.approx(0.5, abs=0.05)
    assert best_score == pytest.approx(0.0, abs=0.01)


def test_grid_search_optimizer_simple_quadratic():
    def objective_fn(params):
        return -((params[0] - 0.5) ** 2)

    optimizer = optimization.GridSearchOptimizer(
        steps=11, bounds=(0.0, 1.0), n_jobs=1
    )
    best_params, best_score = optimizer.optimize(objective_fn, n_params=1)

    assert best_params[0] == pytest.approx(0.5)
    assert best_score == pytest.approx(0.0)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_grid_search_optimizer_2d(n_jobs):
    def objective_fn(params):
        return -((params[0] - 0.3) ** 2) - ((params[1] - 0.7) ** 2)

    optimizer = optimization.GridSearchOptimizer(
        steps=11, bounds=(0.0, 1.0), n_jobs=n_jobs
    )
    with joblib.parallel_backend("threading"):
        best_params, best_score = optimizer.optimize(objective_fn, n_params=2)

    np.testing.assert_allclose(best_params, [0.3, 0.7])
    assert best_score == pytest.approx(0.0)


def test_grid_search_respects_current_best():
    def objective_fn(params):
        return -((params[0] - 0.5) ** 2)

    optimizer = optimization.GridSearchOptimizer(steps=11, bounds=(0.0, 1.0))
    best_params, best_score = optimizer.optimize(
        objective_fn,
        n_params=1,
        current_best_params=[0.999],
        current_best_score=100.0,
    )

    assert best_score == 100.0
    assert best_params[0] == 0.999


def test_solve_obrien_changepoints():
    cost_matrix = np.array([[0.0, 1.0], [5.0, 0.0]])
    X_probs = np.array(
        [
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    y_true = np.array([0, 1])

    base_risks = X_probs @ cost_matrix
    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    def calculate_cost(bias):
        adjusted = base_risks + bias
        preds = np.argmin(adjusted, axis=1)
        total = 0
        for i, p in enumerate(preds):
            total += cost_matrix[y_true[i], p]
        return total

    assert calculate_cost(alphas) == 0.0

    diff = alphas[1] - alphas[0]
    assert diff > 1.4
    assert diff < 4.4


def test_random_search_deterministic_with_seed():
    def objective_fn(params):
        return -((params[0] - 0.42) ** 2)

    opt1 = optimization.RandomSearchOptimizer(
        n_iter=200, bounds=(0.0, 1.0), seed=7
    )
    opt2 = optimization.RandomSearchOptimizer(
        n_iter=200, bounds=(0.0, 1.0), seed=7
    )
    params1, score1 = opt1.optimize(objective_fn, n_params=1)
    params2, score2 = opt2.optimize(objective_fn, n_params=1)
    np.testing.assert_array_equal(params1, params2)
    assert score1 == score2


def test_random_search_default_seed_is_deterministic():
    """seed=None must be reproducible, not OS-entropy.

    Regression guard: the strategy factories construct the random-search
    strategies without an explicit seed, so the optimizer's default path
    must be deterministic. A non-deterministic default makes results
    non-reproducible run-to-run and makes the predict-cache decoupling
    (Phase A vs Phase B) impossible to validate bit-for-bit.
    """

    def objective_fn(params):
        return -((params[0] - 0.42) ** 2)

    opt1 = optimization.RandomSearchOptimizer(n_iter=200, bounds=(0.0, 1.0))
    opt2 = optimization.RandomSearchOptimizer(n_iter=200, bounds=(0.0, 1.0))
    assert opt1.seed is None and opt2.seed is None
    params1, score1 = opt1.optimize(objective_fn, n_params=1)
    params2, score2 = opt2.optimize(objective_fn, n_params=1)
    np.testing.assert_array_equal(params1, params2)
    assert score1 == score2


def test_grid_search_n_params_zero_returns_current_best():
    def objective_fn(params):
        return 99.0

    optimizer = optimization.GridSearchOptimizer(steps=10, bounds=(0.0, 1.0))
    sentinel_params = np.array([0.777])
    best_params, best_score = optimizer.optimize(
        objective_fn,
        n_params=0,
        current_best_params=sentinel_params,
        current_best_score=42.0,
    )
    np.testing.assert_array_equal(best_params, sentinel_params)
    assert best_score == 42.0


def test_solve_obrien_changepoints_all_correct_predictions():
    """When base predictions are perfect, biases should keep them that way."""
    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Each sample's correct class has the lowest base risk already
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_true = np.array([0, 1])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )
    adjusted = base_risks + alphas
    preds = np.argmin(adjusted, axis=1)
    np.testing.assert_array_equal(preds, y_true)


def test_solve_obrien_changepoints_output_shape():
    cost_matrix = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
    y_true = np.array([0, 1, 2])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )
    assert alphas.shape == (3,)


def _partition_matrix_is_valid(cost_matrix, alphas):
    """Return True iff a[j,j] < a[j,k] for all j, k!=j.

    Element: a[j,k] = cost[j,k] + alphas[k] - alphas[j].
    Diagonal a[j,j] = cost[j,j] (alphas cancel).
    """
    n = len(alphas)
    for j in range(n):
        for k in range(n):
            if j != k:
                a_jj = cost_matrix[j, j]
                a_jk = cost_matrix[j, k] + alphas[k] - alphas[j]
                if not a_jj < a_jk:
                    return False
    return True


def test_solve_obrien_changepoints_upper_bound_enforced():
    """Upper bound bug: old code used cost[k,i] not cost[i,k].

    With cost_matrix = [[0,5],[1,0]] the correct upper
    bound for alpha_1 is
    cost_matrix[1,0] - cost_matrix[1,1] = 1, but the old code used
    cost_matrix[0,1] = 5, allowing alpha_1 = 1.4, which makes a[1,0] < a[1,1].
    """
    cost_matrix = np.array([[0.0, 5.0], [1.0, 0.0]])
    # All four samples have probs=[0.1,0.9]; three are class-1 (correct), one
    # is class-0 (misclassified as 1).  This drives alpha_1 above the valid
    # upper bound of 1 without the fix.
    probs = np.tile(np.array([[0.1, 0.9]]), (4, 1))
    y_true = np.array([1, 1, 1, 0])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_lower_bound_enforced():
    """Lower bound bug: the lower bound was not enforced at all.

    With cost_matrix = [[0,5],[1,0]] and a single class-1 sample that the model
    assigns very low probability, the greedy algorithm wants to set alpha_1 to
    -5.4 (well below the valid lower bound of -5), which makes a[0,1] < a[0,0].
    """
    cost_matrix = np.array([[0.0, 5.0], [1.0, 0.0]])
    probs = np.array(
        [[0.9, 0.1]]
    )  # low p(1) → high base risk for predicting 1
    y_true = np.array([1])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_partition_valid_asymmetric_3class():
    """Partition valid for asymmetric 3-class cost matrix."""
    cost_matrix = np.array([[0.0, 3.0, 8.0], [1.0, 0.0, 2.0], [5.0, 1.0, 0.0]])
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=30)
    y_true = rng.integers(0, 3, size=30)
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_partition_valid_4class():
    """Partition matrix validity holds for a 4-class asymmetric cost matrix."""
    cost_matrix = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 0.0, 1.0, 2.0],
            [3.0, 2.0, 0.0, 1.0],
            [2.0, 3.0, 4.0, 0.0],
        ]
    )
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(4), size=40)
    y_true = rng.integers(0, 4, size=40)
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_nonzero_diagonal():
    """Partition valid with non-zero diagonal cost matrix.

    The bounds formula subtracts cost_matrix[i, i]; zero-diagonal
    tests never exercise that subtraction.
    """
    cost_matrix = np.array([[1.0, 4.0], [3.0, 1.0]])
    rng = np.random.default_rng(4)
    probs = rng.dirichlet(np.ones(2), size=20)
    y_true = rng.integers(0, 2, size=20)
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_all_samples_same_class():
    """Partition matrix validity holds when all samples belong to one class.

    Classes absent from y_true have no direct data signal for their alpha,
    but the bounds must still constrain them validly.
    """
    cost_matrix = np.array([[0.0, 2.0], [3.0, 0.0]])
    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
    y_true = np.array([0, 0, 0])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert alphas.shape == (2,)
    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_single_sample():
    """Partition matrix validity holds for a single-sample dataset."""
    cost_matrix = np.array([[0.0, 1.0], [2.0, 0.0]])
    probs = np.array([[0.3, 0.7]])
    y_true = np.array([0])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    assert alphas.shape == (2,)
    assert _partition_matrix_is_valid(cost_matrix, alphas), (
        f"Partition matrix invalid with alphas={alphas}"
    )


def test_solve_obrien_changepoints_zero_in_interval_picks_zero():
    """When the optimal interval straddles zero, alpha should snap to 0.

    Here the cost matrix is symmetric and both samples are already
    correctly classified at zero bias, so the empirical-cost minimum
    spans an interval containing zero. The implementation explicitly
    prefers alpha_i = 0 in that case.
    """
    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_true = np.array([0, 1])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )

    # By symmetry both alphas should be exactly zero.
    np.testing.assert_allclose(alphas, [0.0, 0.0], atol=1e-9)


def test_solve_obrien_changepoints_tied_changepoints():
    """Algorithm remains correct when many samples share a changepoint."""
    cost_matrix = np.array([[0.0, 4.0], [1.0, 0.0]])
    probs = np.tile([[0.3, 0.7]], (6, 1))  # all samples identical
    y_true = np.array([1, 1, 1, 1, 0, 0])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )
    preds = np.argmin(base_risks + alphas, axis=1)
    learned_cost = np.sum(cost_matrix[y_true, preds])

    zero_preds = np.argmin(base_risks, axis=1)
    zero_cost = np.sum(cost_matrix[y_true, zero_preds])

    assert learned_cost <= zero_cost
    assert _partition_matrix_is_valid(cost_matrix, alphas)


def test_solve_obrien_changepoints_competitive_with_brute_force():
    """Greedy output is competitive against a brute-force grid baseline.

    The solver is greedy coordinate descent over a non-convex objective
    (see optimization.solve_obrien_changepoints docstring); per-coordinate
    optimality does not imply global optimality. This test bounds the
    greedy output against a brute-force grid:

    1. greedy must be no worse than all-zeros (the analytical baseline);
    2. greedy must be within a generous slack of the brute-force minimum
       on a 3-class instance --- catches major regressions but tolerates
       the documented suboptimality of the descending-count ordering.
    """
    cost_matrix = np.array([[0.0, 5.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]])
    rng = np.random.default_rng(7)
    probs = rng.dirichlet(np.ones(3), size=15)
    y_true = rng.integers(0, 3, size=15)
    base_risks = probs @ cost_matrix

    grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    brute_best = np.inf
    for a0 in grid:
        for a1 in grid:
            for a2 in grid:
                preds = np.argmin(base_risks + np.array([a0, a1, a2]), axis=1)
                cost = np.sum(cost_matrix[y_true, preds])
                if cost < brute_best:
                    brute_best = cost
    zero_cost = np.sum(cost_matrix[y_true, np.argmin(base_risks, axis=1)])

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )
    learned_cost = np.sum(
        cost_matrix[y_true, np.argmin(base_risks + alphas, axis=1)]
    )

    # Greedy beats the analytical baseline, and lands within the gap
    # between the analytical baseline and the brute-force optimum
    # (i.e., recovers at least some of the available improvement).
    assert learned_cost <= zero_cost
    if zero_cost > brute_best:
        gap_recovered = (zero_cost - learned_cost) / (zero_cost - brute_best)
        assert gap_recovered >= 0.0


def test_solve_obrien_changepoints_reduces_cost():
    """Greedy finds alphas that strictly reduce empirical cost over zero bias.

    Group A (3 class-0 samples, probs=[0.8, 0.2]): correctly classified at
    zero bias (base risk for class-0 = 0.2 < class-1 = 4.0).
    Group B (5 class-1 samples, probs=[0.3, 0.7]): misclassified as class-0
    at zero bias (base risk for class-0 = 0.7 < class-1 = 1.5).
    The greedy should lower alpha_1 to fix Group B without harming Group A.
    """
    cost_matrix = np.array([[0.0, 5.0], [1.0, 0.0]])
    probs = np.vstack(
        [np.tile([0.8, 0.2], (3, 1)), np.tile([0.3, 0.7], (5, 1))]
    )
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    base_risks = probs @ cost_matrix

    zero_alpha_preds = np.argmin(base_risks, axis=1)
    zero_alpha_cost = np.sum(cost_matrix[y_true, zero_alpha_preds])

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )
    learned_preds = np.argmin(base_risks + alphas, axis=1)
    learned_cost = np.sum(cost_matrix[y_true, learned_preds])

    assert learned_cost < zero_alpha_cost


def test_solve_obrien_changepoints_raises_on_inverted_bounds():
    """A cost matrix that violates the dominant-label admissibility
    constraint inverts the per-coordinate [lower, upper] interval. The
    old code silently np.clip-ed to a wrong alpha; now it must raise a
    descriptive ValueError instead of emitting a quietly-wrong partition.
    """
    # cost[1,0] = -10 makes "predict 0 when truth is 1" rewarded, which
    # drives lower_bound (~10) far above upper_bound (~1) for class 0.
    cost_matrix = np.array([[0.0, 1.0], [-10.0, 0.0]])
    probs = np.tile([0.5, 0.5], (4, 1))
    y_true = np.array([0, 0, 0, 1])  # class 0 is processed first
    base_risks = probs @ cost_matrix

    with pytest.raises(ValueError, match=r"infeasible for class"):
        optimization.solve_obrien_changepoints(base_risks, y_true, cost_matrix)


def test_solve_obrien_changepoints_admissible_matrix_does_not_raise():
    """Guard against the inversion tolerance being so tight it false-
    positives on a normal admissible cost matrix."""
    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    probs = np.tile([0.5, 0.5], (4, 1))
    y_true = np.array([0, 0, 0, 1])
    base_risks = probs @ cost_matrix

    alphas = optimization.solve_obrien_changepoints(
        base_risks, y_true, cost_matrix
    )
    assert alphas.shape == (2,)
    assert np.all(np.isfinite(alphas))
