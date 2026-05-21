"""Smoke tests for the sensitivity math + MC utilities."""

import numpy as np
import pytest

from experiments.sensitivity import sensitivity_math as sm


def test_kl_zero_at_alpha_one():
    assert sm.gaussian_kl_isotropic(d=2, alpha=1.0) == 0.0


def test_kl_positive_on_both_sides_of_one():
    for alpha in (0.25, 0.5, 1.5, 4.0):
        assert sm.gaussian_kl_isotropic(d=2, alpha=alpha) > 0


def test_kl_invalid_alpha():
    with pytest.raises(ValueError):
        sm.gaussian_kl_isotropic(d=2, alpha=0.0)


def test_kgon_centers_unit_radius():
    centers = sm.kgon_centers(K=6, radius=2.0)
    assert centers.shape == (6, 2)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 2.0)


@pytest.mark.parametrize("K", [2, 3, 4, 5, 8])
def test_simplex_centers_pairwise_geometry(K):
    r = 2.0
    centers = sm.simplex_centers(K, r)
    assert centers.shape == (K, K - 1)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), r, atol=1e-10)
    # Pairwise inner products are -r^2/(K-1); pairwise distances are equal.
    G = centers @ centers.T
    np.testing.assert_allclose(np.diag(G), r**2, atol=1e-10)
    off = G - np.diag(np.diag(G))
    expected_off = -(r**2) / (K - 1) * (np.ones((K, K)) - np.eye(K))
    np.testing.assert_allclose(off, expected_off, atol=1e-10)


@pytest.mark.parametrize("K", [3, 4, 6])
@pytest.mark.parametrize("radius", [0.5, 1.5])
def test_random_unit_shifts_have_exact_magnitude(K, radius):
    rng = np.random.default_rng(0)
    shifts = sm.random_unit_shifts(rng, K, d=K - 1, radius=radius)
    np.testing.assert_allclose(
        np.linalg.norm(shifts, axis=1), radius, atol=1e-10
    )


def test_random_unit_shifts_zero_radius_returns_zeros():
    rng = np.random.default_rng(0)
    shifts = sm.random_unit_shifts(rng, K=4, d=3, radius=0.0)
    np.testing.assert_array_equal(shifts, np.zeros((4, 3)))


def test_asymmetric_10x_cost_shape():
    cm = sm.asymmetric_10x_cost(K=4)
    np.testing.assert_array_equal(np.diag(cm), 0.0)
    # Last row is the 10x cost row.
    expected_last = np.array([10.0, 10.0, 10.0, 0.0])
    np.testing.assert_array_equal(cm[3], expected_last)
    # Other off-diagonals are 1.
    assert cm[0, 1] == 1.0


def test_symmetric_cost_shape():
    cm = sm.symmetric_cost(K=5)
    np.testing.assert_array_equal(np.diag(cm), 0.0)
    off_diag = cm - np.diag(np.diag(cm))
    np.testing.assert_array_equal(off_diag, np.ones((5, 5)) - np.eye(5))


def test_sample_mixture_respects_uniform_priors():
    rng = np.random.default_rng(0)
    means = sm.kgon_centers(K=4, radius=2.0)
    priors = np.full(4, 0.25)
    sigmas = np.ones(4)
    _, y = sm.sample_mixture(rng, means, sigmas, priors, n=400)
    counts = np.bincount(y, minlength=4)
    np.testing.assert_array_equal(counts, np.full(4, 100))


def test_sample_mixture_respects_non_uniform_priors():
    rng = np.random.default_rng(0)
    K = 6
    raw = np.exp(-np.arange(K))
    priors = raw / raw.sum()
    means = sm.kgon_centers(K, radius=2.0)
    sigmas = np.ones(K)
    n = 10_000
    _, y = sm.sample_mixture(rng, means, sigmas, priors, n=n)
    counts = np.bincount(y, minlength=K)
    expected = np.round(n * priors).astype(int)
    np.testing.assert_array_less(np.abs(counts - expected), 2)
    np.testing.assert_array_equal(counts.sum(), n)


def test_true_posterior_sums_to_one():
    rng = np.random.default_rng(0)
    means = sm.kgon_centers(K=4, radius=2.0)
    sigmas = np.ones(4)
    priors = np.full(4, 0.25)
    X = rng.standard_normal((50, 2))
    p = sm.true_posterior(X, means, sigmas, priors)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)


def test_regret_zero_at_alpha_one_mc():
    """Same-sample evaluation gives exactly zero regret when P0 == Q."""
    rng = np.random.default_rng(123)
    K = 4
    means = sm.kgon_centers(K, radius=2.0)
    sigmas = np.ones(K)
    priors = np.full(K, 1.0 / K)
    cost = sm.asymmetric_10x_cost(K)
    c_opt, c_mis, _ = sm.monte_carlo_costs(
        rng, means, sigmas, sigmas, priors, cost, n_mc=5_000
    )
    assert c_opt == c_mis


@pytest.mark.parametrize("K", [2, 3, 4, 5])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 2.0, 4.0])
def test_regret_nonneg(K, alpha):
    rng = np.random.default_rng(7 * K + int(100 * alpha))
    means = sm.kgon_centers(K, radius=2.0)
    priors = np.full(K, 1.0 / K)
    cost = sm.asymmetric_10x_cost(K)
    sigma_P0 = np.ones(K)
    sigma_Q = sigma_P0.copy()
    sigma_Q[0] = np.sqrt(alpha)
    c_opt, c_mis, _ = sm.monte_carlo_costs(
        rng, means, sigma_P0, sigma_Q, priors, cost, n_mc=20_000
    )
    # Regret is non-negative in expectation. With the asymmetric 10x
    # cost the per-sample standard deviation is up to ~5, so allow a
    # generous MC tolerance to keep this test stable.
    assert c_mis >= c_opt - 0.02


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_meanshift_regret_nonneg_under_symmetric_cost(K):
    """Regular-simplex per-class unit shifts give non-negative regret
    even under symmetric 0/1 cost (the variance perturbation gives
    identically zero regret in that setting, so the mean-shift
    experiment is the test of K-flattening that the variance one
    cannot probe).
    """
    rng = np.random.default_rng(13 * K)
    r = 0.5
    means_P0 = sm.simplex_centers(K, radius=2.0)
    shifts = sm.random_unit_shifts(rng, K, d=K - 1, radius=r)
    means_Q = means_P0 + shifts
    sigmas = np.ones(K)
    priors = np.full(K, 1.0 / K)
    cost = sm.symmetric_cost(K)
    c_opt, c_mis, _ = sm.monte_carlo_costs(
        rng,
        means_Q=means_Q,
        sigmas_P0=sigmas,
        sigmas_Q=sigmas,
        priors=priors,
        cost_matrix=cost,
        n_mc=20_000,
        means_P0=means_P0,
    )
    assert c_mis >= c_opt - 0.005
