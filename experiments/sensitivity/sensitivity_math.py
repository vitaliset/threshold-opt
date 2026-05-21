"""Closed-form Gaussian utilities for the KL-sensitivity experiment.

Pure numpy. No classifier is fit anywhere; the data-generating process is
known and the Bayes-optimal decision rule is evaluated directly on true
posteriors.

The experiment perturbs a single class's isotropic variance by a factor
``alpha`` and measures the cost of using P0's Bayes-optimal rule under
the perturbed distribution Q. Posteriors are computed via
``log_class_densities`` plus ``logsumexp`` for stability; the Bayes
decision rule itself mirrors ``AnalyticalMulticlass.predict`` in
``src/threshold_opt/multiclass_strategies.py`` (argmin of
``posterior @ cost_matrix``).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def gaussian_kl_isotropic(d: int, alpha: float) -> float:
    """Closed-form KL[N(mu, alpha I_d) || N(mu, I_d)].

    Equal means, isotropic variance, only the second factor scaled by
    alpha. With every class's variance scaled by the same alpha and
    equal priors, the joint KL(Q || P0) over (x, y) is this same
    expression (the per-class KLs are identical and the priors sum to
    one).
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return (d / 2.0) * (alpha - 1.0 - np.log(alpha))


def kgon_centers(K: int, radius: float) -> np.ndarray:
    """K class centers on a regular polygon of given radius in d=2."""
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    angles = 2.0 * np.pi * np.arange(K) / K
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def simplex_centers(K: int, radius: float) -> np.ndarray:
    """K class centers at the vertices of a regular simplex in R^{K-1}.

    Every pair of vertices is equidistant (no geometric privilege for
    "adjacent" classes, unlike the K-gon in R^2), so a per-class
    independent perturbation spreads its posterior shift across all
    K-1 simplex tangent directions instead of concentrating on the
    2-3 nearest neighbours.

    Returns shape ``(K, K-1)``. Pairwise inner products are
    ``-radius**2 / (K-1)``; norms are ``radius``.
    """
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    # Centered canonical basis: K vectors in R^K in the null space of 1_K.
    embed = np.eye(K) - 1.0 / K
    # Orthonormal basis of null(1_K) (K-1 directions).
    Q, _ = np.linalg.qr(embed[:, : K - 1])
    coords = embed @ Q  # shape (K, K-1)
    coords = coords * (radius / np.linalg.norm(coords, axis=1, keepdims=True))
    return coords


def random_unit_shifts(
    rng: np.random.Generator, K: int, d: int, radius: float
) -> np.ndarray:
    """K independent per-class shifts at exact magnitude ``radius``.

    Each row is a unit vector in ``R^d`` drawn uniformly on the sphere
    ``S^{d-1}`` and scaled to length ``radius``. The per-class
    Euclidean length is deterministic, so the joint KL between
    ``P0`` (unshifted means) and ``Q`` (shifted means)
    ``r^2 / (2 sigma^2)`` is itself deterministic under uniform
    priors; only the direction of each shift varies seed-to-seed.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if radius == 0:
        return np.zeros((K, d))
    raw = rng.standard_normal((K, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return radius * raw / norms


def asymmetric_10x_cost(K: int) -> np.ndarray:
    """K x K cost matrix; misclassifying the last class costs 10x.

    Matches the convention in ``experiments/multiclass/config.py``
    (``create_mild_asymmetry_cost_matrix``).
    """
    cm = np.ones((K, K))
    np.fill_diagonal(cm, 0.0)
    cm[K - 1, :] = 10.0
    cm[K - 1, K - 1] = 0.0
    return cm


def asymmetric_5x_cost(K: int) -> np.ndarray:
    """K x K cost matrix; misclassifying the last class costs 5x.

    Same shape as ``asymmetric_10x_cost`` with a milder asymmetry,
    used to probe how the empirical K-decay exponent depends on the
    cost-asymmetry strength.
    """
    cm = np.ones((K, K))
    np.fill_diagonal(cm, 0.0)
    cm[K - 1, :] = 5.0
    cm[K - 1, K - 1] = 0.0
    return cm


def asymmetric_2x_cost(K: int) -> np.ndarray:
    """K x K cost matrix; misclassifying the last class costs 2x.

    At a 2x penalty, the centered-cost-column squared norm
    ``||g_j||^2 = c_j - mean(c_j) 1`` is approximately K-invariant for
    every action (a fixed point of the K-dependence of the
    Asymmetric-Mx family), which removes the prefactor K-scaling that
    otherwise shifts the empirical regret K-decay exponent away from
    the dimensional-dilution corollary's leading order of 1.
    """
    cm = np.ones((K, K))
    np.fill_diagonal(cm, 0.0)
    cm[K - 1, :] = 2.0
    cm[K - 1, K - 1] = 0.0
    return cm


def symmetric_cost(K: int) -> np.ndarray:
    """K x K symmetric 0/1 loss matrix (1 off-diagonal, 0 on-diagonal)."""
    return np.ones((K, K)) - np.eye(K)


def log_class_densities(
    X: np.ndarray, means: np.ndarray, sigmas: np.ndarray
) -> np.ndarray:
    """Log isotropic Gaussian density log p(x | y=k) for each (sample, class).

    ``X``: ``(N, d)``. ``means``: ``(K, d)``. ``sigmas``: ``(K,)`` (per-class
    standard deviation; covariance is ``sigma**2 * I_d``).
    Returns ``(N, K)``.
    """
    X = np.asarray(X, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    d = X.shape[1]
    diff = X[:, None, :] - means[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    var = sigmas**2
    return (
        -0.5 * sq / var[None, :]
        - (d / 2.0) * np.log(2.0 * np.pi * var)[None, :]
    )


def true_posterior(
    X: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """P(y=k | x) for each (sample, class) via softmax of log-joint."""
    log_lik = log_class_densities(X, means, sigmas)
    log_joint = log_lik + np.log(np.asarray(priors, dtype=np.float64))[None, :]
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def bayes_rule(
    X: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    priors: np.ndarray,
    cost_matrix: np.ndarray,
) -> np.ndarray:
    """Bayes-optimal action per sample under the given posterior and cost.

    Mirrors the one-liner in
    ``threshold_opt.multiclass_strategies.AnalyticalMulticlass.predict``
    but operates on closed-form posteriors rather than predicted ones.
    """
    posterior = true_posterior(X, means, sigmas, priors)
    risk = posterior @ np.asarray(cost_matrix, dtype=np.float64)
    return np.argmin(risk, axis=1)


def sample_mixture(
    rng: np.random.Generator,
    means: np.ndarray,
    sigmas: np.ndarray,
    priors: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n samples from the mixture, stratified by class.

    Per-class counts are proportional to ``priors``. The fractional
    remainders are allocated to the classes with the largest leftover
    mass so the per-class counts match a rounded version of
    ``n * priors``; this keeps MC variance low while still respecting
    a non-uniform prior. With uniform priors the allocation reduces
    to ``np.array_split``-style equal counts.
    """
    means = np.asarray(means, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    K, d = means.shape
    raw = n * priors
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder > 0:
        fractional = raw - counts
        extra_idx = np.argsort(-fractional)[:remainder]
        counts[extra_idx] += 1
    X_parts = []
    y_parts = []
    for k in range(K):
        nk = int(counts[k])
        if nk == 0:
            continue
        X_parts.append(means[k] + sigmas[k] * rng.standard_normal((nk, d)))
        y_parts.append(np.full(nk, k, dtype=int))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def sample_uniform_in_V(
    rng: np.random.Generator, K: int, r: float, n: int
) -> np.ndarray:
    """Sample n vectors uniformly on the radius-r sphere in V.

    ``V = {v in R^K : 1^T v = 0}`` is the (K-1)-dimensional centered
    tangent space of the simplex. Samples are drawn as iid standard
    normals in ``R^K``, projected to V by subtracting the mean, then
    rescaled to length ``r``. The result is uniform on the radius-r
    sphere in V (the Gaussian distribution is rotationally invariant in
    V after centering, so the normalized direction is uniform).

    Returns ``(n, K)``.
    """
    raw = rng.standard_normal((n, K))
    raw = raw - raw.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return r * raw / norms


def synthetic_lemma_test(
    rng: np.random.Generator,
    K: int,
    r: float,
    cost_matrix: np.ndarray,
    n_p: int,
    n_u: int,
    *,
    dirichlet_alpha: float = 2.0,
) -> tuple[float, float, float]:
    """Lemma "dimensional dilution" direct test in its native setting.

    Samples ``n_p`` reference posteriors from
    ``Dirichlet(dirichlet_alpha)`` on the simplex interior, and at each
    one samples ``n_u`` perturbations uniformly on the radius-r sphere
    in the (K-1)-dimensional tangent space V (Assumption "Diffuse
    perturbation direction"). For the cost-sensitive rule ``delta(p) =
    argmin_j (p^T cost_matrix)_j``, computes:

    * ``dphi_sq_linear`` = MC estimate of
      ``E_p[E_u[(g(p)^T u)^2]]``, the squared inner product of the
      Lemma's gradient with the perturbation. This is the leading-order
      term in the Lemma's expansion, valid for any p (the lemma's
      smoothness remainder is identically zero for the linearisation).
      Predicted to equal ``r^2/(K-1) E_p[||g(p)||^2]`` exactly.

    * ``dphi_sq_full`` = MC estimate of
      ``E_p[E_u[(phi_delta(p+u) - phi_delta(p))^2]]``, the full squared
      functional change including the contribution from samples where
      ``p + u`` lands in a different decision region of ``delta``. For
      piecewise-linear ``phi_delta`` and Dirichlet-sampled ``p``, this
      differs from the linearised quantity by an action-flip term whose
      magnitude depends on the cost matrix's range and the distance
      from p to the nearest decision boundary. The Lemma's smoothness
      assumption (Eq.~``eq:smoothness_bound``) explicitly excludes the
      flip regime, so the linearised quantity is the right object to
      verify the leading-order prediction.

    * ``g_sq_mean`` = MC estimate of ``E_p[||g_delta(p)||^2]``, the
      Lemma's prefactor.

    Regressing ``log(dphi_sq_linear / g_sq_mean)`` against ``log(K - 1)``
    at fixed r recovers the K-decay exponent the Lemma predicts (1 at
    leading order, exactly across cost configurations).
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    p_0 = rng.dirichlet(np.full(K, dirichlet_alpha), size=n_p)
    a_0 = np.argmin(p_0 @ cost_matrix, axis=1)
    c_at_a = cost_matrix[a_0]
    g = c_at_a - c_at_a.mean(axis=1, keepdims=True)
    g_sq_mean = float((g * g).sum(axis=1).mean())
    u = sample_uniform_in_V(rng, K, r, n_p * n_u).reshape(n_p, n_u, K)
    # Linearised quantity: (g^T u)^2.
    delta_phi_linear = np.einsum("pk,puk->pu", g, u)
    dphi_sq_linear = float((delta_phi_linear**2).mean())
    # Full quantity: (phi(p+u) - phi(p))^2 with the cost-sensitive
    # argmin re-evaluated at p+u (so action flips contribute).
    p_plus_u = p_0[:, None, :] + u
    a_pert = np.argmin(p_plus_u @ cost_matrix, axis=2)
    c_at_a_pert = cost_matrix[a_pert]
    phi_pert = (c_at_a_pert * p_plus_u).sum(axis=2)
    phi_p = (c_at_a * p_0).sum(axis=1, keepdims=True)
    delta_phi_full = phi_pert - phi_p
    dphi_sq_full = float((delta_phi_full**2).mean())
    return dphi_sq_linear, dphi_sq_full, g_sq_mean


def monte_carlo_costs(
    rng: np.random.Generator,
    means_Q: np.ndarray,
    sigmas_P0: np.ndarray,
    sigmas_Q: np.ndarray,
    priors: np.ndarray,
    cost_matrix: np.ndarray,
    n_mc: int,
    *,
    means_P0: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate (C_opt, C_misspec, regret_sq_mean) under Q via shared-
    sample Monte Carlo.

    Draws one batch ``(X, y) ~ Q`` and evaluates both rules on the same
    points. Same-sample evaluation collapses MC variance of the regret
    estimate ``C_misspec - C_opt`` whenever the rules agree on a point.
    ``means_P0`` defaults to ``means_Q`` (variance-only perturbation);
    pass a distinct array for perturbations that move the class means,
    e.g. mean rotation.

    ``regret_sq_mean`` is the MC estimate of
    ``E_X[(C_{Y, delta_0(X)} - C_{Y, delta_Q(X)})^2]``, the second
    moment of the per-sample realised regret.
    """
    if means_P0 is None:
        means_P0 = means_Q
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    X, y = sample_mixture(rng, means_Q, sigmas_Q, priors, n_mc)
    a_opt = bayes_rule(X, means_Q, sigmas_Q, priors, cost_matrix)
    a_mis = bayes_rule(X, means_P0, sigmas_P0, priors, cost_matrix)
    per_sample_opt = cost_matrix[y, a_opt]
    per_sample_mis = cost_matrix[y, a_mis]
    per_sample_diff = per_sample_mis - per_sample_opt
    c_opt = float(per_sample_opt.mean())
    c_mis = float(per_sample_mis.mean())
    regret_sq_mean = float((per_sample_diff**2).mean())
    return c_opt, c_mis, regret_sq_mean
