"""Shared local-linear smoother with bootstrap + local-SD prediction band.

Used by ``generate_variance_figure`` to render scatter + smoothed trend +
95% prediction-band panels with consistent statistical conventions.
"""

from __future__ import annotations

import numpy as np

N_GRID = 80
N_BOOTSTRAP = 400
BANDWIDTH_FRACTION = 0.18


def local_linear(
    x_eval: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Vectorised Gaussian-kernel local linear regression.

    At each ``x_eval[i]`` minimises
    ``sum_j w_ij (y_j - (a + b * (x_j - x_eval[i])))^2`` with weights
    ``w_ij = exp(-0.5 * ((x_j - x_eval[i]) / bandwidth)^2)``. The
    closed-form intercept ``a`` is the smoothed estimate at
    ``x_eval[i]``. Falls back to the weighted mean where the local
    design is too thin for a linear fit.
    """
    dx = x_obs[None, :] - x_eval[:, None]
    w = np.exp(-0.5 * (dx / bandwidth) ** 2)
    S0 = w.sum(axis=1)
    S1 = (w * dx).sum(axis=1)
    S2 = (w * dx**2).sum(axis=1)
    T0 = (w * y_obs).sum(axis=1)
    T1 = (w * dx * y_obs).sum(axis=1)
    denom = S0 * S2 - S1**2
    return np.where(
        np.abs(denom) > 1e-12,
        (S2 * T0 - S1 * T1) / denom,
        T0 / np.maximum(S0, 1e-12),
    )


def fit_loess_band(
    x: np.ndarray, y: np.ndarray, n_grid: int = N_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local-linear regression on (x, y) with a 95% prediction band.

    Returns ``(x_grid, mean, lower, upper)``. The band combines the
    bootstrap CI of the trend (``N_BOOTSTRAP`` resamples) with a
    Gaussian-kernel-weighted local residual SD, so it covers roughly
    95% of future seeds at each x rather than just the trend's
    uncertainty.
    """
    if x.size < 4:
        x_grid = np.linspace(
            x.min() if x.size else 0.0,
            x.max() if x.size else 1.0,
            n_grid,
        )
        zeros = np.zeros_like(x_grid)
        return x_grid, zeros, zeros, zeros
    x_range = float(max(x.max() - x.min(), 1e-3))
    bandwidth = BANDWIDTH_FRACTION * x_range
    x_grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    mean = local_linear(x_grid, x, y, bandwidth)

    rng = np.random.default_rng(0)
    n = x.size
    boot = np.empty((N_BOOTSTRAP, n_grid))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot[b] = local_linear(x_grid, x[idx], y[idx], bandwidth)
    ci_lower = np.percentile(boot, 2.5, axis=0)
    ci_upper = np.percentile(boot, 97.5, axis=0)
    ci_half = (ci_upper - ci_lower) / 2.0

    fit_at_obs = local_linear(x, x, y, bandwidth)
    resid_sq = (y - fit_at_obs) ** 2
    dx = x[None, :] - x_grid[:, None]
    w = np.exp(-0.5 * (dx / bandwidth) ** 2)
    local_var = (w * resid_sq).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    local_sd = np.sqrt(local_var)

    pi_half = np.sqrt(ci_half**2 + (1.96 * local_sd) ** 2)
    lower = mean - pi_half
    upper = mean + pi_half
    return x_grid, np.asarray(mean), np.asarray(lower), np.asarray(upper)
