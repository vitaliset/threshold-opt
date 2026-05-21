"""Post-hoc calibration methods.

Implements Dirichlet calibration (Kull et al., NeurIPS 2019) as a
self-contained module, avoiding an external dependency on `dirichletcal`.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import softmax


class DirichletCalibrator:
    """Dirichlet calibration (Kull et al., NeurIPS 2019).

    Learns a linear mapping in log-probability space:

        calibrated = softmax(W @ log(p) + b)

    where W is (K x K) and b is (K,), parameterised as a single
    (K x K+1) weight matrix.  Trained by minimising NLL on a held-out
    calibration set.

    Parameters
    ----------
    reg_lambda : float
        Weight-matrix L2 regularisation. With ``reg_mu is None`` (the
        DirL2 baseline in Kull et al., 2019) it penalises every entry of
        ``W`` with no normalisation. Under DirODIR (``reg_mu`` set) it
        penalises only the off-diagonal entries of ``W`` with the
        ``1/(k(k-1))`` factor from the paper.
    reg_mu : float or None
        Selects the DirODIR (Off-Diagonal and Intercept Regularised)
        variant of Kull et al. (2019). When not None, ``reg_lambda``
        switches to off-diagonal-only and ``reg_mu`` applies an L2
        penalty ``mu * (1/k) * sum(b**2)`` to the intercept; the
        diagonal of ``W`` is left free. When None, the plain DirL2
        baseline is used and the intercept is unregularised.
    """

    def __init__(self, reg_lambda: float = 0.0, reg_mu: float | None = None):
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.weights_: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        probs: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
    ) -> "DirichletCalibrator":
        """Fit the calibrator on held-out probabilities and labels."""
        n_samples, n_classes = probs.shape
        y = np.asarray(y)

        # Initialise as identity mapping (no calibration).
        W0 = np.zeros((n_classes, n_classes + 1))
        W0[:, :n_classes] = np.eye(n_classes)

        log_p = np.log(np.clip(probs, 1e-12, 1.0))
        X = np.hstack([log_p, np.ones((n_samples, 1))])

        result = minimize(
            _dirichlet_loss_and_grad,
            W0.flatten(),
            args=(X, y, n_classes, self.reg_lambda, self.reg_mu),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        self.weights_ = result.x.reshape(n_classes, n_classes + 1)
        return self

    def predict_proba(
        self, probs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return calibrated probabilities."""
        if self.weights_ is None:
            raise ValueError("Calibrator not fitted.")
        n_samples = probs.shape[0]
        log_p = np.log(np.clip(probs, 1e-12, 1.0))
        X = np.hstack([log_p, np.ones((n_samples, 1))])
        logits = X @ self.weights_.T
        return softmax(logits, axis=1)


def _dirichlet_loss_and_grad(
    w_flat: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    n_classes: int,
    reg_lambda: float,
    reg_mu: float | None,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Dirichlet-calibration NLL (+ L2/ODIR reg) and its analytical gradient.

    `X` is the precomputed `[log_p | 1]` design matrix of shape
    (n_samples, n_classes + 1). `W` is reshaped from `w_flat` as
    (n_classes, n_classes + 1).
    """
    n_samples = X.shape[0]
    W = w_flat.reshape(n_classes, n_classes + 1)
    logits = X @ W.T

    # Softmax and log-softmax in one numerically-stable pass.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)
    probs_soft = exp_shifted / sum_exp

    row_idx = np.arange(n_samples)
    nll = -np.mean(log_probs[row_idx, y])

    # d(nll)/d(logits) = (softmax - onehot(y)) / n_samples
    # d(nll)/d(W)      = d(logits).T @ X   (since logits = X @ W.T)
    dlogits = probs_soft
    dlogits[row_idx, y] -= 1.0
    dlogits /= n_samples
    grad_W = dlogits.T @ X

    # Regularisation follows Kull et al. (2019). ``reg_mu`` acts as a
    # mode switch rather than an additive penalty:
    #   reg_mu is None -> DirL2 baseline from the paper: plain L2 on the
    #                     full weight matrix (intercept excluded).
    #   reg_mu is set  -> DirODIR (eq. in §3): reg_lambda penalises the
    #                     off-diagonal entries of W with a 1/(k(k-1))
    #                     factor, reg_mu penalises the intercept b with
    #                     a 1/k factor, and the diagonal of W is free.
    loss = nll
    if reg_mu is None:
        if reg_lambda > 0:
            loss += reg_lambda * np.sum(W[:, :n_classes] ** 2)
            grad_W[:, :n_classes] += 2.0 * reg_lambda * W[:, :n_classes]
    else:
        if reg_lambda > 0 and n_classes > 1:
            off_diag = W[:, :n_classes].copy()
            np.fill_diagonal(off_diag, 0.0)
            norm_off = 1.0 / (n_classes * (n_classes - 1))
            loss += reg_lambda * norm_off * np.sum(off_diag**2)
            grad_W[:, :n_classes] += 2.0 * reg_lambda * norm_off * off_diag
        if reg_mu > 0:
            norm_b = 1.0 / n_classes
            loss += reg_mu * norm_b * np.sum(W[:, -1] ** 2)
            grad_W[:, -1] += 2.0 * reg_mu * norm_b * W[:, -1]

    return float(loss), grad_W.ravel()
