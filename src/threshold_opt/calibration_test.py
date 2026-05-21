import numpy as np
import pytest
from scipy.optimize import check_grad

from threshold_opt.calibration import (
    DirichletCalibrator,
    _dirichlet_loss_and_grad,
)


@pytest.fixture
def three_class_data():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=30)
    y = rng.integers(0, 3, size=30)
    return probs, y


def test_output_is_valid_probability_matrix(three_class_data):
    probs, y = three_class_data
    cal = DirichletCalibrator()
    cal.fit(probs, y)
    out = cal.predict_proba(probs)

    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(axis=1), np.ones(len(probs)), atol=1e-6)
    assert np.all(out >= 0)


def test_predict_before_fit_raises():
    probs = np.array([[0.5, 0.3, 0.2]])
    cal = DirichletCalibrator()
    with pytest.raises(ValueError, match="not fitted"):
        cal.predict_proba(probs)


def test_weights_shape_after_fit(three_class_data):
    probs, y = three_class_data
    cal = DirichletCalibrator()
    cal.fit(probs, y)
    # (K, K+1): K weight columns + 1 bias column
    assert cal.weights_.shape == (3, 4)


def test_training_nll_does_not_increase(three_class_data):
    """L-BFGS-B starts from the identity mapping, so training NLL after
    calibration is ≤ NLL of the uncalibrated probabilities."""
    probs, y = three_class_data

    nll_before = -np.mean(
        np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
    )

    cal = DirichletCalibrator()
    cal.fit(probs, y)
    calibrated = cal.predict_proba(probs)
    nll_after = -np.mean(
        np.log(np.clip(calibrated[np.arange(len(y)), y], 1e-12, 1.0))
    )

    assert nll_after <= nll_before + 1e-6


def test_binary_case():
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    y = np.array([0, 1, 0, 1])

    cal = DirichletCalibrator()
    cal.fit(probs, y)
    out = cal.predict_proba(probs)

    assert out.shape == (4, 2)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-6)
    assert cal.weights_.shape == (2, 3)


def test_odir_reg_output_is_valid(three_class_data):
    """ODIR regularisation (reg_mu) still produces valid probabilities."""
    probs, y = three_class_data
    cal = DirichletCalibrator(reg_mu=0.1)
    cal.fit(probs, y)
    out = cal.predict_proba(probs)

    np.testing.assert_allclose(out.sum(axis=1), np.ones(len(probs)), atol=1e-6)
    assert np.all(out >= 0)


def test_lambda_reg_output_is_valid(three_class_data):
    """L2 regularisation (reg_lambda) still produces valid probabilities."""
    probs, y = three_class_data
    cal = DirichletCalibrator(reg_lambda=0.5)
    cal.fit(probs, y)
    out = cal.predict_proba(probs)

    np.testing.assert_allclose(out.sum(axis=1), np.ones(len(probs)), atol=1e-6)
    assert np.all(out >= 0)


def test_fit_returns_self(three_class_data):
    probs, y = three_class_data
    cal = DirichletCalibrator()
    result = cal.fit(probs, y)
    assert result is cal


@pytest.mark.parametrize(
    "reg_lambda,reg_mu",
    [
        (0.0, None),
        (0.5, None),
        (0.0, 0.25),
        (0.1, 0.1),
    ],
)
def test_analytical_gradient_matches_finite_difference(reg_lambda, reg_mu):
    """The analytical Jacobian must agree with scipy's finite-difference
    approximation at a random perturbation of the identity initialisation,
    for every combination of regularisation flags."""
    rng = np.random.default_rng(7)
    n_samples, n_classes = 50, 4

    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y = rng.integers(0, n_classes, size=n_samples)

    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    X = np.hstack([log_p, np.ones((n_samples, 1))])

    w = rng.normal(size=n_classes * (n_classes + 1)) * 0.3
    w[: n_classes * n_classes] += np.eye(n_classes).ravel()

    def loss_only(w_):
        return _dirichlet_loss_and_grad(
            w_, X, y, n_classes, reg_lambda, reg_mu
        )[0]

    def grad_only(w_):
        return _dirichlet_loss_and_grad(
            w_, X, y, n_classes, reg_lambda, reg_mu
        )[1]

    err = check_grad(loss_only, grad_only, w)
    assert err < 1e-5, f"check_grad error {err:.2e} too large"


def test_odir_regularises_off_diagonal_and_intercept_only():
    """Under DirODIR (reg_mu set), reg_lambda must leave the diagonal of
    W untouched and reg_mu must only penalise the intercept. We pin this
    down by evaluating the loss at a hand-picked W and checking the
    penalty term analytically against the paper's formula
    lambda * (1/(k(k-1))) * sum_off w_ij^2 + mu * (1/k) * sum_j b_j^2.
    """
    n_samples, n_classes = 5, 3
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y = np.zeros(n_samples, dtype=int)
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    X = np.hstack([log_p, np.ones((n_samples, 1))])

    W = np.array(
        [
            [2.0, 0.5, 0.5, 1.0],
            [0.5, 2.0, 0.5, 1.0],
            [0.5, 0.5, 2.0, 1.0],
        ]
    )
    reg_lambda, reg_mu = 0.7, 0.3

    loss_plain, _ = _dirichlet_loss_and_grad(
        W.ravel(), X, y, n_classes, 0.0, None
    )
    loss_reg, _ = _dirichlet_loss_and_grad(
        W.ravel(), X, y, n_classes, reg_lambda, reg_mu
    )

    # Paper formula for ODIR penalty. Off-diagonal entries are all 0.5
    # (6 of them), diagonal is ignored, intercept is all 1.0.
    off_sq = 6 * 0.5**2
    b_sq = 3 * 1.0**2
    expected_penalty = (
        reg_lambda * off_sq / (n_classes * (n_classes - 1))
        + reg_mu * b_sq / n_classes
    )
    assert (loss_reg - loss_plain) == pytest.approx(
        expected_penalty, abs=1e-12
    )


def test_diril2_penalises_full_weight_matrix():
    """DirL2 (reg_mu None): reg_lambda*sum(W[:,:k]**2), intercept free."""
    n_samples, n_classes = 4, 2
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y = np.zeros(n_samples, dtype=int)
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    X = np.hstack([log_p, np.ones((n_samples, 1))])

    W = np.array([[1.5, 0.3, 2.0], [0.7, 1.1, 2.0]])
    reg_lambda = 0.4

    loss_plain, _ = _dirichlet_loss_and_grad(
        W.ravel(), X, y, n_classes, 0.0, None
    )
    loss_reg, _ = _dirichlet_loss_and_grad(
        W.ravel(), X, y, n_classes, reg_lambda, None
    )

    expected_penalty = reg_lambda * np.sum(W[:, :n_classes] ** 2)
    assert (loss_reg - loss_plain) == pytest.approx(
        expected_penalty, abs=1e-12
    )


def test_strongly_miscalibrated_model_corrected():
    """A model that always predicts class 0 with 90% confidence, but the
    true labels are mostly class 1.  After calibration the predicted
    probability for the true class should increase on average."""
    n = 40
    # Model always outputs [0.9, 0.05, 0.05]
    probs = np.tile(np.array([[0.9, 0.05, 0.05]]), (n, 1))
    # True labels are mostly class 1
    y = np.array([1] * 30 + [0] * 10)

    cal = DirichletCalibrator()
    cal.fit(probs, y)
    calibrated = cal.predict_proba(probs)

    # After calibration class 1 should have higher predicted probability
    assert calibrated[0, 1] > probs[0, 1]
