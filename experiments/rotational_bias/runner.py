"""Generate results for the rotational-bias controlled experiment.

This synthetic study is the controlled witness for the boundary-rotation
regime that motivates the Voronoi decision class in the dissertation
(Chapter "Experimental analysis: multiclass classification", section
"Controlled validation: boundary rotation scenario", LaTeX label
``sec:boundary_rotation_experiment``).

WHY THIS CONSTRUCTION (canonical record)
----------------------------------------
The thesis previously carried a formal lemma + proof in an appendix
("Naive Bayes misspecification and boundary rotation"). That appendix was
removed on purpose: Naive Bayes is only the simplest closed-form instance
of a general mechanism, and the prose was rewritten to avoid leaning on NB
as if the model itself mattered. The derivation is preserved here, since
this module is now the only place the exact construction and its rationale
live (the appendix text is otherwise only in git history before the
NB-removal commit).

Setup. Three Gaussian classes k = 0, 1, 2 in R^3 with per-class
covariances Sigma_k (see ``hetero_covs``) and means ``HETERO_MEANS``. A
GaussianNB estimator assumes a diagonal covariance per class. Every
Sigma_k has unit diagonal, so NB effectively uses Sigma_hat_k = I for all
k: the estimator is misspecified precisely in the off-diagonal
(correlation) structure, and that misspecification differs by class.

Mechanism. Let Delta_k := Sigma_k^{-1} - I be the per-class
misspecification matrix. Working through the Gaussian log-likelihoods, the
true optimal boundary between classes i and j in the NB probability
simplex satisfies

    log(p_hat_i / p_hat_j)
        = (1/2) x^T (Delta_i - Delta_j) x
          + (mu_j^T Delta_j - mu_i^T Delta_i) x
          + (c_j - c_i),

a quadratic in x whenever Delta_i != Delta_j. On the NB argmax surface the
left side is zero but the quadratic correction is generically nonzero, and
because it varies with position the optimal boundary does not merely
translate, it rotates in the simplex. Concretely for classes 0 and 1,
Sigma_0 and Sigma_1 differ only in the sign of rho in the off-diagonal, so

    Delta_0 - Delta_1
        = (-2 rho / (1 - rho^2)) * [[0, 1, 0], [1, 0, 0], [0, 0, 0]],

i.e. the correction reduces to -(2 rho / (1 - rho^2)) * x_1 * x_2 and the
true boundary acquires a component along e_2, the direction perpendicular
to e_0 - e_1 (1-indexed features x, simplex vertices e). An affine rule
fixes the class-i-vs-j boundary normal at e_i - e_j and tunes only a
scalar offset, so it cannot introduce that e_l (l not in {i, j})
component and incurs irreducible excess risk; the Voronoi class, whose
boundary normal mu_i - mu_j is a free parameter, can.

PARAMETER RATIONALE
-------------------
- ``RHO = 0.8``: strong enough that the quadratic correction (scaling like
  rho / (1 - rho^2)) visibly rotates the boundary, while staying in
  (-1, 1) so every Sigma_k is positive definite (eigenvalues 1 +/- rho
  and 1).
- ``hetero_covs``: classes 0 and 1 carry opposite-sign rho in the
  NumPy [0, 1] entry, class 2 carries rho in the [0, 2] entry, so the
  Delta_k differ in different coordinate planes; this makes the
  misspecification genuinely class-dependent (Delta_i - Delta_j nonzero
  for every pair).
- ``IMBAL_WEIGHTS = 10:2:1``: skews the argmax intersection away from the
  simplex barycentre, further handicapping translation-only (affine) rules
  relative to the rotation the Voronoi class can apply.
- ``HETERO_MEANS``: modest, partially overlapping separations so the
  decision boundaries fall where the NB misspecification is active rather
  than in near-deterministic tails.

Saves to experiments/results/:
  - results_rotational_bias_<timestamp>.parquet

The results table (tex/tables/results_rotational_bias.tex) is rendered from
this parquet by ``make tables``. The companion simplex-regions figure
(tex/figures/simplex_regions.tex) is produced independently of this parquet
by ``experiments.analysis.generate_simplex_regions_figure`` (run via
``make figures``), which refits the strategies in-process.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB

from experiments.common import seeds
from experiments.common.config import RANDOM_SEARCH_N_ITER
from experiments.common.git_metadata import write_parquet_with_metadata
from threshold_opt import metrics
from threshold_opt.multiclass_strategies import (
    ArgmaxMulticlass,
    FullMatrixPartitioningStrategy,
    MultidimensionalThresholdStrategy,
    ThresholdRatioMulticlass,
    VoronoiTuningMulticlass,
)
from threshold_opt.optimization import RandomSearchOptimizer

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -- Constants ---------------------------------------------------------------
N_TRAIN = 6_000
N_VAL = 3_000
N_TEST = 10_000
N_SEEDS = 10
MASTER_SEED = 42
# Shared decision-layer random-search budget (single source of truth in
# experiments/common/config.py). The controlled experiment uses the same
# budget as the OpenML campaign so the thesis can state one uniform value.
N_ITER = RANDOM_SEARCH_N_ITER
# All constants below are justified in the module docstring.
RHO = 0.8  # off-diagonal correlation; must stay in (-1, 1) for PD Sigma_k

IMBAL_WEIGHTS = np.array([10, 2, 1], dtype=float)  # skews argmax intersection
IMBAL_WEIGHTS = IMBAL_WEIGHTS / IMBAL_WEIGHTS.sum()

# Partially overlapping so boundaries land where NB misspecification is active.
HETERO_MEANS = [
    np.array([0, 0, 0]),
    np.array([2, 0, 0]),
    np.array([1, 2, 0]),
]

METRIC_FUNCTIONS = {
    "Macro-F1": lambda yt, yp: f1_score(
        yt, yp, average="macro", zero_division=0
    ),
    "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
}


# -- Helpers -----------------------------------------------------------------


def hetero_covs(rho):
    """Per-class covariances with class-dependent correlation structure.

    Unit diagonals throughout, so a diagonal-covariance GaussianNB sees
    Sigma_hat_k = I for every class and is misspecified only off-diagonal.
    Class 0 and class 1 carry +rho and -rho in the [0, 1] entry; class 2
    carries +rho in the [0, 2] entry. The off-diagonals sit in different
    coordinate planes so Delta_i - Delta_j is nonzero for every pair,
    which is what rotates the optimal simplex boundary. See the module
    docstring for the full derivation and the choice of ``rho``.
    """
    return [
        np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]]),
        np.array([[1, -rho, 0], [-rho, 1, 0], [0, 0, 1]]),
        np.array([[1, 0, rho], [0, 1, 0], [rho, 0, 1]]),
    ]


def generate_imbalanced_hetero(means, covs, n_total, class_weights, rng):
    """Sample with specified class proportions and per-class covariances."""
    K = len(means)
    counts = np.round(class_weights * n_total).astype(int)
    counts[-1] = n_total - counts[:-1].sum()
    Xs, ys = [], []
    for k in range(K):
        Xs.append(rng.multivariate_normal(means[k], covs[k], size=counts[k]))
        ys.append(np.full(counts[k], k, dtype=int))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def make_strategy_factories(n_iter):
    return {
        "Argmax": lambda nb, seed: ArgmaxMulticlass(nb),
        "ThresholdRatio": lambda nb, seed: ThresholdRatioMulticlass(
            nb, optimizer=RandomSearchOptimizer(n_iter=n_iter, seed=seed)
        ),
        "MultidimensionalThreshold": lambda nb, seed: (
            MultidimensionalThresholdStrategy(
                nb,
                optimizer=RandomSearchOptimizer(
                    n_iter=n_iter, bounds=(-1, 1), seed=seed
                ),
            )
        ),
        "VoronoiTuning": lambda nb, seed: VoronoiTuningMulticlass(
            nb, n_iter=n_iter, seed=seed
        ),
        "FullMatrixPartitioning": lambda nb, seed: (
            FullMatrixPartitioningStrategy(
                nb,
                optimizer=RandomSearchOptimizer(
                    n_iter=n_iter, bounds=(-1, 1), seed=seed
                ),
            )
        ),
    }


# -- Main --------------------------------------------------------------------


def run():
    factories = make_strategy_factories(N_ITER)
    metric_objs = {
        name: metrics.MetricObjective(
            metric_fn=fn,
            higher_is_better=True,
        )
        for name, fn in METRIC_FUNCTIONS.items()
    }

    covs = hetero_covs(RHO)
    seed_vec = seeds.sample_seeds(N_SEEDS, MASTER_SEED)
    print(f"Replicate seeds (master={MASTER_SEED}): {seed_vec}")
    records = []
    for obj_name, obj in metric_objs.items():
        for seed in seed_vec:
            rng = np.random.default_rng(seed)
            X_train, y_train = generate_imbalanced_hetero(
                HETERO_MEANS, covs, N_TRAIN, IMBAL_WEIGHTS, rng
            )
            X_val, y_val = generate_imbalanced_hetero(
                HETERO_MEANS, covs, N_VAL, IMBAL_WEIGHTS, rng
            )
            X_test, y_test = generate_imbalanced_hetero(
                HETERO_MEANS, covs, N_TEST, IMBAL_WEIGHTS, rng
            )

            nb = GaussianNB()
            nb.fit(X_train, y_train)

            for name, factory in factories.items():
                strat = factory(nb, seed)
                strat.fit(X_val, y_val, obj)
                y_pred = strat.predict(X_test)
                score = METRIC_FUNCTIONS[obj_name](y_test, y_pred)
                records.append(
                    {
                        "metric": obj_name,
                        "seed": seed,
                        "strategy": name,
                        "score": score,
                    }
                )
            print(f"  {obj_name} seed={seed} done")

    print(f"Total: {len(records)} records")
    return pd.DataFrame(records)


if __name__ == "__main__":
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    df = run()
    path = RESULTS_DIR / f"results_rotational_bias_{ts}.parquet"
    write_parquet_with_metadata(df, path)
    print(f"Saved {path}")
