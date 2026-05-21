"""Generate tex/figures/simplex_regions.tex from the tikz_simplex module.

Five-panel probability-simplex figure for the boundary-rotation scenario
(K=3, correlation 0.8, imbalanced heteroscedastic): Argmax, ThresholdRatio,
MultidimensionalThreshold, Voronoi, FullMatrixPartitioning. Each panel shades
the predicted-class regions and overlays a subsample of held-out test
posteriors coloured by true class.

The figure is self-contained: it regenerates the synthetic data and refits
the five strategies in-process with fixed seeds, so it depends on no parquet.
The decision-layer search budget is kept in lockstep with the runner via
``experiments.common.config.RANDOM_SEARCH_N_ITER`` so the visualised
boundaries match the experiment.

Usage:
    uv run python -m experiments.analysis.generate_simplex_regions_figure
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

from experiments.analysis.tikz_simplex import (
    bary_to_cart_array,
    render_multi_panel,
    render_simplex_panel,
)
from experiments.common.config import RANDOM_SEARCH_N_ITER
from experiments.rotational_bias.runner import (
    generate_imbalanced_hetero,
    hetero_covs,
)
from threshold_opt import metrics
from threshold_opt.multiclass_strategies import (
    ArgmaxMulticlass,
    FullMatrixPartitioningStrategy,
    MultidimensionalThresholdStrategy,
    ThresholdRatioMulticlass,
    VoronoiTuningMulticlass,
)
from threshold_opt.optimization import RandomSearchOptimizer

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tex" / "figures" / "simplex_regions.tex"

# Scenario parameters (same as experiments/rotational_bias/runner.py).
IMBAL_WEIGHTS = np.array([10, 2, 1], dtype=float)
IMBAL_WEIGHTS = IMBAL_WEIGHTS / IMBAL_WEIGHTS.sum()
HETERO_MEANS = [
    np.array([0, 0, 0]),
    np.array([2, 0, 0]),
    np.array([1, 2, 0]),
]
VIZ_RHO = 0.8

# Shared decision-layer random-search seed for all five panels. Selected
# by a fixed, reproducible rule rather than hand-picked: among the
# experiment's replicate seeds (``seeds.sample_seeds(n_seeds=10,
# master_seed=42)`` in experiments/rotational_bias/runner.py), this is the
# seed whose FullMatrixPartitioning Macro-F1 on this fixed visualization
# dataset is closest to the mean over those ten seeds. The panels are
# therefore a representative replicate of the averaged result reported in
# the rotational-bias table, not a best or worst case. Reproduce the
# choice by fitting FullMatrix at each replicate seed on this dataset and
# taking argmin |F1(seed) - mean F1|. Data generation stays at
# ``default_rng(0)`` so the panels remain mutually comparable.
FIGURE_SEED = 432652533

STRAT_ORDER = [
    "Argmax",
    "ThresholdRatio",
    "MultiDimThreshold",
    "Voronoi",
    "FullMatrix",
]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]


def _strategy_factories(n_iter: int) -> dict:
    return {
        "Argmax": lambda nb, seed: ArgmaxMulticlass(nb),
        "ThresholdRatio": lambda nb, seed: ThresholdRatioMulticlass(
            nb, optimizer=RandomSearchOptimizer(n_iter=n_iter, seed=seed)
        ),
        "MultiDimThreshold": lambda nb, seed: (
            MultidimensionalThresholdStrategy(
                nb,
                optimizer=RandomSearchOptimizer(
                    n_iter=n_iter, bounds=(-1, 1), seed=seed
                ),
            )
        ),
        "Voronoi": lambda nb, seed: VoronoiTuningMulticlass(
            nb, n_iter=n_iter, seed=seed
        ),
        "FullMatrix": lambda nb, seed: FullMatrixPartitioningStrategy(
            nb,
            optimizer=RandomSearchOptimizer(
                n_iter=n_iter, bounds=(-1, 1), seed=seed
            ),
        ),
    }


def _get_params(strategy_name: str, strat) -> np.ndarray:
    if strategy_name == "Argmax":
        return np.array([1.0, 1.0, 1.0])
    if strategy_name == "ThresholdRatio":
        return strat.thresholds
    if strategy_name == "MultiDimThreshold":
        return strat.biases
    if strategy_name == "Voronoi":
        return strat.best_centroids
    if strategy_name == "FullMatrix":
        return strat.partition_matrix
    raise ValueError(strategy_name)


def build(n_iter: int = RANDOM_SEARCH_N_ITER, seed: int = FIGURE_SEED) -> str:
    """Return the TikZ source for the simplex-regions figure.

    *n_iter* is the decision-layer random-search budget; it defaults to
    ``RANDOM_SEARCH_N_ITER`` so the committed figure stays in lockstep with
    the runner. A smaller value is only for fast structural smoke tests and
    will not reproduce the committed figure byte-for-byte.

    *seed* is the shared strategy random-search seed for every panel; it
    defaults to ``FIGURE_SEED`` (see the module-level note on how that
    representative replicate is chosen). Data generation is independent of
    *seed* (fixed at ``default_rng(0)``) so the panels stay comparable.
    """
    covs_viz = hetero_covs(VIZ_RHO)
    rng_viz = np.random.default_rng(0)

    X_train_viz, y_train_viz = generate_imbalanced_hetero(
        HETERO_MEANS, covs_viz, 6_000, IMBAL_WEIGHTS, rng_viz
    )
    X_val_viz, y_val_viz = generate_imbalanced_hetero(
        HETERO_MEANS, covs_viz, 3_000, IMBAL_WEIGHTS, rng_viz
    )
    X_test_viz, y_test_viz = generate_imbalanced_hetero(
        HETERO_MEANS, covs_viz, 2_000, IMBAL_WEIGHTS, rng_viz
    )

    nb_viz = GaussianNB()
    nb_viz.fit(X_train_viz, y_train_viz)
    probs_test_viz = nb_viz.predict_proba(X_test_viz)

    obj_viz = metrics.MetricObjective(
        metric_fn=lambda yt, yp: f1_score(
            yt, yp, average="macro", zero_division=0
        ),
        higher_is_better=True,
    )

    strats_viz = {}
    for name, factory in _strategy_factories(n_iter).items():
        strat = factory(nb_viz, seed)
        strat.fit(X_val_viz, y_val_viz, obj_viz)
        strats_viz[name] = strat

    # Subsample 200 held-out test posteriors for the scatter overlay.
    rng_sub = np.random.default_rng(42)
    sub_idx = rng_sub.choice(len(probs_test_viz), size=200, replace=False)
    sub_x, sub_y = bary_to_cart_array(probs_test_viz[sub_idx])
    data_pts = [
        (float(sub_x[i]), float(sub_y[i]), int(y_test_viz[sub_idx[i]]))
        for i in range(len(sub_idx))
    ]

    panels = []
    for sname, label in zip(STRAT_ORDER, PANEL_LABELS):
        params = _get_params(sname, strats_viz[sname])
        kwargs = dict(title=label, data_points=data_pts, show_medians=False)
        if sname == "Voronoi":
            kwargs["prototype_label_anchors"] = ["west", "west", "south"]
        panels.append(render_simplex_panel(sname, params, **kwargs))

    return render_multi_panel(panels)


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
