# threshold_opt

Threshold optimization for binary and multiclass classification.

Given a trained classifier and a cost matrix (or metric objective), `threshold_opt` finds the decision rule that minimizes expected cost. It supports binary and multiclass problems, with strategies ranging from closed-form analytical solutions to empirical search and calibration-based approaches.

This library is part of a [master's dissertation](https://github.com/vitaliset/threshold-opt/blob/main/tex/carlo_master.tex) on multiclass threshold optimization. The full experiment pipeline lives in [`experiments/`](https://github.com/vitaliset/threshold-opt/blob/main/experiments/README.md).

## Install

```bash
pip install threshold-opt
```

The package is published as `threshold-opt` and imported as `threshold_opt`.

For development (includes experiment dependencies):

```bash
git clone https://github.com/vitaliset/threshold-opt.git
cd threshold-opt
uv sync
```

## Quick example

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from threshold_opt import binary_strategies

# Train any sklearn-compatible classifier
X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# Define a cost matrix: missing cancer (FN) costs 10x more than a false alarm (FP)
cost_matrix = np.array([
    [0.0, 1.0],   # TN=0, FP=1
    [10.0, 0.0],  # FN=10, TP=0
])

# Find the optimal threshold analytically (no validation data needed)
strategy = binary_strategies.AnalyticalThreshold(model)
strategy.fit(np.array([]), np.array([]), cost_matrix)
y_pred = strategy.predict(X_val)

# Or search empirically on a validation set
strategy = binary_strategies.EmpiricalThreshold(model)
strategy.fit(X_val, y_val, cost_matrix)
y_pred = strategy.predict(X_val)
```

## What's inside

| Module | Description |
|--------|-------------|
| `binary_strategies` | 11 strategies for binary classification: default, analytical, empirical search, and calibration variants (isotonic, Platt, temperature, Dirichlet) |
| `multiclass_strategies` | 17 strategies for multiclass: argmax, risk minimization, threshold ratios, multidimensional thresholds, Voronoi tuning, sequential selection, risk partitioning, linear scaling, full-matrix partitioning, and calibration variants (isotonic, Platt, temperature, Dirichlet) |
| `calibration` | `DirichletCalibrator` for post-hoc probability calibration in log-space |
| `metrics` | Cost matrix utilities, metric objectives (`MetricObjective`), vectorized scoring, analytical threshold derivation |
| `optimization` | Grid search and random search optimizers for strategy parameter tuning |
| `base` | `DecisionStrategy` abstract base class |

All strategies follow a unified interface: `fit(X, y, objective)` then `predict(X)`. Objectives can be cost matrices (numpy arrays), `MetricObjective` instances, or callable factories.

### Public API

`threshold_opt` ships its public API as importable submodules; there is no flat top-level namespace re-export. Stable surface:

- **Strategy contract** ([`base.DecisionStrategy`](https://github.com/vitaliset/threshold-opt/blob/main/src/threshold_opt/base.py)). All strategies subclass `DecisionStrategy` and expose:
    - `fit(X, y, optimization_objective)` — fit the decision layer on validation data and return `self`.
    - `predict(X)` — emit class predictions.
    - `needs_validation_set: bool` — `True` if the strategy consumes a held-out validation split (calibrators, single-split tuners), `False` otherwise (default, analytical, CV-tuners).
- **Optimization objectives** ([`metrics.OptimizationObjective`](https://github.com/vitaliset/threshold-opt/blob/main/src/threshold_opt/metrics.py)). Either a numpy `(K, K)` cost matrix (cell `[i, j]` = cost of predicting `j` when the truth is `i`), or a `metrics.MetricObjective(metric_fn, higher_is_better, ...)` wrapping any `(y_true, y_pred) -> float` callable. `MetricObjective.vectorized_key` is the optional fast path used by binary strategies; binary metric objectives that omit it still work via a per-threshold loop.
- **Calibrator** ([`calibration.DirichletCalibrator`](https://github.com/vitaliset/threshold-opt/blob/main/src/threshold_opt/calibration.py)). Standalone `fit(probs, y)` / `predict_proba(probs)` calibrator with optional ODIR penalties (`reg_lambda`, `reg_mu`). Used by `DirichletCalibrationArgmax` / `DirichletCalibrationAnalytical` and freely composable.
- **Optimizers** ([`optimization.RandomSearchOptimizer`, `GridSearchOptimizer`](https://github.com/vitaliset/threshold-opt/blob/main/src/threshold_opt/optimization.py)). Pluggable parameter optimizers for tuning strategies; subclass `optimization.ParameterOptimizer` to add new ones.
- **Shared CV seed** ([`base.CV_RANDOM_STATE`](https://github.com/vitaliset/threshold-opt/blob/main/src/threshold_opt/base.py)). Every K-fold loop in the library uses this constant, so external precompute caches that key on it stay in sync with strategies' lazy refits.

Internal symbols (prefixed with `_`, e.g. `base._CVFoldPredictions`, `binary_strategies._positive_class_probs`) are not part of the public API and may change without notice.

## Dissertation and experiments

This repo also contains the experiment pipeline and LaTeX source for the dissertation. See [`experiments/README.md`](https://github.com/vitaliset/threshold-opt/blob/main/experiments/README.md) for how to run experiments, generate tables, and build the PDF.

## Development

```bash
make test      # run pytest
make lint      # ruff + ruff-format + chktex
make help      # all available targets
```

## Releasing

The package is uploaded to [PyPI](https://pypi.org/project/threshold-opt/) by [`.github/workflows/publish.yml`](https://github.com/vitaliset/threshold-opt/blob/main/.github/workflows/publish.yml), through [trusted publishing](https://docs.pypi.org/trusted-publishers/). To release a new version:

1. Set `version` in `pyproject.toml`.
2. Publish a GitHub release whose tag is `v` followed by that version (`v0.1.0` for version `0.1.0`).

The workflow checks that the tag matches the version, builds the package and uploads it. Releases with any other tag publish nothing.

## Project structure

```
src/threshold_opt/     Library: strategies, calibration, metrics, optimization
experiments/           Experiment pipeline, analysis, cloud scripts
tex/                   LaTeX dissertation source
lean/                  Lean 4 + Mathlib formalization of the dissertation's proofs
```
