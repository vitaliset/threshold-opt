"""Configuration and Type Definitions."""

import dataclasses
from typing import Any, Optional

# Shared decision-layer random-search budget (number of sampled
# configurations per fit). Single source of truth for every study that
# tunes a decision rule by random search: the multiclass OpenML campaign
# and the rotational-bias controlled experiment both read this. The
# binary campaign tunes its single threshold by exhaustive grid search
# (G = 301) rather than random search, so it does not consume this value;
# the sensitivity runners do no decision-layer search at all.
#
# Budget = 301: with pure random search, P(best of N draws lands in the
# top q-fraction) = 1 - (1 - q)^N, so N = 301 puts the selected config
# in the top ~1% with ~95% confidence. 1000 was over-search (top 1% at
# 99.99%) at >3x the cost for no practical gain; 301 also matches the
# binary grid resolution G = 301. The optimizer is deterministically
# seeded, so this budget is reproducible run to run.
RANDOM_SEARCH_N_ITER = 301


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for a single experimental run."""

    dataset_name: str
    model_name: str
    optimization_objective_name: str
    strategy_name: str
    random_seed: int
    test_size: float
    val_size: float  # Fraction of total dataset
    noise_level: float = 0.0


@dataclasses.dataclass
class ExperimentResult:
    """Typed container for experiment artifacts."""

    config: ExperimentConfig
    objective_value: float
    learned_params: Any = None
    model_roc_auc_split: Optional[float] = None
    model_roc_auc_full: Optional[float] = None
    # Mean Euclidean distance from the test-set predicted probability
    # vector to its nearest simplex vertex, i.e. the argmax-vertex
    # distance averaged over the test set. A confidence proxy: 0 means
    # the model pins every prediction to a vertex, larger values mean
    # the predictions sit further into the simplex interior.
    model_mean_vertex_distance_split: Optional[float] = None
    model_mean_vertex_distance_full: Optional[float] = None
    # Binned Expected Calibration Error on the test split. Binary case
    # uses positive-class probabilities against binary labels; multiclass
    # case uses top-label ECE (argmax confidence against correctness).
    model_ece_split: Optional[float] = None
    model_ece_full: Optional[float] = None
    strategy_fit_seconds: Optional[float] = None
