"""Multiclass Experiment Configuration."""

from collections.abc import Mapping, Sequence

import catboost
import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb
from imblearn.metrics import geometric_mean_score
from sklearn import linear_model
from sklearn import metrics as sklearn_metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from experiments.common import config as common_config
from experiments.common.model_wrappers import (
    TabICLv2Classifier,
    TabMClassifier,
    TabPFN25SizeFiltered,
)
from threshold_opt import metrics

# Re-export generic types for convenience
ExperimentConfig = common_config.ExperimentConfig
ExperimentResult = common_config.ExperimentResult

# -- datasets -----------------------------------------------------------------
# 13 multiclass datasets chosen for coverage across size, number of
# classes (K), imbalance, and *ranking difficulty*. OVR-weighted ROC AUC
# is structurally inflated (one-vs-rest ranking stays easy even when the
# argmax decision is hard), so the classic benchmarks all sit at
# ~0.99-1.0 AUC and a decision-strategy comparison cannot discriminate on
# them. The suite was rebuilt around *measured* RF/LGBM OVR-weighted AUC:
# all but iris land in ~0.65-0.92, with cmc the hardest (~0.67) and a
# dense cluster in the 0.80-0.88 band (abalone, yeast, wine_quality_red,
# fot_proving, jannis). Only iris (~1.0) is kept, as a deliberate
# near-separable baseline (does a strategy avoid breaking on trivial
# data?). Dropped over successive revisions for being near-separable or
# redundant: wine, car, pendigits, then digits, segment, vowel, letter,
# then cardiotocography and artificial_characters (the 1-seed smoke
# measured them at OVR-AUC 1.000 / 0.996 -- their labels are
# near-deterministic from features, so they do not discriminate
# strategies despite moderate accuracy).
#
# K range: 3-10. Size range: 150-83k. New additions require
# N >= 500. abalone and wine_quality_red also restore an ordinal-label
# motif (age groups / quality scores) for the Ordinal/Quadratic cost
# objectives, replacing the dropped car. O'Brien, Gupta & Gray (2008)
# overlap now only via iris (their car/wine/segment overlaps were
# dropped; RiskPartitioning still implements O'Brien's method regardless
# of dataset overlap). O'Brien 2008 datasets excluded for N < 500:
# glass (214), ecoli (336), dermatology (366); also Bridges 2, Horse
# colic, Audiology, Flag, Flare2. robert (OpenML 41165) was considered
# but is unfetchable: OpenML serves an ARFF whose md5 fails its own
# metadata checksum, so sklearn rejects it for every user.
DATASETS: Sequence[str] = [
    # Small (N < 1k)
    "iris",  # N=150,   K=3   -- O'Brien 2008; near-separable baseline
    "vehicle",  # N=846,   K=4   -- AUC ~0.91
    # Medium (1k-10k)
    "cmc",  # N=1473,  K=3   -- hardest; OVR-AUC ~0.67
    "yeast",  # N=1484,  K=10  -- classic; OVR-AUC ~0.83 (IR~93)
    "wine_quality_red",  # N=1599,  K=6   -- ordinal; OVR-AUC ~0.86
    "abalone",  # N=4177,  K=3   -- balanced ordinal; OVR-AUC ~0.83
    "fot_proving",  # N=6118,  K=6   -- AMLB; OVR-AUC ~0.86
    "satimage",  # N=6435,  K=6   -- AUC ~0.99
    "gesture_phase",  # N=9873,  K=5   -- AMLB; OVR-AUC ~0.9
    # Large (N > 10k)
    "volkert",  # N=58310, K=10  -- AMLB; OVR-AUC ~0.91
    "jannis",  # N=83526, K=4   -- AMLB; OVR-AUC ~0.85
]

CPU_MODELS = {
    "LGBMClassifier()": lgb.LGBMClassifier(
        random_state=42, verbose=-1, n_jobs=1
    ),
    "LogisticRegression()": linear_model.LogisticRegression(random_state=42),
    "XGBoost": xgb.XGBClassifier(
        random_state=42, verbosity=0, eval_metric="mlogloss", n_jobs=1
    ),
    "CatBoost": catboost.CatBoostClassifier(
        random_state=42, verbose=0, thread_count=1
    ),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=1),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=1),
    "GaussianNB": GaussianNB(),
}

GPU_MODELS = {
    "TabM": TabMClassifier(random_state=42),
    # TabPFN-2.5 (the only retained TabPFN-family model; the "25" mark
    # is dropped from the key now that there is nothing to disambiguate
    # it from).
    "TabPFN": TabPFN25SizeFiltered(
        random_state=42,
        max_samples=10_000,
        max_features=2_000,
        max_classes=10,
    ),
    "TabICL": TabICLv2Classifier(
        random_state=42,
        max_samples=10_000,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ),
}

MODELS = {**CPU_MODELS, **GPU_MODELS}


def create_symmetric_cost_matrix(n_classes: int) -> np.ndarray:
    """Creates a symmetric (uniform) cost matrix (0 on diag, 1 elsewhere)."""
    cm = np.ones((n_classes, n_classes))
    np.fill_diagonal(cm, 0)
    return cm


def create_mild_asymmetry_cost_matrix(n_classes: int) -> np.ndarray:
    """Creates a matrix where misclassifying the last class is costly (10x)."""
    cm = np.ones((n_classes, n_classes))
    np.fill_diagonal(cm, 0)
    cm[n_classes - 1, :] = 10.0
    cm[n_classes - 1, n_classes - 1] = 0.0
    return cm


def create_severe_asymmetry_cost_matrix(n_classes: int) -> np.ndarray:
    """Creates a matrix where misclassifying the last class is very
    costly (100x)."""
    cm = np.ones((n_classes, n_classes))
    np.fill_diagonal(cm, 0)
    cm[n_classes - 1, :] = 100.0
    cm[n_classes - 1, n_classes - 1] = 0.0
    return cm


def create_custom_cost_matrix(n_classes: int) -> np.ndarray:
    """Creates a custom cost matrix, preserving legacy behavior for Iris."""
    if n_classes == 3:
        return np.array([[0, 1, 1], [1, 0, 5], [1, 5, 0]])

    return create_mild_asymmetry_cost_matrix(n_classes)


def create_ordinal_cost_matrix(n_classes: int) -> np.ndarray:
    """Linear distance cost: ``cost[i, j] = |i - j|``, diagonal 0.

    Penalises predictions that are farther from the true class. Relevant
    when class labels have an implicit ordering (``car`` quality levels,
    credit grades). See O'Brien, Gupta & Gray (2008) for the family of
    ordinal misclassification costs.
    """
    i = np.arange(n_classes)
    return np.abs(i[:, None] - i[None, :]).astype(float)


def create_quadratic_cost_matrix(n_classes: int) -> np.ndarray:
    """Quadratic distance cost: ``cost[i, j] = (i - j)**2``, diagonal 0.

    Same ordinal motivation as :func:`create_ordinal_cost_matrix` but with
    a steeper penalty on distant errors. Common in medical triage scoring
    and ordinal regression literature.
    """
    i = np.arange(n_classes)
    return ((i[:, None] - i[None, :]) ** 2).astype(float)


METRIC_OBJECTIVES: Mapping[str, metrics.OptimizationObjective] = {
    "F1 Macro": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.f1_score(
            yt, yp, average="macro", zero_division=0
        ),
        higher_is_better=True,
    ),
    "F2 Macro": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.fbeta_score(
            yt, yp, beta=2, average="macro", zero_division=0
        ),
        higher_is_better=True,
    ),
    "F0.5 Macro": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.fbeta_score(
            yt, yp, beta=0.5, average="macro", zero_division=0
        ),
        higher_is_better=True,
    ),
    "Balanced Error": metrics.MetricObjective(
        metric_fn=lambda yt, yp: (
            1.0 - sklearn_metrics.balanced_accuracy_score(yt, yp)
        ),
        higher_is_better=False,
    ),
    # Cost-matrix form of balanced error: cost(y, a) = 1 / (K * pi(y))
    # for a != y, 0 otherwise, with pi from the training labels.
    # Materialized per-task by the pipeline.
    "Balanced Error Cost": metrics.BALANCED_ERROR_COST,
    "MCC": metrics.MetricObjective(
        metric_fn=sklearn_metrics.matthews_corrcoef,
        higher_is_better=True,
    ),
    "GMean": metrics.MetricObjective(
        metric_fn=lambda yt, yp: geometric_mean_score(yt, yp, average="macro"),
        higher_is_better=True,
    ),
    "Symmetric Cost": create_symmetric_cost_matrix,
    "Asymmetric 10x Cost": create_mild_asymmetry_cost_matrix,
    "Asymmetric 100x Cost": create_severe_asymmetry_cost_matrix,
    "Custom Cost": create_custom_cost_matrix,
    "Ordinal Cost": create_ordinal_cost_matrix,
    "Quadratic Cost": create_quadratic_cost_matrix,
}
