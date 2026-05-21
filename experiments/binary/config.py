"""Binary Experiment Configuration."""

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
# 14 binary datasets chosen along four axes:
#   1. Size spread — from 208 (sonar) to 48k (adult).
#   2. Imbalance diversity — IR ranges from 1.0 (madelon) to 42.0
#      (mammography), with good coverage at every order of magnitude.
#   3. Overlap with foundational work — pima_diabetes, german_credit,
#      and sonar appear in Sheng & Ling (2006); adult appears
#      in Elkan (2001).
#   4. Bayes-error spread — a strategy comparison only discriminates
#      where the optimal achievable cost is well above zero, so the
#      suite spans near-separable (wdbc ~1-2% error) through
#      genuinely hard: pima/german (~0.78-0.84 AUC) and the two hard
#      additions jm1 (~0.73 AUC, NASA defect, extends the pc3 line)
#      and madelon (Guyon NeurIPS-2003 designed-hard synthetic).
DATASETS: Sequence[str] = [
    # Small (N < 1k)
    "sonar",  # N=208,   IR=1.1  — CLIMB/UCI; also in Sheng & Ling 2006
    "ionosphere",  # N=351,   IR=1.8  — CLIMB/UCI classic
    "wdbc",  # N=569,   IR=1.7  — breast cancer diagnostic
    "pima_diabetes",  # N=768,   IR=1.9  — "Diabetes" in Sheng & Ling 2006
    "german_credit",  # N=1000,  IR=2.3  — "Credit-g" in Sheng & Ling 2006
    # Medium (1k-10k)
    "pc3",  # N=1563,  IR=8.8  — NASA PROMISE software defects
    "ozone",  # N=2534,  IR=14.8 — high imbalance
    "madelon",  # N=2600,  IR=1.0  — Guyon NeurIPS-2003 designed-hard
    "spambase",  # N=4601,  IR=1.5  — mid-size, near-balanced
    "phoneme",  # N=5404,  IR=2.4  — mid-size numeric
    "jm1",  # N=10885, IR=4.2  — NASA defect; hard, extends pc3
    "mammography",  # N=11183, IR=42.0 — high imbalance
    # Large (N > 10k)
    "bank_marketing",  # N=45k,  IR=7.5  — large, moderate imbalance
    "adult",  # N=48k,  IR=3.2  — workhorse benchmark; Elkan 2001
]

CPU_MODELS = {
    "LGBMClassifier()": lgb.LGBMClassifier(
        random_state=42, verbose=-1, n_jobs=1
    ),
    "LogisticRegression()": linear_model.LogisticRegression(random_state=42),
    "XGBoost": xgb.XGBClassifier(
        random_state=42, verbosity=0, eval_metric="logloss", n_jobs=1
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
    ),
    "TabICL": TabICLv2Classifier(
        random_state=42,
        max_samples=10_000,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ),
}

MODELS = {**CPU_MODELS, **GPU_MODELS}

METRIC_OBJECTIVES: Mapping[str, metrics.OptimizationObjective] = {
    # [[C_00 (TN), C_01 (FP)], [C_10 (FN), C_11 (TP)]]
    "Symmetric Cost": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Asymmetric 10x Cost": np.array([[0.0, 10.0], [1.0, 0.0]]),
    "Asymmetric 100x Cost": np.array([[0.0, 100.0], [1.0, 0.0]]),
    "Non-Zero Diagonal Cost": np.array([[0.8, 2.0], [1.0, 0.5]]),
    "F1": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.f1_score(
            yt, yp, zero_division=0
        ),
        higher_is_better=True,
        vectorized_key="f1",
    ),
    "MCC": metrics.MetricObjective(
        metric_fn=sklearn_metrics.matthews_corrcoef,
        higher_is_better=True,
        vectorized_key="mcc",
    ),
    "Balanced Error": metrics.MetricObjective(
        metric_fn=lambda yt, yp: (
            1.0 - sklearn_metrics.balanced_accuracy_score(yt, yp)
        ),
        higher_is_better=False,
        vectorized_key="balanced_error",
    ),
    # Cost-matrix form of balanced error: cost(y, a) = 1 / (K * pi(y))
    # for a != y, 0 otherwise, with pi from the training labels.
    # Materialized per-task by the pipeline.
    "Balanced Error Cost": metrics.BALANCED_ERROR_COST,
    "F2": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.fbeta_score(
            yt, yp, beta=2, zero_division=0
        ),
        higher_is_better=True,
        vectorized_key="f_beta",
        beta=2.0,
    ),
    "F0.5": metrics.MetricObjective(
        metric_fn=lambda yt, yp: sklearn_metrics.fbeta_score(
            yt, yp, beta=0.5, zero_division=0
        ),
        higher_is_better=True,
        vectorized_key="f_beta",
        beta=0.5,
    ),
    "GMean": metrics.MetricObjective(
        metric_fn=geometric_mean_score,
        higher_is_better=True,
        vectorized_key="gmean",
    ),
}
