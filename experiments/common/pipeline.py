"""Shared experiment pipeline logic."""

import argparse
import contextlib
import datetime
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
from sklearn import metrics as sklearn_metrics
from sklearn import model_selection, utils
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from tqdm import tqdm

from experiments.common import config, data_loader, seeds, thread_limits
from experiments.common.errors import SkippedModelError
from experiments.common.git_metadata import write_parquet_with_metadata
from threshold_opt import base, cv, metrics

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Patches joblib so the tqdm bar advances on task *completion*."""

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


logger = logging.getLogger(__name__)


# SkippedModelError is defined in experiments.common.errors and imported
# above; it stays importable as ``pipeline.SkippedModelError`` (model
# wrappers and tests rely on that path) while also being importable by
# frozen_model without an import cycle.

TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Number of folds used by every CV-variant strategy (multiclass
# ThresholdRatioMulticlass etc., binary EmpiricalThreshold) and by the
# pipeline-level precompute that seeds their shared cache. Must match
# the strategies' own ``cv_folds`` default — all current strategies use
# 5 — and stay in sync with ``threshold_opt.base.CV_RANDOM_STATE`` so
# the pre-populated cache is interchangeable with a strategy's own
# lazy refit.
CV_PRECOMPUTE_FOLDS = 5

# Results are written here. Created on first run. Override with the
# RESULTS_DIR env var to send output to a mounted disk or other location
# (useful on cloud VMs).
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "experiments/results"))


def create_run_dir(prefix: str, resume: Optional[str] = None) -> Path:
    """Creates (or reuses) a timestamped run directory under RESULTS_DIR.

    If *resume* is given, reuses that directory so completed tasks are skipped.
    """
    if resume:
        run_dir = Path(resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        logger.info("Resuming run from %s", run_dir)
        return run_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created run directory: %s", run_dir)
    return run_dir


def _task_result_path(run_dir: Path, dataset: str, seed: int) -> Path:
    """Returns the path for a single (dataset, seed) result file."""
    return run_dir / f"{dataset}_seed{seed}.parquet"


def _parquet_is_readable(path: Path) -> bool:
    """Cheap footer-only integrity probe for a per-task parquet.

    A run interrupted mid-write (now rare thanks to the atomic write in
    :func:`git_metadata.write_parquet_with_metadata`, but still possible
    for files produced by older runs) leaves a file that exists but has
    no valid parquet footer. ``pq.read_metadata`` reads only the footer,
    so this stays cheap even for large parts. Returns ``False`` on any
    read error so the caller re-runs the task instead of skipping it.
    """
    try:
        pq.read_metadata(str(path))
        return True
    except Exception:
        return False


def _save_task_results(
    results: List[config.ExperimentResult],
    path: Path,
    learned_params_column_name: str = "learned_params",
) -> None:
    """Saves results for a single task to a parquet file."""
    if not results:
        return
    data_dicts = [
        {
            **r.config.__dict__,
            learned_params_column_name: r.learned_params,
            "objective_value": r.objective_value,
            "model_roc_auc_split": r.model_roc_auc_split,
            "model_roc_auc_full": r.model_roc_auc_full,
            "model_mean_vertex_distance_split": (
                r.model_mean_vertex_distance_split
            ),
            "model_mean_vertex_distance_full": (
                r.model_mean_vertex_distance_full
            ),
            "model_ece_split": r.model_ece_split,
            "model_ece_full": r.model_ece_full,
            "strategy_fit_seconds": r.strategy_fit_seconds,
        }
        for r in results
    ]
    write_parquet_with_metadata(pd.DataFrame(data_dicts), path)


def merge_run_results(
    run_dir: Path,
    output_filename_prefix: str,
    learned_params_column_name: str = "learned_params",
) -> Path:
    """Merges all per-task parquets in *run_dir* into a single results file."""
    parts = sorted(run_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {run_dir}")
    frames: list[pd.DataFrame] = []
    reference_cols: Optional[set[str]] = None
    reference_part: Optional[Path] = None
    for p in parts:
        frame = pd.read_parquet(p)
        cols = set(frame.columns)
        if reference_cols is None:
            reference_cols, reference_part = cols, p
        elif cols != reference_cols:
            logger.warning(
                "SCHEMA_MISMATCH merging %s: columns differ from %s"
                " (missing here: %s; extra here: %s). Concatenating"
                " anyway -- gap columns become NaN; inspect before"
                " trusting the merged file.",
                p.name,
                reference_part.name if reference_part else "?",
                sorted(reference_cols - cols),
                sorted(cols - reference_cols),
            )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    print(df.head())
    # Keep merged output inside the run directory so dev/test runs
    # don't pollute the shared results folder.
    suffix = run_dir.name.removeprefix(f"run_{output_filename_prefix}_")
    out_path = run_dir / f"{output_filename_prefix}_{suffix}.parquet"
    write_parquet_with_metadata(df, out_path)
    logger.info("Merged %d files -> %s", len(parts), out_path)
    return out_path


def setup_file_logging(run_dir: Path) -> None:
    """Adds a FileHandler to the root logger so logs are persisted.

    Idempotent: repeated calls with the same ``run_dir`` are no-ops. This
    lets joblib/loky workers — which start as fresh Python processes and
    so miss the main-process FileHandler attachment — attach their own
    handler at task entry without duplicating records for the single
    threading-backend case.

    Also sets the root logger level to INFO. Runners call
    ``logging.basicConfig(level=INFO)`` at module top, but loky workers
    import ``pipeline`` (not the runner) so they never run that
    ``basicConfig``. Without the explicit setLevel here, worker INFO
    records get dropped at the logger before reaching any handler and
    the run.log freezes after the main-process header lines.
    """
    log_path = (run_dir / "run.log").resolve()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == log_path:
                    return
            except (OSError, ValueError):
                continue
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)
    logger.info("Logging to %s", log_path)


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    """Adds --n-jobs, --n-seeds, and --resume to an argument parser."""
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 = all cores).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds (default: 3).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help=(
            "Master seed used to sample the per-replicate seed vector "
            "(default: 42). The sampled seeds are persisted to "
            "<run_dir>/seeds.json on first run; --resume reuses that "
            "vector regardless of this flag."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing run directory to resume.",
    )
    # Predict-cache decoupling. The two flags are mutually exclusive:
    # --dump-predictions is Phase A (fit + cache every predict_proba),
    # --from-predictions is Phase B (replay the cache, no model fits).
    # See experiments/common/frozen_model.py.
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--dump-predictions",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Phase A: wrap every model in a recorder, run the normal"
            " pipeline, and write each (dataset, seed, model) prediction"
            " cache under DIR. The normal per-task parquets are still"
            " written too."
        ),
    )
    cache_group.add_argument(
        "--from-predictions",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Phase B: skip model construction, replay the prediction"
            " cache written by --dump-predictions DIR, and run the same"
            " strategy/objective loop. Produces the identical per-task"
            " parquets."
        ),
    )
    # --models and --device are mutually exclusive: --models picks by name
    # from all models, --device picks a whole group. Combining them was a
    # sharp edge (device filtered first, then name filter on that subset,
    # silently producing an empty selection when the names weren't in the
    # chosen group) so argparse now rejects the combination up front.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model keys to run "
            "(default: all models in the study config). "
            "Mutually exclusive with --device."
        ),
    )
    group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help=(
            "Run only CPU or GPU model group "
            "(defined by CPU_MODELS/GPU_MODELS in study config). "
            "Mutually exclusive with --models."
        ),
    )


def filter_models(
    models: Mapping[str, BaseEstimator],
    selection: Optional[str],
) -> Mapping[str, BaseEstimator]:
    """Filters *models* by a comma-separated *selection* string.

    Returns *models* unchanged when *selection* is ``None``.
    Raises ``SystemExit`` if any requested key is not found.
    """
    if selection is None:
        return models
    keys = [k.strip() for k in selection.split(",")]
    missing = [k for k in keys if k not in models]
    if missing:
        raise SystemExit(
            f"Unknown model(s): {missing}. Available: {list(models.keys())}"
        )
    return {k: models[k] for k in keys}


def select_models(
    all_models: Mapping[str, BaseEstimator],
    cpu_models: Mapping[str, BaseEstimator],
    gpu_models: Mapping[str, BaseEstimator],
    device: Optional[str],
    models_filter: Optional[str],
) -> Mapping[str, BaseEstimator]:
    """Pick models based on ``--device`` and ``--models`` flags.

    *device* selects a group (``"cpu"`` or ``"gpu"``). *models_filter*
    picks by name from *all_models*. At most one may be set; the CLI
    enforces that via a mutually exclusive argparse group.
    """
    if device == "cpu":
        return cpu_models
    if device == "gpu":
        return gpu_models
    return filter_models(all_models, models_filter)


def _split_data(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    random_seed: int,
) -> Optional[
    Tuple[
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
    ]
]:
    """Splits the dataset into Train, Val, and Test sets."""
    try:
        X, y = utils.shuffle(X, y, random_state=random_seed)
        X_train_full, X_test, y_train_full, y_test = (
            model_selection.train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                stratify=y,
                random_state=random_seed,
            )
        )
        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X_train_full,
            y_train_full,
            test_size=VAL_SIZE / (1.0 - TEST_SIZE),
            stratify=y_train_full,
            random_state=random_seed,
        )
        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            X_train_full,
            y_train_full,
        )
    except ValueError as e:
        logger.error("Failed to split data: %s", e)
        return None


_SKIPPED = object()  # sentinel: model was intentionally skipped


ECE_N_BINS = 15


def _compute_ece(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    n_classes: int,
    n_bins: int = ECE_N_BINS,
) -> Optional[float]:
    """Binned Expected Calibration Error on held-out predictions.

    Wraps :func:`sklearn.calibration.calibration_curve` for the per-bin
    observed/predicted means and weights each bin by its sample
    fraction. Binary case: predicted class-1 probabilities against the
    binary labels (matching the dissertation formula). Multiclass case:
    top-label ECE, i.e. argmax confidence against argmax correctness
    (Guo et al., 2017).
    """
    try:
        y_true_arr = np.asarray(y_true)
        p = np.asarray(y_score, dtype=np.float64)
        n = y_true_arr.shape[0]
        if n == 0:
            return None
        if n_classes == 2:
            confidence = p[:, 1]
            target = (y_true_arr == 1).astype(np.int64)
        else:
            confidence = p.max(axis=1)
            target = (p.argmax(axis=1) == y_true_arr).astype(np.int64)
        # Clip in case of floating-point overshoot from predict_proba.
        confidence = np.clip(confidence, 0.0, 1.0)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        counts, _ = np.histogram(confidence, bins=edges)
        if not counts.any():
            return None
        prob_true, prob_pred = calibration_curve(
            target, confidence, n_bins=n_bins, strategy="uniform"
        )
        weights = counts[counts > 0] / float(counts.sum())
        return float(np.sum(weights * np.abs(prob_true - prob_pred)))
    except Exception as e:
        logger.debug("ECE computation failed: %s", e)
        return None


def _compute_model_diagnostics(
    fitted_model: Any,
    X_test: npt.ArrayLike,
    y_test: npt.ArrayLike,
    n_classes: int,
    model_name: str,
    split_label: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return ``(roc_auc, mean_vertex_distance, ece)`` for a fitted model.

    ``mean_vertex_distance`` is the mean Euclidean distance from each
    test-set predicted probability vector to its nearest simplex vertex
    (the argmax vertex). Useful as a confidence proxy: zero means every
    prediction is pinned to a vertex, larger values mean the
    predictions sit further into the simplex interior.

    ``ece`` is the binned Expected Calibration Error on the test split
    (see :func:`_compute_ece`).

    Returns ``(None, None, None)`` when predictions cannot be obtained;
    individual entries may still be ``None`` when a specific metric
    cannot be computed.
    """
    if (
        not fitted_model
        or fitted_model is _SKIPPED
        or not hasattr(fitted_model, "predict_proba")
    ):
        return None, None, None

    try:
        y_score = fitted_model.predict_proba(X_test)
    except Exception as e:
        logger.debug(
            "predict_proba unavailable for %s on %s data: %s",
            model_name,
            split_label,
            e,
        )
        return None, None, None

    roc_auc: Optional[float] = None
    try:
        if n_classes == 2:
            roc_auc = float(
                sklearn_metrics.roc_auc_score(y_test, y_score[:, 1])
            )
        else:
            roc_auc = float(
                sklearn_metrics.roc_auc_score(
                    y_test,
                    y_score,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except Exception as e:
        # Stratified splits on small / many-class datasets can drop a
        # class from y_true that still appears in y_score, which breaks
        # roc_auc_score. ROC AUC is optional diagnostic metadata, so
        # debug rather than error.
        logger.debug(
            "ROC AUC unavailable for %s on %s data: %s",
            model_name,
            split_label,
            e,
        )

    # Mean distance from each probability vector to its argmax vertex.
    # distance(p, e_k) = sqrt(1 - 2*p_k + ||p||^2), minimised over k at
    # the argmax, so the closed form is
    #     sqrt(1 - 2*p_max + ||p||^2)
    # with p_max the top class probability.
    try:
        p = np.asarray(y_score, dtype=np.float64)
        p_max = p.max(axis=1)
        sq_norm = np.square(p).sum(axis=1)
        # Numerical noise can push the interior term slightly negative.
        squared_dist = np.maximum(1.0 - 2.0 * p_max + sq_norm, 0.0)
        mean_vertex_distance: Optional[float] = float(
            np.sqrt(squared_dist).mean()
        )
    except Exception as e:
        logger.debug(
            "Vertex-distance unavailable for %s on %s data: %s",
            model_name,
            split_label,
            e,
        )
        mean_vertex_distance = None

    ece = _compute_ece(y_test, y_score, n_classes)

    return roc_auc, mean_vertex_distance, ece


def _train_model(
    model_name: str,
    model_inst: BaseEstimator,
    dataset_name: str,
    X_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
) -> Optional[BaseEstimator]:
    """Clones and fits a model instance.

    Returns the fitted model, ``None`` on real failures, or the
    ``_SKIPPED`` sentinel when the model intentionally opts out
    (e.g. TabPFN on an oversized dataset).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    logger.info("[%s] Training %s...", dataset_name, model_name)
    model_inst = clone(model_inst)
    t_start = time.perf_counter()
    try:
        model_inst.fit(X_train, y_train)
        logger.info(
            "[%s] Trained %s in %.2fs",
            dataset_name,
            model_name,
            time.perf_counter() - t_start,
        )
        return model_inst
    except SkippedModelError as e:
        logger.info("Skipping %s on %s: %s", model_name, dataset_name, e)
        return _SKIPPED
    except Exception as e:
        logger.error(
            "Model %s failed to fit on %s: %s", model_name, dataset_name, e
        )
        return None


def _precompute_cv_fold_results(
    model_template: BaseEstimator,
    X: npt.ArrayLike,
    y: np.ndarray,
    cv_folds: int,
) -> list[base._CVFoldPredictions]:
    """Refit ``model_template`` on each of K training folds and store
    (y_val, predict_proba(X_val)) for every fold.

    CV-variant strategies (multiclass ``ThresholdRatioMulticlass`` etc.,
    binary ``EmpiricalThreshold``) all use the same deterministic
    ``StratifiedKFold(n_splits=cv_folds, shuffle=True,
    random_state=base.CV_RANDOM_STATE)``, so the K refits can be shared
    instead of each strategy recomputing them for every objective.
    ``probs`` is always the 2D ``predict_proba`` output; binary
    consumers extract the positive column via
    :func:`threshold_opt.binary_strategies._positive_class_probs`.
    """
    kf = model_selection.StratifiedKFold(
        n_splits=cv.safe_n_splits(y, cv_folds),
        shuffle=True,
        random_state=base.CV_RANDOM_STATE,
    )
    out: list[base._CVFoldPredictions] = []
    for train_idx, val_idx in kf.split(X, y):
        fold_model = clone(model_template)
        fold_model.fit(X[train_idx], y[train_idx])
        out.append(
            base._CVFoldPredictions(
                y=y[val_idx],
                probs=fold_model.predict_proba(X[val_idx]),
            )
        )
    return out


def _any_strategy_self_tunes_with_cv(
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    model_inst: BaseEstimator,
) -> bool:
    """Probe each factory once to see if the run has any CV-tuning strategy."""
    for strat_factory in strategies.values():
        try:
            instance = strat_factory(model_inst)
        except Exception:
            continue
        if getattr(instance, "self_tunes_with_cv", False):
            return True
    return False


def _evaluate_result(
    dataset_name: str,
    random_seed: int,
    model_name: str,
    strat_name: str,
    objective_name: str,
    strategy: base.DecisionStrategy,
    opt_objective: metrics.OptimizationObjective,
    X_test: npt.ArrayLike,
    y_test: npt.ArrayLike,
    n_classes: int,
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]],
    val_size: float,
    noise_level: float,
    model_roc_auc_split: Optional[float] = None,
    model_roc_auc_full: Optional[float] = None,
    model_mean_vertex_distance_split: Optional[float] = None,
    model_mean_vertex_distance_full: Optional[float] = None,
    model_ece_split: Optional[float] = None,
    model_ece_full: Optional[float] = None,
    strategy_fit_seconds: Optional[float] = None,
) -> Optional[config.ExperimentResult]:
    """Predicts and evaluates a single experiment configuration."""
    try:
        y_pred = strategy.predict(X_test)
    except Exception as e:
        logger.error(
            "Prediction failed for %s / %s: %s", strat_name, objective_name, e
        )
        return None

    objective_value = None
    if isinstance(opt_objective, np.ndarray):
        if opt_objective.shape[0] != n_classes:
            logger.warning(
                "Skipping eval for %s: cost matrix shape %s"
                " mismatches n_classes %s",
                objective_name,
                opt_objective.shape,
                n_classes,
            )
            return None

        objective_value = metrics.calculate_empirical_risk(
            y_test, y_pred, opt_objective
        )

    elif isinstance(opt_objective, metrics.MetricObjective):
        objective_value = opt_objective.metric_fn(y_test, y_pred)
    else:
        logger.error(
            "Unknown objective type for evaluation: %s", objective_name
        )
        return None

    learned_params = None
    if param_extractor:
        learned_params = param_extractor(strategy)

    exp_config = config.ExperimentConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        optimization_objective_name=objective_name,
        strategy_name=strat_name,
        random_seed=random_seed,
        test_size=TEST_SIZE,
        val_size=val_size,
        noise_level=noise_level,
    )

    return config.ExperimentResult(
        config=exp_config,
        learned_params=learned_params,
        objective_value=objective_value,
        model_roc_auc_split=model_roc_auc_split,
        model_roc_auc_full=model_roc_auc_full,
        model_mean_vertex_distance_split=model_mean_vertex_distance_split,
        model_mean_vertex_distance_full=model_mean_vertex_distance_full,
        model_ece_split=model_ece_split,
        model_ece_full=model_ece_full,
        strategy_fit_seconds=strategy_fit_seconds,
    )


def run_experiment_pipeline(
    dataset_name: str,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    random_seed: int,
    models: Mapping[str, BaseEstimator],
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    objectives: Mapping[str, metrics.OptimizationObjective],
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]] = None,
    noise_level: float = 0.0,
) -> List[config.ExperimentResult]:
    """Runs the full experiment pipeline for a given dataset and data."""
    splits = _split_data(X, y, random_seed)
    if not splits:
        return []
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_full,
        y_train_full,
    ) = splits

    logger.info(
        "[%s | seed=%d] Data shapes | Train: %s, Val: %s, Test: %s",
        dataset_name,
        random_seed,
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    logger.info(
        "[%s | seed=%d] Data shapes | Train_full: %s, Test: %s",
        dataset_name,
        random_seed,
        X_train_full.shape,
        X_test.shape,
    )

    n_classes = len(np.unique(y))
    results = []

    for model_name, model_inst in models.items():
        logger.info(
            "[%s | seed=%d] Training %s on split data...",
            dataset_name,
            random_seed,
            model_name,
        )
        fitted_model_split = _train_model(
            model_name, model_inst, dataset_name, X_train, y_train
        )
        logger.info(
            "[%s | seed=%d] Training %s on full data...",
            dataset_name,
            random_seed,
            model_name,
        )
        fitted_model_full = _train_model(
            f"{model_name}_full",
            model_inst,
            dataset_name,
            X_train_full,
            y_train_full,
        )

        model_split_skipped = fitted_model_split is _SKIPPED
        model_full_skipped = fitted_model_full is _SKIPPED

        (
            roc_auc_split,
            vertex_dist_split,
            ece_split,
        ) = _compute_model_diagnostics(
            fitted_model_split,
            X_test,
            y_test,
            n_classes,
            model_name,
            "split",
        )
        (
            roc_auc_full,
            vertex_dist_full,
            ece_full,
        ) = _compute_model_diagnostics(
            fitted_model_full,
            X_test,
            y_test,
            n_classes,
            model_name,
            "full",
        )

        # Share K-fold refits across every CV strategy (and across every
        # objective each strategy evaluates). All CV variants use the
        # same StratifiedKFold(random_state=base.CV_RANDOM_STATE), so
        # one precompute feeds everyone. Gate on: we have a trained full
        # model and at least one CV strategy is registered in this run.
        shared_fold_results: Optional[list[base._CVFoldPredictions]] = None
        if (
            fitted_model_full
            and not model_full_skipped
            and _any_strategy_self_tunes_with_cv(strategies, model_inst)
        ):
            try:
                cv_t0 = time.perf_counter()
                shared_fold_results = _precompute_cv_fold_results(
                    model_template=fitted_model_full,
                    X=X_train_full,
                    y=y_train_full,
                    cv_folds=CV_PRECOMPUTE_FOLDS,
                )
                logger.info(
                    "[%s | seed=%d | %s] CV fold precompute: %d folds in"
                    " %.2fs",
                    dataset_name,
                    random_seed,
                    model_name,
                    len(shared_fold_results),
                    time.perf_counter() - cv_t0,
                )
            except cv.InsufficientCVData as e:
                # A class in y_train_full has too few members for K-fold.
                # Expected on tiny/imbalanced splits; the CV strategies
                # fall back, hit the same condition, and emit the single
                # authoritative CV_SKIP warning in the strategy loop.
                logger.debug(
                    "[%s | seed=%d | %s] CV fold precompute skipped: %s",
                    dataset_name,
                    random_seed,
                    model_name,
                    e,
                )
            except Exception as e:
                logger.warning(
                    "[%s | seed=%d | %s] CV fold precompute failed: %s",
                    dataset_name,
                    random_seed,
                    model_name,
                    e,
                )

        for strat_name, strat_factory in strategies.items():
            strategy_instance_for_check = strat_factory(model_inst)
            try:
                needs_val = strategy_instance_for_check.needs_validation_set
                self_tunes_with_cv = getattr(
                    strategy_instance_for_check, "self_tunes_with_cv", False
                )
            except Exception as e:
                logger.warning(
                    "Could not determine tuning method for strategy"
                    " %s: %s. Skipping.",
                    strat_name,
                    e,
                )
                continue

            if self_tunes_with_cv and needs_val:
                logger.warning(
                    "Strategy %s with self_tunes_with_cv=True must have"
                    " needs_validation_set=False. Skipping.",
                    strat_name,
                )
                continue

            if needs_val:
                if not fitted_model_split or model_split_skipped:
                    if not model_split_skipped:
                        logger.warning(
                            "Skipping strategy %s: needs validation"
                            " but split model failed.",
                            strat_name,
                        )
                    continue
                model_to_use = fitted_model_split
                X_fit, y_fit = X_val, y_val
                val_size_fit = VAL_SIZE
            else:
                if not fitted_model_full or model_full_skipped:
                    if not model_full_skipped:
                        logger.warning(
                            "Skipping strategy %s: does not need"
                            " validation but full model failed.",
                            strat_name,
                        )
                    continue
                model_to_use = fitted_model_full
                X_fit, y_fit = X_train_full, y_train_full
                val_size_fit = 0.0

            strategy = strat_factory(model_to_use)
            if shared_fold_results is not None and self_tunes_with_cv:
                # Hand the precomputed K-fold predictions to the CV
                # strategy so _build_cv_objective_fn (multiclass) /
                # EmpiricalThreshold._fit (binary) skip their internal
                # refit loop on every objective.
                strategy._fold_results = shared_fold_results
            for objective_name, opt_objective in objectives.items():
                if isinstance(opt_objective, metrics.DataDependentCostBuilder):
                    opt_objective = np.asarray(
                        opt_objective.build_fn(y_train_full)
                    )
                elif callable(opt_objective) and not isinstance(
                    opt_objective, metrics.MetricObjective
                ):
                    opt_objective = opt_objective(n_classes)
                    opt_objective = np.asarray(opt_objective)

                strat_t_start = time.perf_counter()
                try:
                    strategy.fit(X_fit, y_fit, opt_objective)
                except cv.InsufficientCVData as e:
                    # Countable, greppable marker: a CV-tuning strategy
                    # could not run K-fold because some class is too
                    # small in this split. The (dataset, model, strategy,
                    # objective) cell is intentionally absent from
                    # results -- audit run.log for CV_SKIP before
                    # trusting completeness.
                    logger.warning(
                        "CV_SKIP [%s | seed=%d | %s] strategy %s /"
                        " objective %s: %s",
                        dataset_name,
                        random_seed,
                        model_name,
                        strat_name,
                        objective_name,
                        e,
                    )
                    continue
                except ValueError as e:
                    if "requires a cost matrix" in str(e):
                        logger.debug(
                            "[%s | seed=%d | %s] %s skipped"
                            " for %s (expected: %s)",
                            dataset_name,
                            random_seed,
                            model_name,
                            strat_name,
                            objective_name,
                            e,
                        )
                    else:
                        logger.warning(
                            "[%s | seed=%d | %s] Skipping"
                            " strategy %s for objective"
                            " %s: %s",
                            dataset_name,
                            random_seed,
                            model_name,
                            strat_name,
                            objective_name,
                            e,
                        )
                    continue
                except (IndexError, RuntimeError) as e:
                    # Safety net for sklearn-internal edge cases (e.g.
                    # CalibratedClassifierCV hitting a fold that drops a
                    # class raises IndexError from _enforce_prediction_order).
                    # Log and skip instead of letting the worker die.
                    logger.warning(
                        "[%s | seed=%d | %s] Skipping strategy %s"
                        " for objective %s (%s): %s",
                        dataset_name,
                        random_seed,
                        model_name,
                        strat_name,
                        objective_name,
                        type(e).__name__,
                        e,
                    )
                    continue

                strat_fit_seconds = time.perf_counter() - strat_t_start
                logger.info(
                    "[%s | seed=%d | %s] Strat-fit %s for %s in %.2fs",
                    dataset_name,
                    random_seed,
                    model_name,
                    strat_name,
                    objective_name,
                    strat_fit_seconds,
                )

                res = _evaluate_result(
                    dataset_name=dataset_name,
                    random_seed=random_seed,
                    model_name=model_name,
                    strat_name=strat_name,
                    objective_name=objective_name,
                    strategy=strategy,
                    opt_objective=opt_objective,
                    X_test=X_test,
                    y_test=y_test,
                    n_classes=n_classes,
                    param_extractor=param_extractor,
                    val_size=val_size_fit,
                    noise_level=noise_level,
                    model_roc_auc_split=roc_auc_split,
                    model_roc_auc_full=roc_auc_full,
                    model_mean_vertex_distance_split=vertex_dist_split,
                    model_mean_vertex_distance_full=vertex_dist_full,
                    model_ece_split=ece_split,
                    model_ece_full=ece_full,
                    strategy_fit_seconds=strat_fit_seconds,
                )
                if res:
                    results.append(res)

    return results


def export_results(
    all_results: List[config.ExperimentResult],
    output_filename_prefix: str,
    learned_params_column_name: str = "learned_params",
) -> None:
    """Serializes results to a timestamped Parquet file in the
    results/ directory."""
    data_dicts = [
        {
            **r.config.__dict__,
            learned_params_column_name: r.learned_params,
            "objective_value": r.objective_value,
            "model_roc_auc_split": r.model_roc_auc_split,
            "model_roc_auc_full": r.model_roc_auc_full,
            "model_mean_vertex_distance_split": (
                r.model_mean_vertex_distance_split
            ),
            "model_mean_vertex_distance_full": (
                r.model_mean_vertex_distance_full
            ),
            "model_ece_split": r.model_ece_split,
            "model_ece_full": r.model_ece_full,
            "strategy_fit_seconds": r.strategy_fit_seconds,
        }
        for r in all_results
    ]
    df = pd.DataFrame(data_dicts)
    print(df.head())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{output_filename_prefix}_{timestamp}.parquet"
    write_parquet_with_metadata(df, out_path)
    logger.info("Results saved to %s", out_path)


def _run_single_seed(
    dataset_name: str,
    random_seed: int,
    models: Mapping[str, BaseEstimator],
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    objectives: Mapping[str, metrics.OptimizationObjective],
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]],
    run_dir: Optional[Path] = None,
    learned_params_column_name: str = "learned_params",
) -> List[config.ExperimentResult]:
    """Loads a dataset and runs the experiment pipeline for a single seed.

    When *run_dir* is set, results are saved incrementally to a per-task
    parquet file and already-completed tasks are skipped.
    """
    thread_limits.cap_torch()

    if run_dir:
        setup_file_logging(run_dir)
        result_path = _task_result_path(run_dir, dataset_name, random_seed)
        if result_path.exists():
            if _parquet_is_readable(result_path):
                logger.info(
                    "Skipping %s seed=%d (already completed)",
                    dataset_name,
                    random_seed,
                )
                return []
            logger.warning(
                "Re-running %s seed=%d: existing per-task parquet %s is"
                " unreadable (likely a truncated write from an"
                " interrupted run); it will be overwritten.",
                dataset_name,
                random_seed,
                result_path.name,
            )

    try:
        X, y = data_loader.load_dataset(dataset_name)
    except Exception as e:
        logger.error("Failed to load %s: %s", dataset_name, e)
        return []
    results = run_experiment_pipeline(
        dataset_name=dataset_name,
        X=X,
        y=y,
        random_seed=random_seed,
        models=models,
        strategies=strategies,
        objectives=objectives,
        param_extractor=param_extractor,
    )

    if run_dir and results:
        _save_task_results(results, result_path, learned_params_column_name)

    return results


def run_experiments_and_export(
    datasets: List[str],
    models: Mapping[str, BaseEstimator],
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    objectives: Mapping[str, metrics.OptimizationObjective],
    output_filename_prefix: str,
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]] = None,
    learned_params_column_name: str = "learned_params",
    n_seeds: int = 10,
    master_seed: int = 42,
    n_jobs: int = -1,
    resume: Optional[str] = None,
) -> None:
    """Runs experiments for multiple datasets and seeds, and exports
    results.

    Replicate seeds are sampled once from ``master_seed`` and persisted
    to ``<run_dir>/seeds.json``. When *resume* points to an existing run
    directory, the persisted seed vector is reused (ignoring the current
    ``n_seeds`` / ``master_seed``) so already-completed (dataset, seed)
    combinations match the still-pending ones.
    """
    run_dir = create_run_dir(output_filename_prefix, resume=resume)
    setup_file_logging(run_dir)

    seed_vec = seeds.load_or_sample_seeds(run_dir, n_seeds, master_seed)
    logger.info("Replicate seeds (master=%d): %s", master_seed, seed_vec)
    tasks = [(ds, s) for ds in datasets for s in seed_vec]
    logger.info(
        "Running %d total tasks (%d datasets x %d seeds).",
        len(tasks),
        len(datasets),
        len(seed_vec),
    )

    t0 = datetime.datetime.now()

    with _tqdm_joblib(tqdm(total=len(tasks), desc="Tasks", unit="task")):
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_run_single_seed)(
                dataset_name=ds,
                random_seed=seed,
                models=models,
                strategies=strategies,
                objectives=objectives,
                param_extractor=param_extractor,
                run_dir=run_dir,
                learned_params_column_name=learned_params_column_name,
            )
            for ds, seed in tasks
        )

    merge_run_results(
        run_dir, output_filename_prefix, learned_params_column_name
    )

    elapsed = datetime.datetime.now() - t0
    logger.info(
        "Finished %d tasks in %s. Results: %s",
        len(tasks),
        str(elapsed).split(".")[0],
        run_dir,
    )


# -- Predict-cache decoupling (Phase A / Phase B) ----------------------------
#
# These orchestrators are thin variants of run_experiments_and_export. They
# reuse every existing helper (create_run_dir, seeds.load_or_sample_seeds,
# run_experiment_pipeline, _save_task_results, merge_run_results) and call
# the *unchanged* run_experiment_pipeline per task, so no pipeline or
# strategy logic changes. The only added work is wrapping models in a
# RecordingModel (Phase A) or FrozenProbaModel (Phase B) and serializing /
# loading the per-(dataset, seed, model) prediction cache.


def _representative_objectives(
    objectives: Mapping[str, metrics.OptimizationObjective],
) -> Mapping[str, metrics.OptimizationObjective]:
    """Minimal objective subset that still makes every strategy execute.

    Phase A exists only to *trigger and record* each model's
    ``predict_proba`` / ``decision_function`` calls. The set of inputs
    those calls receive -- ``X_val`` / ``X_train_full``, ``X_test``, and
    the calibration CV folds of the fit set -- is fixed by the *strategy*
    (its ``needs_validation_set`` flag and whether it wraps the model in
    ``CalibratedClassifierCV``), never by the *objective*: the objective
    only parameterizes the decision-rule optimization, which runs over
    the already-obtained posterior and issues no further model calls. The
    single objective-*type* branch any strategy makes is metric vs. cost
    matrix (the ``"requires a cost matrix"`` skip). So one
    :class:`~metrics.MetricObjective` plus one cost-form objective drives
    every strategy down the same predict path it would take under the
    full set, yielding a complete cache; Phase B then replays that cache
    across *all* objectives. This collapses the long Phase A tail (e.g.
    multiclass: 24 strategies x 13 objectives of discarded optimization
    -> x 2) while staying bit-identical, which is gate-enforced by
    ``frozen_model_test.test_phase_ab_matches_monolithic`` (Phase B runs
    the full objective set and must match the monolithic run to 1e-12;
    any under-recording surfaces as a ``MissingPredictionError``).

    The Phase A results parquet is correspondingly a 2-objective stub --
    it is only a ``--resume`` completion marker, never the scientific
    output (that is Phase B's full-objective parquet).
    """
    metric_item: Optional[tuple] = None
    cost_item: Optional[tuple] = None
    for name, obj in objectives.items():
        if isinstance(obj, metrics.MetricObjective):
            if metric_item is None:
                metric_item = (name, obj)
        elif cost_item is None:
            cost_item = (name, obj)
        if metric_item is not None and cost_item is not None:
            break
    chosen = [it for it in (metric_item, cost_item) if it is not None]
    if not chosen:
        return objectives
    return dict(chosen)


def _run_single_seed_phase_a(
    dataset_name: str,
    random_seed: int,
    models: Mapping[str, BaseEstimator],
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    objectives: Mapping[str, metrics.OptimizationObjective],
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]],
    cache_dir: Path,
    run_dir: Optional[Path] = None,
    learned_params_column_name: str = "learned_params",
) -> List[config.ExperimentResult]:
    """Phase A task: record every predict_proba, then dump the cache.

    Each model gets its own fresh store for this (dataset, seed). The
    pipeline runs over a representative objective subset (see
    :func:`_representative_objectives`) -- enough to drive every
    strategy's model-predict calls -- and the recorded store is
    serialized so Phase B can replay it across the full objective set.
    """
    from experiments.common import frozen_model

    thread_limits.cap_torch()
    if run_dir:
        setup_file_logging(run_dir)
        result_path = _task_result_path(run_dir, dataset_name, random_seed)
        if result_path.exists() and _parquet_is_readable(result_path):
            logger.info(
                "Skipping %s seed=%d (already completed)",
                dataset_name,
                random_seed,
            )
            return []

    try:
        X, y = data_loader.load_dataset(dataset_name)
    except Exception as e:
        logger.error("Failed to load %s: %s", dataset_name, e)
        return []

    stores: dict[str, frozen_model.PredictionStore] = {}
    wrapped_models: dict[str, BaseEstimator] = {}
    for model_name, model_inst in models.items():
        store: frozen_model.PredictionStore = {}
        stores[model_name] = store
        wrapped_models[model_name] = frozen_model.RecordingModel(
            model_inst, store=store
        )

    phase_a_objectives = _representative_objectives(objectives)
    logger.info(
        "[%s | seed=%d] Phase A objective subset: %s"
        " (cache is objective-independent; Phase B replays the full"
        " %d objectives)",
        dataset_name,
        random_seed,
        list(phase_a_objectives),
        len(objectives),
    )

    results = run_experiment_pipeline(
        dataset_name=dataset_name,
        X=X,
        y=y,
        random_seed=random_seed,
        models=wrapped_models,
        strategies=strategies,
        objectives=phase_a_objectives,
        param_extractor=param_extractor,
    )

    for model_name, store in stores.items():
        path = frozen_model.dump_store(
            store, cache_dir, dataset_name, random_seed, model_name
        )
        logger.info(
            "[%s | seed=%d | %s] cached %d predictions -> %s",
            dataset_name,
            random_seed,
            model_name,
            len(store),
            path.name,
        )

    if run_dir and results:
        _save_task_results(results, result_path, learned_params_column_name)
    return results


def _run_single_seed_phase_b(
    dataset_name: str,
    random_seed: int,
    model_names: List[str],
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    objectives: Mapping[str, metrics.OptimizationObjective],
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]],
    cache_dir: Path,
    run_dir: Optional[Path] = None,
    learned_params_column_name: str = "learned_params",
) -> List[config.ExperimentResult]:
    """Phase B task: replay the cached predictions, no model ever fits."""
    from experiments.common import frozen_model

    thread_limits.cap_torch()
    if run_dir:
        setup_file_logging(run_dir)
        result_path = _task_result_path(run_dir, dataset_name, random_seed)
        if result_path.exists() and _parquet_is_readable(result_path):
            logger.info(
                "Skipping %s seed=%d (already completed)",
                dataset_name,
                random_seed,
            )
            return []

    try:
        X, y = data_loader.load_dataset(dataset_name)
    except Exception as e:
        logger.error("Failed to load %s: %s", dataset_name, e)
        return []

    frozen_models: dict[str, BaseEstimator] = {}
    for model_name in model_names:
        try:
            store = frozen_model.load_store(
                cache_dir, dataset_name, random_seed, model_name
            )
        except FileNotFoundError as e:
            logger.warning(
                "[%s | seed=%d | %s] no Phase A cache, skipping model: %s",
                dataset_name,
                random_seed,
                model_name,
                e,
            )
            continue
        frozen_models[model_name] = frozen_model.FrozenProbaModel(store)

    if not frozen_models:
        logger.warning(
            "[%s | seed=%d] no cached models found; nothing to replay.",
            dataset_name,
            random_seed,
        )
        return []

    results = run_experiment_pipeline(
        dataset_name=dataset_name,
        X=X,
        y=y,
        random_seed=random_seed,
        models=frozen_models,
        strategies=strategies,
        objectives=objectives,
        param_extractor=param_extractor,
    )

    if run_dir and results:
        _save_task_results(results, result_path, learned_params_column_name)
    return results


def run_experiments_with_cache(
    datasets: List[str],
    models: Mapping[str, BaseEstimator],
    strategies: Mapping[str, Callable[[BaseEstimator], base.DecisionStrategy]],
    objectives: Mapping[str, metrics.OptimizationObjective],
    output_filename_prefix: str,
    cache_dir: str,
    phase: str,
    param_extractor: Optional[Callable[[base.DecisionStrategy], Any]] = None,
    learned_params_column_name: str = "learned_params",
    n_seeds: int = 10,
    master_seed: int = 42,
    n_jobs: int = -1,
    resume: Optional[str] = None,
) -> None:
    """Run the campaign in Phase A (``"dump"``) or Phase B (``"from"``).

    Same seed sampling, run-directory and parquet handling as
    :func:`run_experiments_and_export`; only the per-task callable
    differs (record-and-dump vs. replay-from-cache).
    """
    cache_path = Path(cache_dir)
    run_dir = create_run_dir(output_filename_prefix, resume=resume)
    setup_file_logging(run_dir)

    seed_vec = seeds.load_or_sample_seeds(run_dir, n_seeds, master_seed)
    logger.info(
        "Predict-cache %s phase | replicate seeds (master=%d): %s",
        phase,
        master_seed,
        seed_vec,
    )
    tasks = [(ds, s) for ds in datasets for s in seed_vec]
    logger.info(
        "Running %d total tasks (%d datasets x %d seeds). Cache dir: %s",
        len(tasks),
        len(datasets),
        len(seed_vec),
        cache_path,
    )

    model_names = list(models.keys())
    t0 = datetime.datetime.now()

    with _tqdm_joblib(tqdm(total=len(tasks), desc="Tasks", unit="task")):
        if phase == "dump":
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_run_single_seed_phase_a)(
                    dataset_name=ds,
                    random_seed=seed,
                    models=models,
                    strategies=strategies,
                    objectives=objectives,
                    param_extractor=param_extractor,
                    cache_dir=cache_path,
                    run_dir=run_dir,
                    learned_params_column_name=learned_params_column_name,
                )
                for ds, seed in tasks
            )
        else:
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_run_single_seed_phase_b)(
                    dataset_name=ds,
                    random_seed=seed,
                    model_names=model_names,
                    strategies=strategies,
                    objectives=objectives,
                    param_extractor=param_extractor,
                    cache_dir=cache_path,
                    run_dir=run_dir,
                    learned_params_column_name=learned_params_column_name,
                )
                for ds, seed in tasks
            )

    merge_run_results(
        run_dir, output_filename_prefix, learned_params_column_name
    )

    elapsed = datetime.datetime.now() - t0
    logger.info(
        "Finished %d %s-phase tasks in %s. Results: %s",
        len(tasks),
        phase,
        str(elapsed).split(".")[0],
        run_dir,
    )
