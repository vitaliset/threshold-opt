from pathlib import Path

import pytest

from experiments.cloud.inspect_run import inspect_log


def test_classified_pattern_is_not_unclassified(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-04-17 02:41:31 - ERROR - Model LGBM failed to fit on iris: bad\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["model_fit_failed"]) == 1
    assert unclassified == []


def test_strategy_skipped_for_objective_is_classified(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-04-16 04:18:05 - WARNING - [sonar | seed=0 | RealMLP]"
        " Skipping strategy IsotonicAnalytical for objective F1:"
        " IsotonicAnalytical requires a cost matrix\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["skipped_strategy_objective"]) == 1
    assert unclassified == []


def test_unmatched_line_lands_in_unclassified(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-04-17 - INFO - Loading dataset: sonar\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: weird\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert all(not v for v in matches.values())
    assert len(unclassified) == 3


def test_empty_lines_are_skipped(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("\n\n   \n")

    matches, unclassified, _, _ = inspect_log(log)

    assert all(not v for v in matches.values())
    assert unclassified == []


def test_silent_pattern_classifies_without_leaking_to_unclassified(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-04-16 04:17:58 - INFO - Loading dataset: sonar (ID: 40)\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["dataset_loaded"]) == 1
    assert unclassified == []


def test_fit_times_are_captured_and_not_unclassified(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "[iris] Trained RealMLP in 12.34s\n[wine] Trained TabM in 3.10s\n"
    )

    matches, unclassified, fit_times, _ = inspect_log(log)

    assert fit_times == [
        ("iris", "RealMLP", 12.34),
        ("wine", "TabM", 3.10),
    ]
    assert all(not v for v in matches.values())
    assert unclassified == []


def test_strat_fit_times_are_captured_and_not_unclassified(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-04-19 10:00:00 - INFO - [iris | seed=0 | RealMLP]"
        " Strat-fit Voronoi for F1 in 4.56s\n"
        "2026-04-19 10:00:05 - INFO - [wine | seed=1 | LGBM]"
        " Strat-fit MultiDim for Accuracy in 0.78s\n"
    )

    matches, unclassified, _, strat_fit_times = inspect_log(log)

    assert len(strat_fit_times) == 2
    assert strat_fit_times[0].dataset == "iris"
    assert strat_fit_times[0].model == "RealMLP"
    assert strat_fit_times[0].strategy == "Voronoi"
    assert strat_fit_times[0].objective == "F1"
    assert strat_fit_times[0].seconds == 4.56
    assert strat_fit_times[1].strategy == "MultiDim"
    assert all(not v for v in matches.values())
    assert unclassified == []


def test_tqdm_progress_is_classified_silently(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "Tasks:  15%|#####     | 12/80 [03:12<18:04, 16.02s/task]\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["tqdm_progress"]) == 1
    assert unclassified == []


def test_voronoi_sampler_exhausted_is_classified_as_error(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-04-19 12:00:00 - ERROR - [digits | seed=3 | RealMLP] Skipping"
        " strategy Voronoi for F1 (RuntimeError): VoronoiCentroidOptimizer"
        " sequential-d\u00b2 sampler failed to place centroid 7 after 500"
        " retries (K=10).\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["voronoi_sampler_exhausted"]) == 1
    assert unclassified == []


@pytest.mark.parametrize(
    "pattern_key,line",
    [
        ("pl_accelerator_banner", "GPU available: True (cuda), used: True"),
        ("pl_accelerator_banner", "TPU available: False, using: 0 TPU cores"),
        ("pl_accelerator_banner", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]"),
        (
            "pl_litlogger_tip",
            "Tip: For seamless cloud logging and experiment"
            " tracking, try installing [litlogger]",
        ),
        (
            "pl_trainer_fit_stopped",
            "`Trainer.fit` stopped: `max_epochs=256` reached.",
        ),
        (
            "pl_tensor_cores_hint",
            "You are using a CUDA device ('NVIDIA L4') that has Tensor Cores."
            " To properly utilize them, you should set `torch.set_float32...`",
        ),
        (
            "pl_pytree_deprecation",
            "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
            "pytorch_lightning/utilities/_pytree.py:21:"
            " `isinstance(treespec, LeafSpec)` is deprecated",
        ),
        (
            "tabpfn_download",
            "2026-04-19 23:17:40,994 - INFO - Downloading model to"
            " /home/carlo/.cache/tabpfn/tabpfn-v2.5-classifier.ckpt",
        ),
        (
            "tabpfn_download",
            "2026-04-19 23:17:41,213 - INFO - Attempting HuggingFace"
            " download: tabpfn-v2.5-classifier.ckpt",
        ),
        (
            "hf_http_request",
            "2026-04-19 23:17:41,550 - INFO - HTTP Request: HEAD"
            " https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/"
            "tabpfn.ckpt",
        ),
        (
            "hf_anon_warning",
            "Warning: You are sending unauthenticated requests to the HF Hub",
        ),
        (
            "run_dir_created",
            "2026-04-19 23:17:07,908 - INFO - Created run directory:"
            " experiments/results/run_results_binary_20260419_231707",
        ),
        (
            "run_resumed",
            "2026-04-19 23:51:21,376 - INFO - Resuming run from"
            " experiments/results/run_results_binary_20260419_231707",
        ),
        (
            "resume_skip_task",
            "2026-04-20 05:12:01,688 - INFO - Skipping sonar seed=0"
            " (already completed)",
        ),
        (
            "resume_skip_task",
            "2026-04-20 05:19:52,865 - INFO - Skipping pc3 seed=0"
            " (already completed)",
        ),
        (
            "cv_fold_precompute",
            "2026-04-21 15:25:25,834 - INFO - [wdbc | seed=0 | RealMLP]"
            " CV fold precompute: 5 folds in 12.39s",
        ),
        (
            "cv_fold_precompute",
            "2026-04-21 02:25:39,978 - INFO - [pendigits | seed=0 | RealMLP]"
            " CV fold precompute: 5 folds in 215.54s",
        ),
    ],
)
def test_gpu_noise_patterns_are_classified_silently(
    tmp_path: Path, pattern_key: str, line: str
) -> None:
    log = tmp_path / "run.log"
    log.write_text(line + "\n")

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches[pattern_key]) == 1, (
        f"expected line to match {pattern_key}; matches={matches}"
    )
    assert unclassified == []


@pytest.mark.parametrize(
    "pattern_key,line",
    [
        (
            "sklearn_ypred_classes_header",
            "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
            "sklearn/metrics/_classification.py:2924: UserWarning:"
            " y_pred contains classes not in y_true",
        ),
        (
            "sklearn_ypred_classes_body",
            '  warnings.warn("y_pred contains classes not in y_true")',
        ),
        (
            "sklearn_calibration_divide_header",
            "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
            "sklearn/calibration.py:533: RuntimeWarning:"
            " invalid value encountered in divide",
        ),
        (
            "sklearn_calibration_divide_body",
            "  mean_proba /= len(self.calibrated_classifiers_)",
        ),
        (
            "sklearn_cv_runtime_warning",
            "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
            "sklearn/model_selection/_validation.py:1352: RuntimeWarning:"
            " Mean of empty slice",
        ),
        (
            "sklearn_logistic_convergence_header",
            "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
            "sklearn/linear_model/_logistic.py:406: ConvergenceWarning:"
            " lbfgs failed to converge (status=1):",
        ),
        (
            "sklearn_logistic_convergence_body",
            "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
        ),
        (
            "sklearn_logistic_convergence_body",
            "Increase the number of iterations to improve the convergence"
            " (max_iter=100).",
        ),
        (
            "sklearn_logistic_convergence_body",
            "You might also want to scale the data as shown in:",
        ),
        (
            "sklearn_logistic_convergence_body",
            "https://scikit-learn.org/stable/modules/preprocessing.html",
        ),
        (
            "sklearn_logistic_convergence_body",
            "Please also refer to the documentation for alternative"
            " solver options:",
        ),
        (
            "sklearn_logistic_convergence_body",
            "https://scikit-learn.org/stable/modules/"
            "linear_model.html#logistic-regression",
        ),
        (
            "sklearn_logistic_convergence_body",
            "  n_iter_i = _check_optimize_result(",
        ),
        (
            "sklearn_stratkfold_small_class",
            "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
            "sklearn/model_selection/_split.py:813: UserWarning:"
            " The least populated class in y has only 2 members,"
            " which is less than n_splits=5.",
        ),
        ("warnings_warn_continuation", "warnings.warn("),
        ("warnings_warn_continuation", "  warn("),
        (
            "merge_summary",
            "2026-04-21 15:56:58,411 - INFO - Merged 12 files ->"
            " experiments/results/run_results_multiclass_20260421_153202/"
            "results_multiclass_20260421_153202.parquet",
        ),
        (
            "run_finished",
            "2026-04-21 15:56:58,412 - INFO - Finished 12 tasks in 0:24:55."
            " Results: experiments/results/"
            "run_results_multiclass_20260421_153202",
        ),
        (
            "results_saved",
            "2026-04-21 15:56:58,500 - INFO - Results saved to"
            " experiments/results/results_binary_20260421_155658.parquet",
        ),
        (
            "pandas_head_truncation",
            "  dataset_name        model_name  ... model_roc_auc_full"
            " strategy_fit_seconds",
        ),
        (
            "pandas_head_truncation",
            "0          car  LGBMClassifier()  ...           0.999938"
            "             0.001046",
        ),
        ("pandas_shape_footer", "[5 rows x 13 columns]"),
    ],
)
def test_sklearn_noise_patterns_are_classified_silently(
    tmp_path: Path, pattern_key: str, line: str
) -> None:
    log = tmp_path / "run.log"
    log.write_text(line + "\n")

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches[pattern_key]) == 1, (
        f"expected line to match {pattern_key}; matches={matches}"
    )
    assert unclassified == []


def test_joblib_worker_stopped_is_classified_as_warning(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "/home/carlo/mestrado-draft/.venv/lib/python3.12/site-packages/"
        "joblib/externals/loky/process_executor.py:782: UserWarning: A"
        " worker stopped while some jobs were given to the executor. This"
        " can be caused by a too short worker timeout or by a memory leak.\n"
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["joblib_worker_stopped"]) == 1
    assert unclassified == []


def test_joblib_remote_traceback_is_classified_as_error(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        'joblib.externals.loky.process_executor._RemoteTraceback: """\n'
    )

    matches, unclassified, _, _ = inspect_log(log)

    assert len(matches["joblib_remote_traceback"]) == 1
    assert unclassified == []
