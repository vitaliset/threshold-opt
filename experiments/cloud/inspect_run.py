"""Inspect a run log for known error/warning patterns.

Usage:
    uv run python -m experiments.cloud.inspect_run run-multiclass.log
    uv run python -m experiments.cloud.inspect_run \
        experiments/results/run_*/run.log
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Level = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(frozen=True)
class LogPattern:
    regex: str
    level: Level
    description: str
    silent: bool = False


MAX_UNIQUE_MESSAGES = 200
SLOWEST_FITS_TO_SHOW = 10

FIT_TIME_PATTERN = re.compile(
    r"\[(?P<dataset>.+?)\] Trained (?P<model>.+?) in (?P<seconds>[\d.]+)s"
)
STRAT_FIT_TIME_PATTERN = re.compile(
    r"\[(?P<dataset>.+?) \| seed=(?P<seed>\d+) \| (?P<model>.+?)\]"
    r" Strat-fit (?P<strategy>.+?) for (?P<objective>.+?)"
    r" in (?P<seconds>[\d.]+)s"
)

PATTERNS: dict[str, LogPattern] = {
    # Silent patterns: classified but not printed in the summary. Keeps routine
    # pipeline-progress lines out of the "unclassified" triage queue without
    # cluttering the output. Promote to visible (flip silent=False) if you
    # ever need to see them.
    "logging_header": LogPattern(
        regex=r"Logging to (.+)",
        level="INFO",
        description="Run logging header",
        silent=True,
    ),
    "task_count_header": LogPattern(
        regex=r"Running (\d+) total tasks",
        level="INFO",
        description="Task count header",
        silent=True,
    ),
    "dataset_loaded": LogPattern(
        regex=r"Loading dataset: (.+?) \(ID: (\d+)\)",
        level="INFO",
        description="Dataset loaded",
        silent=True,
    ),
    "data_shapes": LogPattern(
        regex=r"\[(.+?) \| seed=(\d+)\] Data shapes \|",
        level="INFO",
        description="Per-seed data shapes",
        silent=True,
    ),
    "training_started_per_split": LogPattern(
        regex=(
            r"\[(.+?) \| seed=(\d+)\] Training (.+?) on"
            r" (split|full) data\.\.\."
        ),
        level="INFO",
        description="Training started (split/full)",
        silent=True,
    ),
    "training_started": LogPattern(
        regex=r"\[(.+?)\] Training (.+?)\.\.\.",
        level="INFO",
        description="Training started (per-model)",
        silent=True,
    ),
    "model_fit_failed": LogPattern(
        regex=r"Model (.+?) failed to fit on (.+?): (.+)",
        level="ERROR",
        description="Model training error",
    ),
    "prediction_failed": LogPattern(
        regex=r"Prediction failed for (.+?) / (.+?): (.+)",
        level="ERROR",
        description="Strategy prediction error",
    ),
    "skipped_model": LogPattern(
        regex=r"Skipping (.+?) on (.+?): (.+)",
        level="INFO",
        description="Model intentionally skipped",
    ),
    "skipped_strategy_model_failed": LogPattern(
        regex=(
            r"Skipping strategy (.+?): "
            r"(does not need validation but full model failed|"
            r"needs validation but split model failed)"
        ),
        level="WARNING",
        description="Strategy skipped (model failed)",
    ),
    "skipped_strategy_objective": LogPattern(
        regex=r"Skipping strategy (.+?) for objective (.+?): (.+)",
        level="WARNING",
        description="Strategy skipped (incompatible with objective)",
    ),
    "device_error": LogPattern(
        regex=r'Unknown device name "(.+?)"',
        level="ERROR",
        description="Device configuration error",
    ),
    "frozen_estimator": LogPattern(
        regex=r"FrozenEstimator should either be a classifier",
        level="ERROR",
        description="FrozenEstimator classifier tag issue",
    ),
    "voronoi_sampler_exhausted": LogPattern(
        regex=(
            r"VoronoiCentroidOptimizer sequential-d.{1,3}"
            r" sampler failed to place centroid"
        ),
        level="ERROR",
        description="Voronoi sequential-d² sampler exhausted retries",
    ),
    "tqdm_progress": LogPattern(
        regex=r"^Tasks:\s+\d+%\|",
        level="INFO",
        description="tqdm progress bar",
        silent=True,
    ),
    "joblib_remote_traceback": LogPattern(
        regex=r"joblib\.externals\.loky\.process_executor\._RemoteTraceback",
        level="ERROR",
        description="joblib remote worker traceback",
    ),
    # PyTorch Lightning banner — prints per Trainer.fit (TabM).
    "pl_accelerator_banner": LogPattern(
        regex=(
            r"^(GPU available: .+, used: .+"
            r"|TPU available: .+, using: \d+ TPU cores"
            r"|LOCAL_RANK: \d+ - CUDA_VISIBLE_DEVICES: \[.*\])$"
        ),
        level="INFO",
        description="PyTorch Lightning accelerator banner",
        silent=True,
    ),
    "pl_litlogger_tip": LogPattern(
        regex=r"Tip: For seamless cloud logging and experiment tracking",
        level="INFO",
        description="PyTorch Lightning litlogger tip",
        silent=True,
    ),
    "pl_trainer_fit_stopped": LogPattern(
        regex=r"`Trainer\.fit` stopped: `max_epochs=\d+` reached\.",
        level="INFO",
        description="PyTorch Lightning max_epochs reached",
        silent=True,
    ),
    "pl_tensor_cores_hint": LogPattern(
        regex=r"You are using a CUDA device .+ has Tensor Cores",
        level="INFO",
        description="PyTorch Lightning tensor-cores hint",
        silent=True,
    ),
    "pl_pytree_deprecation": LogPattern(
        regex=(
            r"_pytree\.py:\d+: `isinstance\(treespec, LeafSpec\)`"
            r" is deprecated"
        ),
        level="INFO",
        description="PyTorch Lightning _pytree deprecation",
        silent=True,
    ),
    # TabPFN pulls weights from HuggingFace on first use per worker.
    "tabpfn_download": LogPattern(
        regex=(
            r"(Downloading model to |Attempting HuggingFace download:"
            r"|Successfully downloaded to )"
        ),
        level="INFO",
        description="TabPFN HuggingFace download",
        silent=True,
    ),
    "hf_http_request": LogPattern(
        regex=r"HTTP Request: (HEAD|GET|POST) https://(huggingface\.co|cdn-lfs)",
        level="INFO",
        description="HuggingFace HTTP request",
        silent=True,
    ),
    "hf_anon_warning": LogPattern(
        regex=r"You are sending unauthenticated requests to the HF Hub",
        level="INFO",
        description="HuggingFace unauthenticated-request notice",
        silent=True,
    ),
    "run_dir_created": LogPattern(
        regex=r"Created run directory: ",
        level="INFO",
        description="Run directory created",
        silent=True,
    ),
    "run_resumed": LogPattern(
        regex=r"Resuming run from ",
        level="INFO",
        description="Run resumed from existing dir",
        silent=True,
    ),
    "resume_skip_task": LogPattern(
        regex=r"Skipping \S+ seed=\d+ \(already completed\)",
        level="INFO",
        description="Task skipped on resume (already completed)",
        silent=True,
    ),
    "cv_fold_precompute": LogPattern(
        regex=r"CV fold precompute: \d+ folds in [\d.]+s",
        level="INFO",
        description=(
            "Pipeline-level CV fold precompute (shared across CV strategies)"
        ),
        silent=True,
    ),
    # sklearn stderr noise on small / imbalanced multiclass splits. Each
    # warning prints a header line (`<path>:<lineno>: <Category>: <msg>`)
    # followed by a 1-line code excerpt; we silence both so they don't
    # crowd the unclassified triage bucket. Promote to visible if these
    # ever signal something we actually need to act on.
    "sklearn_ypred_classes_header": LogPattern(
        regex=(
            r"sklearn/metrics/_classification\.py:\d+: UserWarning:"
            r" y_pred contains classes not in y_true"
        ),
        level="INFO",
        description=(
            "sklearn y_pred-contains-unseen-classes warning"
            " (benign on imbalanced folds)"
        ),
        silent=True,
    ),
    "sklearn_ypred_classes_body": LogPattern(
        regex=r'warnings\.warn\("y_pred contains classes not in y_true"\)',
        level="INFO",
        description="sklearn y_pred-contains-unseen-classes continuation",
        silent=True,
    ),
    "sklearn_calibration_divide_header": LogPattern(
        regex=(
            r"sklearn/calibration\.py:\d+: RuntimeWarning:"
            r" invalid value encountered in divide"
        ),
        level="INFO",
        description=(
            "sklearn CalibratedClassifierCV divide warning"
            " (class absent from every fold)"
        ),
        silent=True,
    ),
    "sklearn_calibration_divide_body": LogPattern(
        regex=r"mean_proba /= len\(self\.calibrated_classifiers_\)",
        level="INFO",
        description="sklearn CalibratedClassifierCV divide continuation",
        silent=True,
    ),
    "sklearn_cv_runtime_warning": LogPattern(
        regex=r"sklearn/model_selection/_validation\.py:\d+: RuntimeWarning",
        level="INFO",
        description=(
            "sklearn cross-validation RuntimeWarning"
            " (missing class in scoring fold)"
        ),
        silent=True,
    ),
    "sklearn_logistic_convergence_header": LogPattern(
        regex=(r"sklearn/linear_model/_logistic\.py:\d+: ConvergenceWarning"),
        level="INFO",
        description=(
            "sklearn LogisticRegression ConvergenceWarning (max_iter hit)"
        ),
        silent=True,
    ),
    "sklearn_logistic_convergence_body": LogPattern(
        regex=(
            r"(STOP: TOTAL NO\. OF ITERATIONS REACHED LIMIT"
            r"|Increase the number of iterations to improve the convergence"
            r"|You might also want to scale the data as shown in:"
            r"|https://scikit-learn\.org/stable/modules/preprocessing\.html"
            r"|Please also refer to the documentation for"
            r" alternative solver options:"
            r"|https://scikit-learn\.org/stable/modules/"
            r"linear_model\.html#logistic-regression"
            r"|n_iter_i = _check_optimize_result\()"
        ),
        level="INFO",
        description="sklearn LogisticRegression convergence hint continuation",
        silent=True,
    ),
    "sklearn_stratkfold_small_class": LogPattern(
        regex=(
            r"sklearn/model_selection/_split\.py:\d+: UserWarning:"
            r" The least populated class"
        ),
        level="INFO",
        description="sklearn StratifiedKFold least-populated-class warning",
        silent=True,
    ),
    "warnings_warn_continuation": LogPattern(
        regex=r"^\s*(warnings\.warn|warn)\(\s*$",
        level="INFO",
        description="Generic warnings.warn()/warn() continuation line",
        silent=True,
    ),
    # End-of-run bookkeeping + the df.head() preview printed by
    # experiments/common/pipeline.py after merging per-task parquets.
    "merge_summary": LogPattern(
        regex=r"Merged \d+ files -> ",
        level="INFO",
        description="Parquet merge summary",
        silent=True,
    ),
    "run_finished": LogPattern(
        regex=r"Finished \d+ tasks in \d+:\d+:\d+",
        level="INFO",
        description="Run-finished summary",
        silent=True,
    ),
    "results_saved": LogPattern(
        regex=r"Results saved to ",
        level="INFO",
        description="Results-saved summary",
        silent=True,
    ),
    "pandas_head_truncation": LogPattern(
        regex=r"\s\.\.\.\s",
        level="INFO",
        description=(
            "pandas df.head() preview row with column-truncation ellipsis"
        ),
        silent=True,
    ),
    "pandas_shape_footer": LogPattern(
        regex=r"^\[\d+ rows x \d+ columns\]$",
        level="INFO",
        description="pandas df.head() shape footer",
        silent=True,
    ),
    "joblib_worker_stopped": LogPattern(
        regex=(r"A worker stopped while some jobs were given to the executor"),
        level="WARNING",
        description="joblib worker stopped mid-task (auto-respawned)",
    ),
}


@dataclass(frozen=True)
class StratFitTime:
    dataset: str
    seed: str
    model: str
    strategy: str
    objective: str
    seconds: float


def inspect_log(
    path: Path,
) -> tuple[
    dict[str, list[str]],
    list[str],
    list[tuple[str, str, float]],
    list[StratFitTime],
]:
    """Parse log file.

    Returns (classified matches, unclassified lines, model fit times,
    strategy fit times). Any non-empty line that doesn't match a PATTERN
    entry (or one of the timing formats) lands in unclassified. If
    something shows up there often, promote it to PATTERNS so it gets a
    clean summary row.
    """
    matches: dict[str, list[str]] = {k: [] for k in PATTERNS}
    unclassified: list[str] = []
    fit_times: list[tuple[str, str, float]] = []
    strat_fit_times: list[StratFitTime] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        strat_timing = STRAT_FIT_TIME_PATTERN.search(line)
        if strat_timing:
            strat_fit_times.append(
                StratFitTime(
                    dataset=strat_timing["dataset"],
                    seed=strat_timing["seed"],
                    model=strat_timing["model"],
                    strategy=strat_timing["strategy"],
                    objective=strat_timing["objective"],
                    seconds=float(strat_timing["seconds"]),
                )
            )
            continue

        timing = FIT_TIME_PATTERN.search(line)
        if timing:
            fit_times.append(
                (
                    timing["dataset"],
                    timing["model"],
                    float(timing["seconds"]),
                )
            )
            continue

        matched = False
        for name, pat in PATTERNS.items():
            if re.search(pat.regex, line):
                matches[name].append(stripped)
                matched = True
                break
        if not matched:
            unclassified.append(stripped)

    return matches, unclassified, fit_times, strat_fit_times


def _format_per_group_aggregate(
    times_by_group: dict[str, list[float]],
    group_label: str,
) -> None:
    """Print total/count/avg/max per group, sorted by total."""
    if not times_by_group:
        return
    rows = sorted(
        times_by_group.items(), key=lambda kv: sum(kv[1]), reverse=True
    )
    max_name = max(len(g) for g, _ in rows)
    name_w = min(max(max_name, len(group_label)), 28)
    print(f"  Per-{group_label}:")
    for name, values in rows:
        total = sum(values)
        n = len(values)
        avg = total / n
        worst = max(values)
        display = name if len(name) <= name_w else name[: name_w - 1] + "…"
        print(
            f"    {display:<{name_w}s}"
            f"  {total:>7.1f}s / {n:>4d} fits"
            f"  (avg {avg:>6.2f}s, max {worst:>6.2f}s)"
        )


def summarize(
    matches: dict[str, list[str]],
    unclassified: list[str],
    fit_times: list[tuple[str, str, float]],
    strat_fit_times: list[StratFitTime] | None = None,
) -> None:
    """Print a summary of matched patterns, unclassified lines, and timings."""
    strat_fit_times = strat_fit_times or []
    has_issues = False

    for name, lines in matches.items():
        if not lines:
            continue
        pat = PATTERNS[name]
        if pat.silent:
            continue
        count = len(lines)
        has_issues = True

        icon = "!!" if pat.level == "ERROR" else ".."
        print(f"\n  {icon} {pat.description}: {count}")

        unique: Counter[str] = Counter()
        for line in lines:
            m = re.search(pat.regex, line)
            if m:
                unique[m.group(0)] += 1

        items = unique.most_common()
        if len(items) > MAX_UNIQUE_MESSAGES:
            truncated = len(unique) - MAX_UNIQUE_MESSAGES
            items = items[:MAX_UNIQUE_MESSAGES]
        else:
            truncated = 0
        for msg, n in items:
            prefix = f"     ({n}x) " if n > 1 else "          "
            if len(msg) > 120:
                msg = msg[:117] + "..."
            print(f"{prefix}{msg}")
        if truncated:
            print(f"     ... and {truncated} more")

    if unclassified:
        has_issues = True
        total = len(unclassified)
        print(
            f"\n  ?? Unclassified lines (promote to PATTERNS if frequent):"
            f" {total}"
        )
        unique_lines: Counter[str] = Counter(unclassified)
        items = unique_lines.most_common()
        if len(items) > MAX_UNIQUE_MESSAGES:
            truncated = len(unique_lines) - MAX_UNIQUE_MESSAGES
            items = items[:MAX_UNIQUE_MESSAGES]
        else:
            truncated = 0
        for msg, n in items:
            prefix = f"     ({n}x) " if n > 1 else "          "
            if len(msg) > 120:
                msg = msg[:117] + "..."
            print(f"{prefix}{msg}")
        if truncated:
            print(f"     ... and {truncated} more")

    if fit_times:
        total_seconds = sum(sec for _, _, sec in fit_times)
        shown = min(SLOWEST_FITS_TO_SHOW, len(fit_times))
        print(
            f"\n  .. Model fits: {len(fit_times)}"
            f" (total {total_seconds:.1f}s);"
            f" showing top {shown} slowest"
        )
        slowest = sorted(fit_times, key=lambda t: t[2], reverse=True)[:shown]
        for dataset, model, seconds in slowest:
            print(f"     {seconds:>8.2f}s  {model} on {dataset}")
        per_model: dict[str, list[float]] = {}
        for _, model, seconds in fit_times:
            per_model.setdefault(model, []).append(seconds)
        print()
        _format_per_group_aggregate(per_model, "model")

    if strat_fit_times:
        total_seconds = sum(t.seconds for t in strat_fit_times)
        shown = min(SLOWEST_FITS_TO_SHOW, len(strat_fit_times))
        print(
            f"\n  .. Strategy fits: {len(strat_fit_times)}"
            f" (total {total_seconds:.1f}s);"
            f" showing top {shown} slowest"
        )
        slowest_strats = sorted(
            strat_fit_times, key=lambda t: t.seconds, reverse=True
        )[:shown]
        for t in slowest_strats:
            print(
                f"     {t.seconds:>8.2f}s  {t.strategy}"
                f" for {t.objective}  ({t.model} on {t.dataset})"
            )
        per_strategy: dict[str, list[float]] = {}
        for t in strat_fit_times:
            per_strategy.setdefault(t.strategy, []).append(t.seconds)
        print()
        _format_per_group_aggregate(per_strategy, "strategy")

    if not has_issues:
        print("  No issues found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect run logs for known patterns.",
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Log files to inspect.",
    )
    args = parser.parse_args()

    for log_path in args.logs:
        if not log_path.exists():
            print(f"Not found: {log_path}")
            continue

        print(f"\n=== {log_path} ===")
        matches, unclassified, fit_times, strat_fit_times = inspect_log(
            log_path
        )
        summarize(matches, unclassified, fit_times, strat_fit_times)

        errors = sum(
            len(matches[k])
            for k in matches
            if PATTERNS[k].level == "ERROR" and not PATTERNS[k].silent
        )
        warnings = sum(
            len(matches[k])
            for k in matches
            if PATTERNS[k].level == "WARNING" and not PATTERNS[k].silent
        )
        infos = sum(
            len(matches[k])
            for k in matches
            if PATTERNS[k].level == "INFO" and not PATTERNS[k].silent
        )
        silent = sum(len(matches[k]) for k in matches if PATTERNS[k].silent)
        unknown = len(unclassified)
        print(
            f"\n  Totals: {errors} errors, {warnings} warnings,"
            f" {infos} info, {unknown} unclassified, {silent} silent"
        )


if __name__ == "__main__":
    main()
