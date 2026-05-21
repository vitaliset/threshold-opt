"""Unified profiling harness for `threshold_opt` hot-path exploration.

Runs a single-process slice of either the multiclass or binary
experiment pipeline (or a standalone DirichletCalibrator fit) under
cProfile and prints a summary focused on `threshold_opt`.

The harness is intentionally small: one model, one objective, a
configurable subset of strategies on a configurable dataset. That keeps
the profile readable and keeps wall time short enough to iterate on.

Usage:
    uv run python -m experiments.profiling.profile_threshold_opt
    uv run python -m experiments.profiling.profile_threshold_opt \\
        --mode binary --dataset sonar
    uv run python -m experiments.profiling.profile_threshold_opt \\
        --mode dirichlet --n-runs 5 --variant both
    uv run python -m experiments.profiling.profile_threshold_opt \\
        --mode multiclass --n-runs 3
    uv run python -m experiments.profiling.profile_threshold_opt \\
        --dataset wine --top 40
    uv run python -m experiments.profiling.profile_threshold_opt \\
        --strategies VoronoiTuning RiskPartitioningCV
    uv run python -m experiments.profiling.profile_threshold_opt \\
        --save profile.out

The saved file can be inspected later with, e.g., `snakeviz profile.out`.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import time
from typing import Callable, Mapping

import numpy as np
from sklearn import linear_model

from experiments.common import data_loader, pipeline
from experiments.multiclass import config as mc_config
from threshold_opt import (
    base,
    binary_strategies,
    calibration,
    multiclass_strategies,
)

StrategyFactory = Callable[[object], base.DecisionStrategy]

MULTICLASS_STRATEGIES: Mapping[str, StrategyFactory] = {
    "Analytical": multiclass_strategies.AnalyticalMulticlass,
    "ThresholdRatio": multiclass_strategies.ThresholdRatioMulticlass,
    "VoronoiTuning": multiclass_strategies.VoronoiTuningMulticlass,
    "VoronoiTuningCV": lambda model: (
        multiclass_strategies.VoronoiTuningMulticlass(model, use_cv=True)
    ),
    "SequentialSelection": multiclass_strategies.SequentialSelectionMulticlass,
    "RiskPartitioning": multiclass_strategies.RiskPartitioningStrategy,
    "RiskPartitioningCV": lambda model: (
        multiclass_strategies.RiskPartitioningStrategy(
            model, solver="random_search", use_cv=True
        )
    ),
    "LinearScaling": multiclass_strategies.LinearScalingMulticlass,
    "LinearScalingCV": lambda model: (
        multiclass_strategies.LinearScalingMulticlass(model, use_cv=True)
    ),
    "MultidimensionalThreshold": (
        multiclass_strategies.MultidimensionalThresholdStrategy
    ),
    "IsotonicArgmax": multiclass_strategies.IsotonicArgmaxMulticlass,
    "IsotonicAnalytical": multiclass_strategies.IsotonicAnalyticalMulticlass,
    "PlattScalingArgmax": (multiclass_strategies.PlattScalingArgmaxMulticlass),
    "PlattScalingAnalytical": (
        multiclass_strategies.PlattScalingAnalyticalMulticlass
    ),
    "TemperatureScalingArgmax": (
        multiclass_strategies.TemperatureScalingArgmaxMulticlass
    ),
    "TemperatureScalingAnalytical": (
        multiclass_strategies.TemperatureScalingAnalyticalMulticlass
    ),
    "DirichletCalibrationArgmax": (
        multiclass_strategies.DirichletCalibrationMulticlass
    ),
    "DirichletCalibrationAnalytical": (
        multiclass_strategies.DirichletCalibrationAnalyticalMulticlass
    ),
}

BINARY_STRATEGIES: Mapping[str, StrategyFactory] = {
    "Argmax": binary_strategies.ArgmaxThreshold,
    "Analytical": binary_strategies.AnalyticalThreshold,
    "Empirical": binary_strategies.EmpiricalThreshold,
    "EmpiricalCV": lambda model: binary_strategies.EmpiricalThreshold(
        model, use_cv=True
    ),
    "IsotonicArgmax": binary_strategies.IsotonicArgmax,
    "IsotonicAnalytical": binary_strategies.IsotonicAnalytical,
    "PlattScalingArgmax": binary_strategies.PlattScalingArgmax,
    "PlattScalingAnalytical": binary_strategies.PlattScalingAnalytical,
    "TemperatureScalingArgmax": binary_strategies.TemperatureScalingArgmax,
    "TemperatureScalingAnalytical": (
        binary_strategies.TemperatureScalingAnalytical
    ),
    "DirichletCalibrationArgmax": (
        binary_strategies.DirichletCalibrationArgmax
    ),
    "DirichletCalibrationAnalytical": (
        binary_strategies.DirichletCalibrationAnalytical
    ),
}

MODELS = {
    "LogisticRegression()": linear_model.LogisticRegression(random_state=42),
}

MULTICLASS_OBJECTIVES = {
    "Custom Cost": mc_config.create_custom_cost_matrix,
}

BINARY_OBJECTIVES = {
    "Asymmetric 10x Cost": np.array([[0.0, 10.0], [1.0, 0.0]]),
}

DIRICHLET_VARIANTS = {
    # (reg_lambda, reg_mu)
    "diril2": (1e-3, None),
    "dirodir": (1e-3, 1e-3),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile threshold_opt on a single experiment slice.",
    )
    parser.add_argument(
        "--mode",
        choices=["multiclass", "binary", "dirichlet"],
        default="multiclass",
        help=(
            "Which pipeline slice to profile. 'dirichlet' profiles"
            " DirichletCalibrator.fit in isolation on synthetic data."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name (default: digits for multiclass, spambase for"
            " binary). Ignored in dirichlet mode."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for the split. With --n-runs N the pipeline modes"
            " iterate seed, seed+1, ..., seed+N-1; dirichlet mode re-uses"
            " this seed for all N fits."
        ),
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help=(
            "Number of measured iterations (default: 1 for pipeline modes,"
            " 5 for dirichlet mode). Wall-time stats (min/median/mean) are"
            " reported per iteration; cProfile stats are aggregated across"
            " all iterations."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=(
            "Subset of strategies to profile (default: all for the chosen"
            " mode). Ignored in dirichlet mode."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=["diril2", "dirodir", "both"],
        default="both",
        help=(
            "Dirichlet-mode regularisation branch to exercise. 'diril2'"
            " uses reg_mu=None (plain L2), 'dirodir' uses the paper's"
            " ODIR scheme, 'both' runs each sequentially. Ignored in"
            " pipeline modes."
        ),
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=10,
        help="n_classes for synthetic dirichlet-mode data (default: 10).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="n_samples for synthetic dirichlet-mode data (default: 5000).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of rows to print in each stats table.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to dump raw cProfile stats (for snakeviz).",
    )
    return parser.parse_args()


def _run_pipeline_slice(
    dataset: str,
    seed: int,
    strategies: Mapping[str, StrategyFactory],
    objectives: Mapping[str, object],
) -> None:
    X, y = data_loader.load_dataset(dataset)
    pipeline.run_experiment_pipeline(
        dataset_name=dataset,
        X=X,
        y=y,
        random_seed=seed,
        models=MODELS,
        strategies=strategies,
        objectives=objectives,
    )


def _make_dirichlet_synthetic(
    n_samples: int, n_classes: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(n_classes) * 0.8, size=n_samples)
    cum = probs.cumsum(axis=1)
    u = rng.random(n_samples)[:, None]
    y = (u < cum).argmax(axis=1).astype(np.intp)
    return probs, y


def _run_dirichlet_fit(
    probs: np.ndarray,
    y: np.ndarray,
    reg_lambda: float,
    reg_mu: float | None,
) -> None:
    calibration.DirichletCalibrator(reg_lambda=reg_lambda, reg_mu=reg_mu).fit(
        probs, y
    )


def _resolve_pipeline_inputs(
    args: argparse.Namespace,
) -> tuple[str, Mapping[str, StrategyFactory], Mapping[str, object]]:
    if args.mode == "multiclass":
        dataset = args.dataset or "digits"
        strategies = MULTICLASS_STRATEGIES
        objectives = MULTICLASS_OBJECTIVES
    else:
        dataset = args.dataset or "spambase"
        strategies = BINARY_STRATEGIES
        objectives = BINARY_OBJECTIVES

    if args.strategies:
        unknown = set(args.strategies) - set(strategies)
        if unknown:
            raise SystemExit(
                f"Unknown strategies for mode={args.mode}: {sorted(unknown)}."
                f" Known: {sorted(strategies)}"
            )
        strategies = {name: strategies[name] for name in args.strategies}

    return dataset, strategies, objectives


def _format_stats(per_iter: list[float]) -> str:
    n = len(per_iter)
    total = sum(per_iter)
    if n == 1:
        return f"wall={total:.4f}s (1 iter)"
    return (
        f"total={total:.4f}s  n={n}"
        f"  min={min(per_iter):.4f}s"
        f"  median={statistics.median(per_iter):.4f}s"
        f"  mean={statistics.mean(per_iter):.4f}s"
    )


def _print_stats_tables(
    profiler: cProfile.Profile, top: int, label: str
) -> None:
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"=== [{label}] Top {top} by cumulative time (all) ===")
    stats.print_stats(top)

    print(f"\n=== [{label}] Top {top} restricted to threshold_opt ===")
    stats.print_stats("threshold_opt", top)

    print(f"\n=== [{label}] Top {top} by tottime ===")
    stats.sort_stats("tottime").print_stats(top)


def _profile_iterations(
    warmup: Callable[[], None],
    iteration_fns: list[Callable[[], None]],
) -> tuple[cProfile.Profile, list[float]]:
    """Warm up once, then profile a list of per-iteration callables.

    Returns the shared profiler (stats aggregated across iterations) and
    the list of per-iteration wall-clock times.
    """
    warmup()

    profiler = cProfile.Profile()
    per_iter: list[float] = []
    profiler.enable()
    try:
        for fn in iteration_fns:
            t0 = time.perf_counter()
            fn()
            per_iter.append(time.perf_counter() - t0)
    finally:
        profiler.disable()
    return profiler, per_iter


def _run_pipeline_mode(args: argparse.Namespace) -> None:
    dataset, strategies, objectives = _resolve_pipeline_inputs(args)
    n_runs = args.n_runs if args.n_runs is not None else 1

    def warmup() -> None:
        _run_pipeline_slice(dataset, args.seed, strategies, objectives)

    iteration_fns = [
        (
            lambda s=args.seed + i: _run_pipeline_slice(
                dataset, s, strategies, objectives
            )
        )
        for i in range(n_runs)
    ]

    header = (
        f"mode={args.mode}  dataset={dataset!r}  seed={args.seed}"
        f"  n_runs={n_runs}  strategies={len(strategies)}"
    )

    profiler, per_iter = _profile_iterations(warmup, iteration_fns)

    print(f"\n{header}  {_format_stats(per_iter)}\n")
    _print_stats_tables(profiler, args.top, args.mode)

    if args.save:
        pstats.Stats(profiler).dump_stats(args.save)
        print(f"\nRaw stats written to {args.save}")


def _run_dirichlet_mode(args: argparse.Namespace) -> None:
    n_runs = args.n_runs if args.n_runs is not None else 5

    variants = (
        ["diril2", "dirodir"] if args.variant == "both" else [args.variant]
    )

    probs, y = _make_dirichlet_synthetic(
        args.n_samples, args.n_classes, args.seed
    )

    all_profilers: dict[str, cProfile.Profile] = {}
    for variant in variants:
        reg_lambda, reg_mu = DIRICHLET_VARIANTS[variant]

        def warmup(rl=reg_lambda, rm=reg_mu) -> None:
            _run_dirichlet_fit(probs, y, rl, rm)

        iteration_fns = [
            (
                lambda rl=reg_lambda, rm=reg_mu: _run_dirichlet_fit(
                    probs, y, rl, rm
                )
            )
            for _ in range(n_runs)
        ]

        header = (
            f"mode=dirichlet  variant={variant}"
            f"  n_samples={args.n_samples}  n_classes={args.n_classes}"
            f"  seed={args.seed}  n_runs={n_runs}"
            f"  reg_lambda={reg_lambda}  reg_mu={reg_mu}"
        )

        profiler, per_iter = _profile_iterations(warmup, iteration_fns)
        all_profilers[variant] = profiler

        print(f"\n{header}  {_format_stats(per_iter)}\n")
        _print_stats_tables(profiler, args.top, f"dirichlet:{variant}")

    if args.save:
        # With multiple variants, suffix the file name so neither is lost.
        if len(variants) == 1:
            pstats.Stats(all_profilers[variants[0]]).dump_stats(args.save)
            print(f"\nRaw stats written to {args.save}")
        else:
            for variant, profiler in all_profilers.items():
                path = f"{args.save}.{variant}"
                pstats.Stats(profiler).dump_stats(path)
                print(f"\nRaw stats written to {path}")


def main() -> None:
    args = _parse_args()
    if args.mode == "dirichlet":
        _run_dirichlet_mode(args)
    else:
        _run_pipeline_mode(args)


if __name__ == "__main__":
    main()
