"""Experimental runner for multiclass classification experiments.

Example usage:
    uv run python -m experiments.multiclass.runner
    uv run python -m experiments.multiclass.runner --n-seeds 5 --n-jobs 4
    uv run python -m experiments.multiclass.runner \
        --resume experiments/results/run_<dir>
"""

import argparse
import logging
from typing import Any

from experiments.common import pipeline, thread_limits
from experiments.common.config import RANDOM_SEARCH_N_ITER
from experiments.multiclass import config
from threshold_opt import base, multiclass_strategies, optimization

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _rs(bounds: tuple[float, float]) -> optimization.RandomSearchOptimizer:
    """Random-search optimizer at the shared campaign budget.

    Only ``n_iter`` is taken from the shared constant; ``bounds`` mirror
    each strategy's library default exactly, so wiring the budget here
    does not silently change any strategy's search range. ``seed`` and
    ``n_jobs`` stay at the library defaults, matching the campaign's
    pre-existing behavior (the budget is the only thing that changes).
    """
    return optimization.RandomSearchOptimizer(
        n_iter=RANDOM_SEARCH_N_ITER, bounds=bounds
    )


STRATEGIES = {
    "Argmax": multiclass_strategies.ArgmaxMulticlass,
    "IsotonicArgmax": multiclass_strategies.IsotonicArgmaxMulticlass,
    "PlattScalingArgmax": (multiclass_strategies.PlattScalingArgmaxMulticlass),
    "TemperatureScalingArgmax": (
        multiclass_strategies.TemperatureScalingArgmaxMulticlass
    ),
    "DirichletCalibrationArgmax": (
        multiclass_strategies.DirichletCalibrationMulticlass
    ),
    "Analytical": multiclass_strategies.AnalyticalMulticlass,
    "IsotonicAnalytical": (multiclass_strategies.IsotonicAnalyticalMulticlass),
    "PlattScalingAnalytical": (
        multiclass_strategies.PlattScalingAnalyticalMulticlass
    ),
    "TemperatureScalingAnalytical": (
        multiclass_strategies.TemperatureScalingAnalyticalMulticlass
    ),
    "DirichletCalibrationAnalytical": (
        multiclass_strategies.DirichletCalibrationAnalyticalMulticlass
    ),
    "ThresholdRatio": lambda model: (
        multiclass_strategies.ThresholdRatioMulticlass(
            model, optimizer=_rs((0.01, 1.5))
        )
    ),
    "VoronoiTuning": lambda model: (
        multiclass_strategies.VoronoiTuningMulticlass(
            model, n_iter=RANDOM_SEARCH_N_ITER
        )
    ),
    "SequentialSelection": (
        lambda model: multiclass_strategies.SequentialSelectionMulticlass(
            model, optimizer=_rs((0.0, 1.0))
        )
    ),
    "RiskPartitioning": multiclass_strategies.RiskPartitioningStrategy,
    "LinearScaling": lambda model: (
        multiclass_strategies.LinearScalingMulticlass(
            model, optimizer=_rs((0.0, 5.0))
        )
    ),
    "MultidimensionalThreshold": (
        lambda model: multiclass_strategies.MultidimensionalThresholdStrategy(
            model, optimizer=_rs((-1.0, 1.0))
        )
    ),
    "ThresholdRatioCV": lambda model: (
        multiclass_strategies.ThresholdRatioMulticlass(
            model, optimizer=_rs((0.01, 1.5)), use_cv=True
        )
    ),
    "VoronoiTuningCV": lambda model: (
        multiclass_strategies.VoronoiTuningMulticlass(
            model, n_iter=RANDOM_SEARCH_N_ITER, use_cv=True
        )
    ),
    "SequentialSelectionCV": (
        lambda model: multiclass_strategies.SequentialSelectionMulticlass(
            model, optimizer=_rs((0.0, 1.0)), use_cv=True
        )
    ),
    "RiskPartitioningCV": (
        lambda model: multiclass_strategies.RiskPartitioningStrategy(
            model,
            optimizer=_rs((-1.0, 1.0)),
            solver="random_search",
            use_cv=True,
        )
    ),
    "LinearScalingCV": lambda model: (
        multiclass_strategies.LinearScalingMulticlass(
            model, optimizer=_rs((0.0, 5.0)), use_cv=True
        )
    ),
    "MultidimensionalThresholdCV": (
        lambda model: multiclass_strategies.MultidimensionalThresholdStrategy(
            model, optimizer=_rs((-1.0, 1.0)), use_cv=True
        )
    ),
    "FullMatrixPartitioning": lambda model: (
        multiclass_strategies.FullMatrixPartitioningStrategy(
            model, optimizer=_rs((-1.0, 1.0))
        )
    ),
    "FullMatrixPartitioningCV": (
        lambda model: multiclass_strategies.FullMatrixPartitioningStrategy(
            model, optimizer=_rs((-1.0, 1.0)), use_cv=True
        )
    ),
}


def _multiclass_param_extractor(strategy: base.DecisionStrategy) -> Any:
    """Extracts relevant parameters from a multiclass strategy."""
    if not isinstance(
        strategy, multiclass_strategies.MulticlassDecisionStrategy
    ):
        raise ValueError(
            "Multiclass param extractor requires a multiclass strategy, got"
            f" {type(strategy)}."
        )
    if hasattr(strategy, "thresholds") and strategy.thresholds is not None:
        return strategy.thresholds.tolist()
    elif (
        hasattr(strategy, "partition_matrix")
        and strategy.partition_matrix is not None
    ):
        # Flatten the K x K matrix to a 1-D list so the learned_params
        # parquet column stays a uniform list<double> across strategies
        # (mixing flat and nested lists makes pyarrow's Arrow conversion
        # fail). Same convention VoronoiTuning uses for its centroids.
        return strategy.partition_matrix.flatten().tolist()
    elif hasattr(strategy, "biases") and strategy.biases is not None:
        return strategy.biases.tolist()
    elif (
        hasattr(strategy, "best_centroids")
        and strategy.best_centroids is not None
    ):
        return strategy.best_centroids.flatten().tolist()
    elif (
        hasattr(strategy, "scaling_factors")
        and strategy.scaling_factors is not None
    ):
        # Per-truth-class cost weights beta (K-vector). Already 1-D, so
        # .tolist() yields a flat list<double>, matching the uniform
        # learned_params column convention (same as thresholds/biases).
        return strategy.scaling_factors.tolist()
    return None


def main() -> None:
    thread_limits.cap_torch()
    parser = argparse.ArgumentParser(
        description="Run multiclass classification experiments.",
    )
    pipeline.add_common_cli_args(parser)
    args = parser.parse_args()

    if args.device == "gpu" and args.n_jobs == -1:
        args.n_jobs = 1

    cpu_models = getattr(config, "CPU_MODELS", config.MODELS)
    gpu_models = getattr(config, "GPU_MODELS", {})
    models = pipeline.select_models(
        config.MODELS,
        cpu_models,
        gpu_models,
        args.device,
        args.models,
    )

    if args.dump_predictions or args.from_predictions:
        pipeline.run_experiments_with_cache(
            datasets=config.DATASETS,
            models=models,
            strategies=STRATEGIES,
            objectives=config.METRIC_OBJECTIVES,
            output_filename_prefix="results_multiclass",
            cache_dir=args.dump_predictions or args.from_predictions,
            phase="dump" if args.dump_predictions else "from",
            param_extractor=_multiclass_param_extractor,
            learned_params_column_name="learned_params",
            n_seeds=args.n_seeds,
            master_seed=args.master_seed,
            n_jobs=args.n_jobs,
            resume=args.resume,
        )
        return

    pipeline.run_experiments_and_export(
        datasets=config.DATASETS,
        models=models,
        strategies=STRATEGIES,
        objectives=config.METRIC_OBJECTIVES,
        output_filename_prefix="results_multiclass",
        param_extractor=_multiclass_param_extractor,
        learned_params_column_name="learned_params",
        n_seeds=args.n_seeds,
        master_seed=args.master_seed,
        n_jobs=args.n_jobs,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
