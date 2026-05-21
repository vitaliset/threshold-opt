"""Experimental runner for binary classification experiments.

Example usage:
    uv run python -m experiments.binary.runner
    uv run python -m experiments.binary.runner --n-seeds 5 --n-jobs 4
    uv run python -m experiments.binary.runner \
        --resume experiments/results/run_<dir>
"""

import argparse
import logging

from experiments.binary import config
from experiments.common import pipeline, thread_limits
from threshold_opt import base, binary_strategies

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

STRATEGIES = {
    "Argmax": binary_strategies.ArgmaxThreshold,
    "IsotonicArgmax": binary_strategies.IsotonicArgmax,
    "IsotonicAnalytical": binary_strategies.IsotonicAnalytical,
    "PlattScalingArgmax": binary_strategies.PlattScalingArgmax,
    "PlattScalingAnalytical": binary_strategies.PlattScalingAnalytical,
    "TemperatureScalingArgmax": (binary_strategies.TemperatureScalingArgmax),
    "TemperatureScalingAnalytical": (
        binary_strategies.TemperatureScalingAnalytical
    ),
    "DirichletCalibrationArgmax": (
        binary_strategies.DirichletCalibrationArgmax
    ),
    "DirichletCalibrationAnalytical": (
        binary_strategies.DirichletCalibrationAnalytical
    ),
    "Analytical": binary_strategies.AnalyticalThreshold,
    "Empirical": binary_strategies.EmpiricalThreshold,
    "EmpiricalCV": lambda model: binary_strategies.EmpiricalThreshold(
        model, use_cv=True
    ),
}


def _binary_param_extractor(strategy: base.DecisionStrategy) -> float:
    """Extracts the threshold from a binary strategy."""
    if not isinstance(strategy, binary_strategies.BinaryDecisionStrategy):
        raise ValueError(
            "Binary param extractor requires a binary strategy, got"
            f" {type(strategy)}."
        )
    return strategy.threshold


def main() -> None:
    thread_limits.cap_torch()
    parser = argparse.ArgumentParser(
        description="Run binary classification experiments.",
    )
    pipeline.add_common_cli_args(parser)
    args = parser.parse_args()

    if args.device == "gpu" and args.n_jobs == -1:
        args.n_jobs = 1

    models = pipeline.select_models(
        config.MODELS,
        config.CPU_MODELS,
        config.GPU_MODELS,
        args.device,
        args.models,
    )

    if args.dump_predictions or args.from_predictions:
        pipeline.run_experiments_with_cache(
            datasets=config.DATASETS,
            models=models,
            strategies=STRATEGIES,
            objectives=config.METRIC_OBJECTIVES,
            output_filename_prefix="results_binary",
            cache_dir=args.dump_predictions or args.from_predictions,
            phase="dump" if args.dump_predictions else "from",
            param_extractor=_binary_param_extractor,
            learned_params_column_name="threshold",
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
        output_filename_prefix="results_binary",
        param_extractor=_binary_param_extractor,
        learned_params_column_name="threshold",
        n_seeds=args.n_seeds,
        master_seed=args.master_seed,
        n_jobs=args.n_jobs,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
