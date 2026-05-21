"""Per-model / per-strategy timing breakdown for the binary pipeline.

Usage:
    uv run python -m experiments.profiling.timing_analysis
    uv run python -m experiments.profiling.timing_analysis \\
        --dataset ionosphere
    uv run python -m experiments.profiling.timing_analysis \\
        --dataset breast-cancer --seed 42

Prints a detailed timing table showing how long each model training
and strategy fit takes, making it easy to spot bottlenecks.
"""

import argparse
import logging
import time

import numpy as np
from sklearn.metrics import f1_score

from experiments.binary import config
from experiments.common import data_loader, pipeline
from threshold_opt import binary_strategies, metrics

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

STRATEGIES = {
    "Argmax": binary_strategies.ArgmaxThreshold,
    "IsotonicArgmax": binary_strategies.IsotonicArgmax,
    "PlattScalingArgmax": binary_strategies.PlattScalingArgmax,
    "TemperatureScalingArgmax": binary_strategies.TemperatureScalingArgmax,
    "DirichletCalibrationArgmax": (
        binary_strategies.DirichletCalibrationArgmax
    ),
    "Analytical": binary_strategies.AnalyticalThreshold,
    "Empirical": binary_strategies.EmpiricalThreshold,
    "EmpiricalCV": lambda model: binary_strategies.EmpiricalThreshold(
        model, use_cv=True
    ),
}

# Use a subset of objectives for faster analysis
OBJECTIVES: dict[str, metrics.OptimizationObjective] = {
    "Symmetric Cost": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "F1": metrics.MetricObjective(
        metric_fn=lambda yt, yp: f1_score(yt, yp, zero_division=0),
        higher_is_better=True,
        vectorized_key="f1",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-model timing analysis.")
    parser.add_argument(
        "--dataset", default="sonar", help="Dataset name (default: sonar)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)."
    )
    args = parser.parse_args()

    print(f"=== TIMING ANALYSIS: {args.dataset}, seed={args.seed} ===\n")

    t0 = time.perf_counter()
    X, y = data_loader.load_dataset(args.dataset)
    t_load = time.perf_counter() - t0
    print(f"[{t_load:.3f}s] Data loading: {X.shape}")

    t0 = time.perf_counter()
    splits = pipeline._split_data(X, y, args.seed)
    if not splits:
        print("ERROR: split failed")
        return
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        _y_test,
        X_train_full,
        y_train_full,
    ) = splits
    t_split = time.perf_counter() - t0
    print(
        f"[{t_split:.3f}s] Data splitting: "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    print("\n=== PER MODEL TIMING ===")

    for model_name, model_inst in config.MODELS.items():
        print(f"\n--- {model_name} ---")

        t0 = time.perf_counter()
        fitted_split = pipeline._train_model(
            model_name, model_inst, args.dataset, X_train, y_train
        )
        t_train_split = time.perf_counter() - t0

        t0 = time.perf_counter()
        fitted_full = pipeline._train_model(
            model_name, model_inst, args.dataset, X_train_full, y_train_full
        )
        t_train_full = time.perf_counter() - t0

        print(
            f"  Train(split)={t_train_split:.3f}s  "
            f"Train(full)={t_train_full:.3f}s"
        )

        if not (fitted_split and fitted_full):
            print("  [SKIPPED - training failed]")
            continue

        for strat_name, strat_factory in STRATEGIES.items():
            if strat_name in ("Analytical", "EmpiricalCV"):
                model_to_use = fitted_full
                X_fit, y_fit = X_train_full, y_train_full
            elif strat_name == "Empirical":
                model_to_use = fitted_split
                X_fit, y_fit = X_val, y_val
            else:
                model_to_use = fitted_split
                X_fit, y_fit = X_val, y_val

            strategy = strat_factory(model_to_use)

            obj_times: dict[str, float] = {}
            for obj_name, obj in OBJECTIVES.items():
                t0 = time.perf_counter()
                try:
                    strategy.fit(X_fit, y_fit, obj)
                except ValueError:
                    pass
                obj_times[obj_name] = time.perf_counter() - t0

            total_strat = sum(obj_times.values())
            times_str = "  ".join(
                f"{k}={v:.3f}s" for k, v in obj_times.items()
            )
            print(f"  {strat_name}: {total_strat:.3f}s  ({times_str})")


if __name__ == "__main__":
    main()
