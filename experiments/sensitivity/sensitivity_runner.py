"""Sensitivity sweep: cost of a misspecified Bayes rule under two
perturbation modes.

Both perturbation modes share the same Monte Carlo harness and parquet
schema:

* ``--mode variance`` is the headline experiment. K isotropic
  Gaussian classes sit on a regular K-gon in d=2; the perturbed
  distribution Q scales every class's variance by a single factor
  ``alpha >= 1``. The variance multiplier alpha is itself K-independent
  by construction, so the per-K curves share a common axis.

* ``--mode meanshift`` is the robustness experiment introduced in
  Section ``sec:scaling_meanshift`` of the thesis. The class means
  sit at the vertices of a regular K-vertex configuration in d=K-1
  (mutually equidistant points); each class's mean is shifted by a
  fresh random unit vector scaled to a deterministic length ``xi``,
  drawn independently per class per seed. Under symmetric 0/1 cost
  the rules disagree (per-class independent shifts break the
  argmax-invariance that makes the variance perturbation degenerate),
  which is the cost-coupling check the section needs.

Both modes record (c_opt, c_misspec, regret) per (K, sweep_param, seed)
and merge into a single parquet at the end of the run.

Usage:
    uv run python -m experiments.sensitivity.sensitivity_runner
    uv run python -m experiments.sensitivity.sensitivity_runner \
        --mode meanshift --cost symmetric
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

from experiments.common import pipeline, seeds
from experiments.common.git_metadata import write_parquet_with_metadata
from experiments.sensitivity import sensitivity_math as sm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DIM = 2
SIGMA = 1.0
PARAM_EQ_TOL = 1e-12

PREFIX_BY_MODE = {
    "variance": "results_variance",
    "meanshift": "results_meanshift",
}

COST_BUILDERS = {
    "asymmetric_10x": (sm.asymmetric_10x_cost, "Asymmetric 10x"),
    "asymmetric_5x": (sm.asymmetric_5x_cost, "Asymmetric 5x"),
    "asymmetric_2x": (sm.asymmetric_2x_cost, "Asymmetric 2x"),
    "symmetric": (sm.symmetric_cost, "Symmetric"),
}


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(","))


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(","))


def _task_path(run_dir, K: int, param: float, seed: int, *, param_name: str):
    return run_dir / f"K{K}_{param_name}{param:g}_seed{seed}.parquet"


def _run_one_task(
    *,
    K: int,
    param: float,
    seed: int,
    param_name: str,
    mode: str,
    dim: int,
    means_P0: np.ndarray,
    sigma_P0: np.ndarray,
    priors: np.ndarray,
    cost: np.ndarray,
    cost_name: str,
    n_mc: int,
    separation: float,
    kl: float,
    run_dir,
) -> float | None:
    """Run one (K, param, seed) Monte Carlo task and write its parquet.

    Returns the realised regret value when the task corresponds to the
    zero-perturbation parameter (for the post-loop sanity check); None
    otherwise. Tasks whose parquet already exists are skipped (resume).
    """
    path = _task_path(run_dir, K, param, int(seed), param_name=param_name)
    if path.exists():
        return None
    rng = np.random.default_rng(int(seed))
    if mode == "variance":
        alpha = param
        means_Q = means_P0
        sigma_Q = np.sqrt(alpha) * sigma_P0
    else:
        sigma_Q = sigma_P0
        shifts = sm.random_unit_shifts(rng, K, dim, param)
        means_Q = means_P0 + shifts
    c_opt, c_mis, regret_sq = sm.monte_carlo_costs(
        rng,
        means_Q,
        sigma_P0,
        sigma_Q,
        priors,
        cost,
        n_mc,
        means_P0=means_P0,
    )
    regret = c_mis - c_opt
    row = {
        "K": K,
        "mode": mode,
        "param_name": param_name,
        "param": param,
        param_name: param,
        "kl": kl,
        "dim": dim,
        "separation": separation,
        "seed": int(seed),
        "n_mc": n_mc,
        "cost_name": cost_name,
        "c_opt": c_opt,
        "c_misspec": c_mis,
        "regret": regret,
        "regret_sq": regret_sq,
    }
    write_parquet_with_metadata(pd.DataFrame([row]), path)
    return regret


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep a perturbation parameter and K to measure the cost"
            " of using P0's Bayes-optimal rule under a perturbed Q."
            " --mode variance scales every class's variance by alpha"
            " (K-gon in R^2); --mode meanshift places means at K"
            " equidistant points in R^{K-1} and shifts each by an iid"
            " random unit direction of fixed magnitude r."
        )
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-mc", type=int, default=100_000)
    parser.add_argument("--Ks", type=str, default="2,3,4,5,7,10,15,20")
    parser.add_argument(
        "--mode",
        choices=("variance", "meanshift"),
        default="variance",
    )
    parser.add_argument(
        "--cost",
        choices=tuple(COST_BUILDERS.keys()),
        default="asymmetric_10x",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        # 17 log-spaced values in [1, 16] (factor ~ 2^(1/4) between
        # adjacent). Restricted to the broader-than-assumed branch.
        default=(
            "1.0,1.189,1.414,1.682,2.0,2.378,2.828,3.364,"
            "4.0,4.757,5.657,6.727,8.0,9.514,11.314,13.454,16.0"
        ),
    )
    parser.add_argument(
        "--shift-radii",
        type=str,
        # 13 values of the per-class shift magnitude r for the
        # meanshift mode. Joint KL = r^2 / (2 sigma^2); with sigma = 1
        # the sweep spans KL in [0, ~1.1].
        default=("0.0,0.1,0.15,0.22,0.32,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5"),
    )
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "joblib worker count for the per-task Monte Carlo. -1 uses"
            " all logical cores. Each task is a single (K, param, seed)"
            " Monte Carlo evaluation, so per-K and per-param work is"
            " replicated across workers; this is cheap compared to the"
            " MC inner loop and lets joblib distribute the (K, param,"
            " seed) grid across cores without further coordination."
        ),
    )
    args = parser.parse_args()

    Ks = _parse_int_list(args.Ks)
    cost_builder, cost_name = COST_BUILDERS[args.cost]
    if args.mode == "variance":
        params = _parse_float_list(args.alphas)
        param_name = "alpha"
        zero_value = 1.0
    else:
        params = _parse_float_list(args.shift_radii)
        param_name = "r"
        zero_value = 0.0

    prefix = PREFIX_BY_MODE[args.mode]
    run_dir = pipeline.create_run_dir(prefix, resume=args.resume)
    pipeline.setup_file_logging(run_dir)
    seed_vec = seeds.load_or_sample_seeds(
        run_dir, args.n_seeds, args.master_seed
    )
    logger.info("Replicate seeds (master=%d): %s", args.master_seed, seed_vec)
    logger.info(
        "mode=%s cost=%s Ks=%s %ss=%s n_mc=%d separation=%.3f",
        args.mode,
        args.cost,
        Ks,
        param_name,
        params,
        args.n_mc,
        args.separation,
    )

    # Precompute the per-K geometric setup once: means, priors, sigmas,
    # cost matrix. These are cheap, but joblib serialises closure
    # arguments to worker processes, so computing once and indexing by
    # K avoids redoing the setup inside every task.
    by_K: dict[int, dict] = {}
    for K in Ks:
        if args.mode == "meanshift":
            # K mutually equidistant class means in R^{K-1}: every pair
            # is at the same Euclidean distance, so per-class
            # independent unit shifts spread the posterior perturbation
            # over the full (K-1)-dimensional tangent space of the
            # probability simplex. The circumradius is rescaled with K
            # so the pairwise inter-class distance is K-independent and
            # equal to 2 * separation (matching the K=2 baseline of the
            # K-gon variance experiment); only the tangent-space
            # dimension varies with K, which is what the proposition is
            # about.
            dim = K - 1
            radius = args.separation * np.sqrt(2.0 * (K - 1) / K)
            means_P0 = sm.simplex_centers(K, radius)
        else:
            dim = DIM
            means_P0 = sm.kgon_centers(K, args.separation)
        by_K[K] = {
            "dim": dim,
            "means_P0": means_P0,
            "sigma_P0": np.full(K, SIGMA),
            "priors": np.full(K, 1.0 / K),
            "cost": cost_builder(K),
        }

    def _kl_for(K: int, param: float) -> float:
        if args.mode == "variance":
            return sm.gaussian_kl_isotropic(DIM, param)
        # meanshift: joint KL = r^2 / (2 sigma^2), deterministic.
        return param**2 / (2.0 * SIGMA**2)

    tasks = [
        (K, param, int(seed))
        for K in Ks
        for param in params
        for seed in seed_vec
    ]
    logger.info(
        "Dispatching %d tasks (Ks=%d, params=%d, seeds=%d) with n_jobs=%d",
        len(tasks),
        len(Ks),
        len(params),
        len(seed_vec),
        args.n_jobs,
    )

    results = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(_run_one_task)(
            K=K,
            param=param,
            seed=seed,
            param_name=param_name,
            mode=args.mode,
            kl=_kl_for(K, param),
            cost_name=cost_name,
            n_mc=args.n_mc,
            separation=args.separation,
            run_dir=run_dir,
            **by_K[K],
        )
        for K, param, seed in tasks
    )

    zero_param_regrets: list[float] = [
        regret
        for (K, param, _seed), regret in zip(tasks, results)
        if regret is not None and abs(param - zero_value) < PARAM_EQ_TOL
    ]

    if zero_param_regrets:
        worst = max(abs(r) for r in zero_param_regrets)
        if worst > PARAM_EQ_TOL:
            logger.warning(
                "Non-zero regret at %s=%g: |worst|=%.3e (expect 0)",
                param_name,
                zero_value,
                worst,
            )

    pipeline.merge_run_results(run_dir, prefix)


if __name__ == "__main__":
    main()
