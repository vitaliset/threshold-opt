"""Synthetic verification of the dimensional-dilution Lemma.

The Lemma states that for a scalar functional phi smooth at an interior
posterior p with gradient g in V, and a perturbation u uniform on the
radius-r sphere in V, ``E_u[(phi(p+u) - phi(p))^2] = r^2/(K-1) ||g||^2 +
O(r^3)``. The mean-shift construction of Section ``sec:scaling_meanshift``
does not satisfy the Lemma's directional isotropy assumption strictly
(the induced posterior perturbation at each X has a structure-tied
covariance, not the identity in V), so we verify the Lemma directly in
its native setting: reference posteriors sampled from a Dirichlet on the
simplex interior, perturbations sampled uniformly on the radius-r sphere
in V, and the cost-sensitive expected cost as the functional. The K-decay
exponent of the integrated ratio
``E_p[E_u[(phi(p+u) - phi(p))^2]] / E_p[||g(p)||^2]``
regressed against ``log(K-1)`` is the empirical confirmation of the
Lemma's leading-order rate.

Records (E_dphi_sq, E_g_sq) per (K, r, seed, cost) and merges into a
single parquet at the end of the run.

Usage:
    uv run python -m experiments.sensitivity.lemma_test_runner
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

OUTPUT_PREFIX = "results_lemma_test"

COST_BUILDERS = {
    "symmetric": (sm.symmetric_cost, "Symmetric"),
    "asymmetric_2x": (sm.asymmetric_2x_cost, "Asymmetric 2x"),
    "asymmetric_5x": (sm.asymmetric_5x_cost, "Asymmetric 5x"),
    "asymmetric_10x": (sm.asymmetric_10x_cost, "Asymmetric 10x"),
}


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(","))


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(","))


def _run_one_task(
    *,
    K: int,
    r: float,
    seed: int,
    cost: np.ndarray,
    cost_name: str,
    n_p: int,
    n_u: int,
    dirichlet_alpha: float,
    run_dir,
):
    path = (
        run_dir
        / f"K{K}_r{r:g}_seed{int(seed)}_{cost_name.replace(' ', '_')}.parquet"
    )
    if path.exists():
        return
    rng = np.random.default_rng(int(seed))
    dphi_sq_linear, dphi_sq_full, g_sq = sm.synthetic_lemma_test(
        rng, K, r, cost, n_p, n_u, dirichlet_alpha=dirichlet_alpha
    )
    row = {
        "K": K,
        "r": r,
        "seed": int(seed),
        "n_p": n_p,
        "n_u": n_u,
        "dirichlet_alpha": dirichlet_alpha,
        "cost_name": cost_name,
        "dphi_sq_linear": dphi_sq_linear,
        "dphi_sq_full": dphi_sq_full,
        "g_sq": g_sq,
    }
    write_parquet_with_metadata(pd.DataFrame([row]), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic verification of the dimensional-dilution Lemma:"
            " sample reference posteriors from Dirichlet on the simplex"
            " interior, sample perturbations uniformly on the radius-r"
            " sphere in V (the simplex tangent space), and verify the"
            " K-decay exponent of E[(phi(p+u) - phi(p))^2] / E[||g||^2]"
            " against log(K-1) matches the Lemma's leading-order"
            " prediction of 1."
        )
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-p", type=int, default=400)
    parser.add_argument("--n-u", type=int, default=1000)
    parser.add_argument("--dirichlet-alpha", type=float, default=2.0)
    parser.add_argument("--Ks", type=str, default="2,3,4,5,7,10,15,20")
    parser.add_argument(
        "--rs",
        type=str,
        default="0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2",
    )
    parser.add_argument(
        "--costs",
        type=str,
        default="symmetric,asymmetric_2x,asymmetric_5x,asymmetric_10x",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    Ks = _parse_int_list(args.Ks)
    rs = _parse_float_list(args.rs)
    cost_keys = tuple(args.costs.split(","))
    cost_specs = [COST_BUILDERS[c] for c in cost_keys]

    run_dir = pipeline.create_run_dir(OUTPUT_PREFIX, resume=args.resume)
    pipeline.setup_file_logging(run_dir)
    seed_vec = seeds.load_or_sample_seeds(
        run_dir, args.n_seeds, args.master_seed
    )
    logger.info("Replicate seeds (master=%d): %s", args.master_seed, seed_vec)
    logger.info(
        "Ks=%s rs=%s costs=%s n_p=%d n_u=%d dirichlet_alpha=%.2f",
        Ks,
        rs,
        cost_keys,
        args.n_p,
        args.n_u,
        args.dirichlet_alpha,
    )

    # Precompute the cost matrices once per (K, cost_key); joblib
    # serialises closure args to workers, so building per task wastes
    # serialisation bandwidth.
    cost_by_K_key: dict[tuple[int, str], tuple[np.ndarray, str]] = {}
    for K in Ks:
        for cost_key, (builder, cost_name) in zip(cost_keys, cost_specs):
            cost_by_K_key[(K, cost_key)] = (builder(K), cost_name)

    tasks = [
        (K, r, int(seed), cost_key)
        for K in Ks
        for r in rs
        for seed in seed_vec
        for cost_key in cost_keys
    ]
    logger.info(
        "Dispatching %d tasks (Ks=%d, rs=%d, seeds=%d, costs=%d)"
        " with n_jobs=%d",
        len(tasks),
        len(Ks),
        len(rs),
        len(seed_vec),
        len(cost_keys),
        args.n_jobs,
    )

    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(_run_one_task)(
            K=K,
            r=r,
            seed=seed,
            cost=cost_by_K_key[(K, cost_key)][0],
            cost_name=cost_by_K_key[(K, cost_key)][1],
            n_p=args.n_p,
            n_u=args.n_u,
            dirichlet_alpha=args.dirichlet_alpha,
            run_dir=run_dir,
        )
        for K, r, seed, cost_key in tasks
    )

    pipeline.merge_run_results(run_dir, OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
