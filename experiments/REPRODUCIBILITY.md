# Reproducibility

This document is the scientific reproducibility contract for the
experiment campaigns (the cloud *operations* how-to lives separately in
[cloud/README.md](cloud/README.md)). It records what is deterministic,
the one known soft spot, the run procedure, and where every constant
lives.

## TL;DR

Given the same code revision and the same machine for model fitting, the
campaign is reproducible run to run. The decision-strategy results are
*fully* deterministic; the only non-bit-exact step is GPU neural-model
training, and the predict-cache decoupling removes even that from the
critical path (replay from a frozen cache is exact).

## What is deterministic (by construction)

| Source of randomness | How it is pinned |
|---|---|
| Replicate seeds | Derived from `master_seed = 42` via `pipeline.seeds.load_or_sample_seeds`; written to `<run_dir>/seeds.json` and reused on `--resume`. |
| Train/val/test split | `train_test_split(random_state=seed)` per replicate. |
| Preprocessing | `ColumnTransformer` fit once per dataset *before* splitting; impute + scale / one-hot are deterministic. |
| CV-fold refits (CV strategies, shared precompute) | `StratifiedKFold(random_state=base.CV_RANDOM_STATE)`. |
| Random-search decision tuning | `RandomSearchOptimizer`: an unset seed coalesces to `DEFAULT_RANDOM_SEARCH_SEED` (no more `default_rng(None)` OS entropy). An explicit `seed` still overrides. |
| CPU base models | scikit-learn / LGBM / XGBoost / CatBoost with `random_state=42` — deterministic, so binary-CPU and multiclass-CPU are bit-reproducible end to end. |

## The one known soft spot

GPU neural-model *fitting* (TabM trains a network; TabPFN / TabICL run a
pretrained forward pass) is subject to standard GPU non-determinism
(cuDNN/atomics, driver, hardware). Therefore:

- Re-running **Phase A** (GPU model fits) from scratch on different
  GPU/driver/hardware may not be bit-identical.
- Everything downstream of the cached predictions **is** exact. The
  decoupling (below) means the decision-strategy comparison — the object
  of study — is reproduced bit-for-bit from a retained Phase-A cache.

Practical rule: **keep the Phase-A prediction caches** alongside the
results; they are the reproducibility anchor for the GPU models.

## Predict-cache decoupling

Each task splits into two phases that, run back to back, are
**bit-identical** to the monolithic pipeline (gate-enforced):

- **Phase A** (`--dump-predictions DIR`): fit every model, record every
  `predict_proba` / `decision_function` keyed by
  `(fitted-state, response-method, input)`, serialize per
  `(dataset, seed, model)`. GPU-bound. Runs the strategy loop over only a
  **representative objective subset** (one metric + one cost matrix, via
  `pipeline._representative_objectives`): the recorded predict inputs
  (`X_val`/`X_train_full`, `X_test`, calibration CV folds) are fixed by
  the *strategy*, never the *objective*, so the subset yields a complete
  cache while collapsing the discarded-optimization tail (multiclass: 24
  strategies x 13 objectives -> x 2). The Phase A parquet is then a
  2-objective `--resume` stub, not the scientific output.
- **Phase B** (`--from-predictions DIR`): replay the cache through
  `FrozenProbaModel`; run the strategy loop over the **full** objective
  set. No model fits. Pure CPU, embarrassingly parallel, `n_iter`-only
  cost. This is the authoritative results parquet.

Equivalence is enforced by
[`common/frozen_model_test.py`](common/frozen_model_test.py)::`test_phase_ab_matches_monolithic`
(monolithic vs Phase A→B, identical `objective_value` to 1e-12 across
every strategy x objective on `iris`), plus regression tests for each
historical bug class (fit-state key collision, `decision_function`
capability mirroring + exact replay, optimizer default-seed
determinism). `make test` must stay green.

## Search budget: `n_iter = 301`

`RANDOM_SEARCH_N_ITER = 301` (`common/config.py`). For pure random
search, P(best of `N` draws lands in the top `q`-fraction) =
`1 - (1 - q)^N`, so `N = 301` puts the selected configuration in the
top ~1% with ~95% confidence. `1000` was over-search (top 1% at
99.99%) at >3x cost for no practical gain, and `301` matches the binary
grid resolution `G = 301`. **Sensitivity arm:** rerun Phase B with the
constant overridden to `1000` (e.g. monkeypatch
`experiments.<study>.runner.RANDOM_SEARCH_N_ITER`) to show strategy
rankings are stable across budget — cheap because Phase A is reused.

## Dataset suites

- **Binary — 14:** sonar, ionosphere, wdbc, pima_diabetes,
  german_credit, pc3, ozone, madelon, spambase, phoneme, jm1,
  mammography, bank_marketing, adult. Dropped: `numerai28`,
  `philippine` (near-random / redundant); `MiniBooNE`, `tic_tac`.
- **Multiclass — 11:** iris (near-separable anchor), vehicle, cmc,
  yeast, wine_quality_red, abalone, fot_proving, satimage,
  gesture_phase, volkert, jannis. Dropped: `wine`/`car`/`pendigits`,
  `digits`/`segment`/`vowel`/`letter`, `cardiotocography`/
  `artificial_characters` (all OVR-AUC ~1.0, no strategy
  discrimination); `helena` (K=100, ~93 s/model-train long pole, added
  no discrimination the K<=10 datasets lacked); `fabert` (K=7, 800
  features → GPU calibration CV ~150 s/strat-fit, the long pole of the
  GPU campaign; dropped to keep the GPU runs tractable).
- Selection criterion: keep datasets where *measured* RF/LGBM
  OVR-weighted AUC is well below ~0.95 (ranking itself is hard). Bands
  to validate post-run are in
  [benchmark_validation_reference.md](benchmark_validation_reference.md).

## How to reproduce a campaign

All commands assume a bootstrapped machine (see
[cloud/README.md](cloud/README.md)) and a pre-warmed OpenML cache
(`prewarm_{binary,multiclass}.py`, run twice).

```bash
# CPU studies — monolithic is simplest and fastest at a single n_iter
make cloud-run STUDY=multiclass N_SEEDS=10 DEVICE=cpu        # ~45 min @80 cores, n_iter=301
make cloud-run STUDY=binary     N_SEEDS=10 DEVICE=cpu

# GPU studies — decouple: Phase A on a GPU box, Phase B on a CPU box
#   Phase A (GPU): fit + cache predictions
bash experiments/cloud/run.sh multiclass 10 --device gpu --n-jobs 4 \
     --dump-predictions /path/to/cache
#   Phase B (CPU): replay cache + strategy search (n_iter=301)
bash experiments/cloud/run.sh multiclass 10 --device gpu --n-jobs 80 \
     --from-predictions /path/to/cache
# (same two-step for STUDY=binary)

# Merge CPU + GPU per study, then regenerate
uv run python experiments/cloud/merge_results.py <cpu_run_dir> <gpu_run_dir> \
     -o experiments/results/results_<study>_merged.parquet
make tables && make win-counts && make ci-plots   # then: cd tex && latexmk carlo_master
```

## Key constants and where they live

| Constant | Value | Location |
|---|---|---|
| `RANDOM_SEARCH_N_ITER` | `301` | `experiments/common/config.py` |
| `DEFAULT_RANDOM_SEARCH_SEED` | `42` | `src/threshold_opt/optimization.py` |
| `master_seed` | `42` | runner CLI default (`--master-seed`) |
| `CV_RANDOM_STATE` | see source | `src/threshold_opt/base.py` |
| Binary / multiclass suites | 14 / 12 | `experiments/{binary,multiclass}/config.py` `DATASETS` |
| OpenML ids | — | `experiments/common/data_loader.py` `DATASET_IDS` |

## Equivalence / regression gate

```bash
uv run pytest experiments/common/frozen_model_test.py -q   # decoupling bit-identical + regressions
make test                                                  # full suite (must be green)
```
The gate runs at a small `n_iter` (monkeypatched) because record/replay
equivalence is `n_iter`-independent; this keeps it a ~20 s check rather
than the full-budget multi-minute run.
