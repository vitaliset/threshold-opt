# Experiments

Experiment pipeline for the master's dissertation on cost-sensitive decision strategies. Runs three studies comparing the strategies in [`threshold_opt`](../src/threshold_opt/) across datasets and models.

## Studies

| Study | Command | What it varies |
|-------|---------|----------------|
| Binary | `uv run python -m experiments.binary.runner` | 12 models, 12 strategies, 6 objectives, 10 datasets |
| Multiclass | `uv run python -m experiments.multiclass.runner` | 12 models, 12 strategies, custom cost matrices, 10 datasets |
| Rotational bias | `uv run python -m experiments.rotational_bias.runner` | Boundary rotation on synthetic 3-class data |

Common flags: `--n-seeds N`, `--n-jobs N`, `--resume DIR`.

## Running experiments

```bash
# Individual study
uv run python -m experiments.binary.runner --n-seeds 3 --n-jobs 4

# All studies (long-running)
make experiments
```

Results are saved incrementally to `experiments/results/run_<name>_<timestamp>/` and merged into a single parquet on completion. Resume an interrupted run with `--resume experiments/results/run_<dir>`.

Override the output directory with `RESULTS_DIR`:

```bash
RESULTS_DIR=/mnt/data/results uv run python -m experiments.binary.runner
```

## Generating tables and PDF

The result files behind the dissertation's tables and figures are attached to the [`original-version` release](https://github.com/vitaliset/threshold-opt/releases/tag/original-version). `make fetch-results` downloads them into `experiments/results/` and checks their SHA-256, so the tables can be regenerated without running the experiments.

```bash
make fetch-results  # download the published result files
make tables    # LaTeX tables from latest results
make figures   # TikZ figures from analysis scripts
make pdf       # build dissertation PDF (runs tables + figures first)
make all       # tables + figures + PDF
```

## Cloud VMs

The full sweep is too heavy for a laptop. See [cloud/README.md](cloud/README.md) for the GCP workflow (VM setup, running studies in tmux, pulling results).

## Structure

```
binary/              Binary classification runner and config
multiclass/          Multiclass runner and config
sensitivity/         K-scaling sensitivity runners (variance / mean-shift) + dimensional-dilution lemma test
rotational_bias/     Rotational-bias runner
common/              Shared pipeline, data loading, thread limits
analysis/            Table/figure generation + freshness tests
profiling/           cProfile and timing harnesses
cloud/               GCP VM bootstrap, run, status, merge scripts
results/             Parquet output (gitignored; filled by the runners or by make fetch-results)
```

## Development

```bash
make test      # run pytest
make lint      # ruff + ruff-format + chktex
make profile   # cProfile harness for threshold_opt hot paths
make timing    # per-model / per-strategy timing breakdown
```
