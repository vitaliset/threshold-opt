# AGENTS.md

Master's dissertation research project on cost-sensitive decision strategies for classification.

## Architecture

Three top-level code directories:

- **`src/threshold_opt/`** — Core library. Strategies inherit from `base.DecisionStrategy` and implement `fit(X, y, objective)` + `predict(X)`. Binary strategies optimize a single threshold; multiclass strategies optimize decision boundaries in the probability simplex. `calibration.py` provides `DirichletCalibrator`; `metrics.py` handles cost matrices, `MetricObjective`, and vectorized scoring; `optimization.py` provides grid/random search.
- **`experiments/`** — Three experiment studies (binary, multiclass, rotational-bias) plus synthetic K-scaling sensitivity runners under `sensitivity/`, shared pipeline in `common/`, analysis/table generation in `analysis/`, cloud VM tooling in `cloud/`, profiling harnesses in `profiling/`.
- **`tex/`** — LaTeX dissertation source, on the IME-USP thesis template (`carlo_master.tex` → `carlo_master.pdf`, + vendored `imegoodies.sty`/`imelooks.sty`/`plainnat-ime.*`/`logos/`/`latexmkrc`); chapters live in `tex/content/`. `make pdf` builds it and **requires `biber`** (template uses biblatex/biber). `tex/supplement/` is a companion tables doc that shares the template look (imegoodies + imelooks, no biblatex) and builds separately via `make supplement`.

## Dependencies

Library dependencies (numpy, scipy, scikit-learn, joblib) are in `[project.dependencies]`. Experiment dependencies (pandas, catboost, tabm, torch, etc.) are in `[dependency-groups] experiments`. Both groups install by default via `[tool.uv] default-groups`. `uv sync` gets everything; `pip install threshold_opt` gets only the library.

## Prefer `make` over raw commands

The [Makefile](Makefile) is the canonical entry point. Run `make help` to see all targets.

Key targets:

- `make test` / `make lint` — pytest and pre-commit. Use these as the "does it still work" check; experiment runners are long-running even with `--n-seeds 1`.
- `make fetch-results` — download the published result parquets into `experiments/results/`. On a fresh clone, run it before `make tables` or `make test-freshness`.
- `make tables` / `make figures` / `make pdf` / `make all` — regenerate LaTeX tables, figures, and build the dissertation.
- `make slides-html` / `make slides-html-check` — build and check the HTML defense deck in `slides/` (reveal.js, MathJax, interactive figures; see `slides/README.md`). `make slides-figures` regenerates its win-rate bar sources from the latest results.
- `make profile` — cProfile harness for `threshold_opt` hot paths ([experiments/profiling/profile_threshold_opt.py](experiments/profiling/profile_threshold_opt.py)).
- `make timing` — per-model / per-strategy timing breakdown ([experiments/profiling/timing_analysis.py](experiments/profiling/timing_analysis.py)).
- `make experiments` — runs all three studies. **Long-running. Do not launch without explicit user approval.** Individual studies: `make experiments-binary` / `-multiclass` / `-rotational-bias`.

For flags the Makefile does not expose, drop to `uv run ...` directly (e.g. `uv run python -m experiments.profiling.profile_threshold_opt --mode binary --dataset sonar`). This project uses [uv](https://docs.astral.sh/uv/) — always prefix Python commands with `uv run`.

## Results

`experiments/results/` is gitignored. Each run writes to `run_<name>_<timestamp>/` and merges into a single parquet on completion. `--resume DIR` continues an interrupted run.

Override the output location with the `RESULTS_DIR` env var (e.g. `RESULTS_DIR=/mnt/data/results make experiments-binary`).

## Running on a cloud VM

See [experiments/cloud/README.md](experiments/cloud/README.md) for the full GCP walkthrough. Quick version:

1. `bash experiments/cloud/bootstrap.sh` on a fresh Ubuntu VM.
2. `make cloud-run STUDY=binary N_SEEDS=10` launches in detached tmux.
3. `make cloud-pull` copies results back to your laptop.
