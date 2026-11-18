.PHONY: help all fetch-results tables ci-plots win-counts figures figures-sampler-analysis pdf supplement \
       slides-figures \
       slides-html slides-html-figures slides-html-check slides-html-shots slides-html-pdf \
       test test-e2e test-freshness test-provenance lint check-refs clean setup profile timing \
       experiments experiments-binary experiments-multiclass experiments-rotational-bias \
       sensitivity-experiments \
       lean-setup lean-build lean-audit lean-clean \
       cloud-bootstrap cloud-run cloud-status cloud-kill cloud-pull

STUDY ?= binary
N_SEEDS ?= 10
MODELS ?=
DEVICE ?=
N_JOBS ?=
INSTANCE ?= mestrado-runner
ZONE ?= us-central1-a
REMOTE_USER ?= $(USER)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

all: tables pdf supplement  ## Build tables + dissertation PDF + per-dataset-tables supplement PDF

setup:  ## Install uv and sync dependencies
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync

experiments: experiments-binary experiments-multiclass experiments-rotational-bias  ## Run all experiment studies

fetch-results:  ## Download the published result files into experiments/results/
	uv run python -m experiments.fetch_results

tables:  ## Generate LaTeX tables from latest results
	uv run python -m experiments.analysis.generate_tables

ci-plots:  ## Generate the inline CI plots used in chapters 4 and 8 (the SHOWCASE_PLOTS set)
	uv run python -m experiments.analysis.generate_ci_plots

win-counts:  ## Generate per-strategy win/tie-count heatmaps (chapters 4 and 8)
	uv run python -m experiments.analysis.generate_win_counts

figures:  ## Generate TikZ figures from analysis scripts (excludes sampler analysis; simplex-regions refits 5 strategies and is the slow entry)
	uv run python -m experiments.analysis.generate_partition_figure
	uv run python -m experiments.analysis.generate_marchetti_figure
	uv run python -m experiments.analysis.generate_cost_partition_figure
	uv run python -m experiments.analysis.generate_cost_partition_3d_figure
	uv run python -m experiments.analysis.generate_obrien_figure
	uv run python -m experiments.analysis.generate_variance_figure
	uv run python -m experiments.analysis.generate_meanshift_figure
	uv run python -m experiments.analysis.generate_lemma_test_table
	uv run python -m experiments.analysis.generate_simplex_regions_figure

figures-sampler-analysis:  ## Full Voronoi-sampler appendix run: overlay + K-sweep up to K=100 (minutes)
	uv run python -m experiments.analysis.generate_sampler_overlay_figure \
	    --plot-metrics

pdf: tables figures  ## Build the dissertation PDF (runs tables + figures first; needs biber)
	cd tex && latexmk carlo_master

supplement: tables  ## Build the per-dataset-tables companion PDF (tex/supplement/supplement.pdf)
	cd tex/supplement && latexmk -pdf supplement.tex

slides-figures:  ## Generate the slide-sized win-rate bar charts (slides/figures/tex/)
	uv run python -m experiments.analysis.generate_win_counts --slides

slides-html:  ## Build the HTML defense deck (slides/index.html; opens offline)
	uv run --no-project python slides/build.py

slides-html-figures:  ## Compile the static figures of the HTML deck to SVG (needs pdflatex, pdftocairo)
	uv run --no-project python slides/tools/build_figures.py

slides-html-check: slides-html  ## Check the HTML deck: formulas of the built page, slides that overflow, MathJax errors
	uv run --no-project python slides/tools/check.py

slides-html-shots: slides-html  ## Screenshot every slide of the HTML deck (slides/.build/shots/)
	uv run --no-project --with pillow python slides/tools/shots.py

slides-html-pdf: slides-html  ## Fallback PDF of the HTML deck, one page per slide (slides/.build/slides-html.pdf)
	uv run --no-project --with pillow python slides/tools/shots.py --pdf

test:  ## Run pytest
	uv run pytest

test-e2e:  ## Run all tests including integration/smoke tests
	uv run pytest --integration -v

test-freshness:  ## Check committed tex/tables/ match current code + latest parquet
	uv run pytest experiments/analysis/freshness_test.py --freshness -v

test-provenance:  ## Check the git metadata of the result parquets against the current code (after a new campaign)
	uv run pytest experiments/analysis/freshness_test.py --provenance -m provenance -v

test-reproducibility:  ## Run end-to-end determinism tests (same seed -> same output)
	uv run pytest --reproducibility -m reproducibility -v

lint:  ## Run pre-commit (ruff, ruff-format, chktex)
	uv run pre-commit run --all-files

check-refs:  ## Audit \ref forward references in the dissertation
	uv run python tex/scripts/find_forward_refs.py

clean:  ## Clean LaTeX build artifacts (main + supplement)
	cd tex && latexmk -C
	cd tex/supplement && latexmk -C

profile:  ## cProfile harness for threshold_opt hot paths (defaults)
	uv run python -m experiments.profiling.profile_threshold_opt

timing:  ## Per-model / per-strategy timing breakdown (binary)
	uv run python -m experiments.profiling.timing_analysis

# Long-running experiment targets (convenience only)
experiments-binary:  ## Binary classification study
	uv run python -m experiments.binary.runner

experiments-multiclass:  ## Multiclass classification study
	uv run python -m experiments.multiclass.runner

experiments-rotational-bias:  ## Rotational-bias study
	uv run python -m experiments.rotational_bias.runner

sensitivity-experiments:  ## Sensitivity sweep: scaling-K (variance + mean-shift, analytical). Minutes.
	uv run python -m experiments.sensitivity.sensitivity_runner --mode variance --cost asymmetric_10x $(if $(N_SEEDS),--n-seeds $(N_SEEDS))
	uv run python -m experiments.sensitivity.sensitivity_runner --mode meanshift --cost symmetric $(if $(N_SEEDS),--n-seeds $(N_SEEDS))
	uv run python -m experiments.sensitivity.sensitivity_runner --mode meanshift --cost asymmetric_2x $(if $(N_SEEDS),--n-seeds $(N_SEEDS))
	uv run python -m experiments.sensitivity.sensitivity_runner --mode meanshift --cost asymmetric_5x $(if $(N_SEEDS),--n-seeds $(N_SEEDS))
	uv run python -m experiments.sensitivity.sensitivity_runner --mode meanshift --cost asymmetric_10x $(if $(N_SEEDS),--n-seeds $(N_SEEDS))

lemma-test:  ## Synthetic verification of dimensional-dilution Lemma: uniform-on-sphere u in V, K-decay exponent of E[(phi(p+u)-phi(p))^2]/E[||g||^2]. Seconds.
	uv run python -m experiments.sensitivity.lemma_test_runner $(if $(N_SEEDS),--n-seeds $(N_SEEDS))

# Lean formalization (see lean/README.md)
# Deliberately not wired into `tables`: tex/tables/formalization.tex is
# committed, so `make pdf` works without a Lean toolchain installed.
lean-setup:  ## Install elan and fetch the Mathlib build cache (first run, GBs)
	curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
	cd lean && lake exe cache get

lean-build:  ## Type-check the Lean formalization (lake build)
	cd lean && lake build

lean-audit: lean-build  ## Axiom provenance -> lean/audit.json + tex/tables/formalization.tex
	cd lean && lake exe audit
	uv run python lean/scripts/render_table.py

lean-clean:  ## Remove Lean build artifacts (lake clean)
	cd lean && lake clean

# Cloud VM helpers (see experiments/cloud/ and AGENTS.md for workflow)
cloud-bootstrap:  ## Provision a fresh Ubuntu VM (uv + deps + system pkgs)
	bash experiments/cloud/bootstrap.sh

cloud-run:  ## Launch STUDY in a detached tmux session (STUDY=binary N_SEEDS=10 DEVICE=cpu|gpu)
	bash experiments/cloud/run.sh $(STUDY) $(N_SEEDS) $(if $(DEVICE),--device $(DEVICE)) $(if $(N_JOBS),--n-jobs $(N_JOBS)) $(if $(MODELS),--models '$(MODELS)')

cloud-status:  ## Progress snapshot for the newest run (elapsed, done count, tqdm tail)
	bash experiments/cloud/status.sh

cloud-kill:  ## Kill the tmux session for STUDY (leaves already-written results in place)
	tmux kill-session -t run-$(STUDY) && echo "killed run-$(STUDY)" \
	  || echo "no session named run-$(STUDY)"

cloud-pull:  ## gcloud scp experiments/results/ from a GCP VM (INSTANCE/ZONE/REMOTE_USER overridable)
	bash experiments/cloud/pull-results.sh $(INSTANCE) $(ZONE) $(REMOTE_USER)
