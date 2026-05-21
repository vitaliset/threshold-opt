"""Generate per-(dataset, model, objective) CI plots from experiment results.

Each output file is a self-contained ``\\input``-able TikZ/pgfplots forest plot
showing one (dataset, model, objective) cell: strategies on the y-axis, mean
objective value on the x-axis, black dot + 1-SE error bar per strategy. Rows
are sorted by mean; the winner and any strategy whose mean $\\pm$ SE interval
overlaps the winner's are bolded, matching the per-dataset appendix tables.

Used by chapters 4 and 8 of the dissertation to embed the showcase plots
listed in ``SHOWCASE_PLOTS``. The CLI default emits exactly those plots;
passing ``--datasets`` / ``--models`` / ``--objectives`` switches into a
general filter mode for ad-hoc plotting (one file per (dataset, model,
objective) combination present in the data after filtering).

Usage:
    uv run python -m experiments.analysis.generate_ci_plots
    uv run python -m experiments.analysis.generate_ci_plots \\
        --datasets adult --models LGBMClassifier\\(\\) --objectives F1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analysis.generate_tables import (
    SCOPE_CONFIG,
    _apply_legacy_strategy_renames,
    _find_latest_parquet,
    _normalize_balanced_error_objective,
    _sanitize_filename,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "tex" / "figures" / "ci_plots"

# Display labels for base classifiers, used for the model name in captions.
MODEL_LABELS: dict[str, str] = {
    "LGBMClassifier()": "LGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "RandomForest": "RandomForest",
    "ExtraTrees": "ExtraTrees",
    "LogisticRegression()": "LogReg",
    "GaussianNB": "GaussianNB",
    "TabM": "TabM",
    "TabPFN": "TabPFN",
    "TabICL": "TabICL",
}

VALID_X_SCALES: tuple[str, ...] = ("linear", "log", "sqrt")

# One illustrative (dataset, model, objective) cell is embedded per
# chapter section, resolved deterministically from the latest parquet so
# the choice is reproducible and not hand-picked. The chosen cell is the
# most representative one: the cell whose per-strategy ordering is
# closest (Spearman rank correlation) to the campaign-wide mean-rank
# ordering for that scope and objective family, rather than an extreme
# or best-case cell. Each ``SELECTION_PURPOSES`` entry is
# ``(scope, family, purpose)``.
SELECTION_PURPOSES: tuple[tuple[str, str, str], ...] = (
    ("binary", "cost", "representative"),
    ("binary", "max", "representative"),
    ("multiclass", "cost", "representative"),
    ("multiclass", "max", "representative"),
)

# Used only when no parquet is available (e.g. unit tests that import the
# module without a results directory). Never written from real runs.
_STATIC_SHOWCASE_FALLBACK: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "binary",
        "spambase",
        "LGBMClassifier()",
        "Asymmetric 10x Cost",
        "linear",
    ),
    ("binary", "mammography", "CatBoost", "F2", "linear"),
    ("multiclass", "vehicle", "CatBoost", "Asymmetric 100x Cost", "log"),
    ("multiclass", "vehicle", "CatBoost", "F0.5 Macro", "linear"),
)


# A candidate cell must carry at least this many strategies, and at
# least this fraction of the strategies seen campaign-wide, for its
# ordering to be a meaningful match to the global ordering (so a
# sparsely-populated cell cannot win on a coincidental partial match).
_MIN_STRATEGIES = 5
_MIN_STRATEGY_FRACTION = 0.5


def _cell_strategy_means(grp: pd.DataFrame) -> pd.Series:
    """Per-strategy mean objective value for one (dataset, model,
    objective) group, NaN strategies dropped."""
    return grp.groupby("strategy_name")["objective_value"].mean().dropna()


def _within_cell_ranks(means: pd.Series, family: str) -> pd.Series:
    """Rank strategies within a cell, best = rank 1 (ascending for cost,
    where lower is better; descending for the higher-is-better
    maximization metrics); ties take the average rank."""
    return means.rank(method="average", ascending=family == "cost")


def _global_mean_ranks(df: pd.DataFrame, family: str) -> pd.Series:
    """Mean within-cell rank of each strategy across every
    (dataset, model, objective) cell of ``family`` in ``df``. This
    scale-free aggregate is the campaign-wide ordering the illustrative
    cell is matched against."""
    rank_sums: dict[str, float] = {}
    rank_counts: dict[str, int] = {}
    for _key, grp in df.groupby(
        ["dataset_name", "model_name", "optimization_objective_name"],
        sort=False,
    ):
        ranks = _within_cell_ranks(_cell_strategy_means(grp), family)
        for strat, r in ranks.items():
            rank_sums[strat] = rank_sums.get(strat, 0.0) + float(r)
            rank_counts[strat] = rank_counts.get(strat, 0) + 1
    return pd.Series({s: rank_sums[s] / rank_counts[s] for s in rank_sums})


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation between two strategy-keyed Series over
    their common index; ``-inf`` when undefined (fewer than three common
    strategies, or a constant input)."""
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return float("-inf")
    x = a.loc[common].rank().to_numpy()
    y = b.loc[common].rank().to_numpy()
    if x.std() == 0 or y.std() == 0:
        return float("-inf")
    return float(np.corrcoef(x, y)[0, 1])


def _cell_representativeness(
    grp: pd.DataFrame, family: str, global_ranks: pd.Series
) -> float:
    """How closely one cell's per-strategy ordering matches the
    campaign-wide ordering (Spearman rank correlation). Cells carrying
    too few strategies score ``-inf``."""
    means = _cell_strategy_means(grp)
    common = set(means.index) & set(global_ranks.index)
    if len(common) < _MIN_STRATEGIES or len(common) < (
        _MIN_STRATEGY_FRACTION * len(global_ranks)
    ):
        return float("-inf")
    return _spearman(_within_cell_ranks(means, family), global_ranks)


def _resolve_x_scale(family: str, means: list[float]) -> str:
    """Cost cells whose strategy means span more than 8x use a log x
    axis so the cost-aware cluster stays legible next to the high-cost
    outliers; max cells always use a linear axis."""
    if family != "cost":
        return "linear"
    positive = [v for v in means if v > 0]
    if len(positive) < 2 or len(positive) != len(means):
        return "linear"
    return "log" if max(positive) / min(positive) > 8.0 else "linear"


def select_illustrative_cell(
    scope: str,
    family: str,
    purpose: str,
    parquet_path: Path | None = None,
) -> tuple[str, str, str, str, str]:
    """Return the most representative ``(scope, dataset, model,
    objective, x_scale)`` cell for ``family`` under ``scope``: the cell
    whose per-strategy ordering best correlates (Spearman) with the
    campaign-wide mean-rank ordering over all ``family`` cells.

    Deterministic: on a Spearman tie (within 1e-9) the lexicographically
    first ``(dataset, model, objective)`` wins, so the same parquet
    always resolves to the same cell.
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    if purpose != "representative":
        raise ValueError(f"Unknown selection purpose {purpose!r}")
    cfg = SCOPE_CONFIG[scope]
    if parquet_path is None:
        parquet_path = _find_latest_parquet(cfg["parquet_pattern"])
    objectives = (
        set(cfg["cost_objectives"])
        if family == "cost"
        else set(cfg["max_metrics"])
    )
    df = pd.read_parquet(parquet_path)
    df = _apply_legacy_strategy_renames(df)
    df = _normalize_balanced_error_objective(df)
    # Restrict to the curated dataset set used by the tables and
    # win-count figures (so the cell can never be a dataset absent from
    # the chapter, e.g. fabert) and to this objective family.
    df = df[
        df["dataset_name"].isin(cfg["datasets"])
        & df["optimization_objective_name"].isin(objectives)
    ]
    if df.empty:
        raise ValueError(
            f"No {family} cells for scope={scope} in {parquet_path}"
        )

    global_ranks = _global_mean_ranks(df, family)
    best_key: tuple[str, str, str] | None = None
    best_score = float("-inf")
    for key, grp in sorted(
        df.groupby(
            ["dataset_name", "model_name", "optimization_objective_name"],
            sort=False,
        ),
        key=lambda kv: kv[0],
    ):
        score = _cell_representativeness(grp, family, global_ranks)
        if score > best_score + 1e-9:
            best_score, best_key = score, key

    if best_key is None:
        raise ValueError(
            f"No representative cell for scope={scope} family={family} "
            f"in {parquet_path}"
        )

    dataset, model, objective = best_key
    cell = df[
        (df["dataset_name"] == dataset)
        & (df["model_name"] == model)
        & (df["optimization_objective_name"] == objective)
    ]
    means = [float(v) for v in _cell_strategy_means(cell)]
    x_scale = _resolve_x_scale(family, means)
    return (scope, dataset, model, objective, x_scale)


def showcase_filename_and_label(
    dataset: str, objective: str, x_scale: str
) -> tuple[str, str]:
    """The .tex filename and the ``\\label`` ``write_ci_plots`` /
    ``render_plot`` will emit for a resolved cell, so the prose can cite
    a stable target."""
    file_suffix = "" if x_scale == "linear" else f"_{x_scale}"
    label_suffix = "" if x_scale == "linear" else f"-{x_scale}"
    ds = _sanitize_filename(dataset)
    obj = _sanitize_filename(objective)
    return (
        f"{ds}_{obj}{file_suffix}.tex",
        f"fig:ci-{ds}-{obj}{label_suffix}",
    )


def resolve_showcase_plots(
    parquet_paths: dict[str, Path] | None = None,
) -> tuple[tuple[str, str, str, str, str], ...]:
    """Resolve every ``SELECTION_PURPOSES`` entry against the latest
    parquet for its scope. Falls back to ``_STATIC_SHOWCASE_FALLBACK``
    when no parquet is available so the module imports without a
    results directory (unit tests)."""
    resolved: list[tuple[str, str, str, str, str]] = []
    for scope, family, purpose in SELECTION_PURPOSES:
        pq = None if parquet_paths is None else parquet_paths.get(scope)
        try:
            resolved.append(
                select_illustrative_cell(scope, family, purpose, pq)
            )
        except FileNotFoundError:
            return _STATIC_SHOWCASE_FALLBACK
    return tuple(resolved)


def pivot_dataset_objective(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot a (dataset, objective) slice into (strategy x model) mean and SE.

    SE is ``std / sqrt(count)`` when ``count > 1`` and falls back to ``std``
    when only one observation is available, mirroring ``format_latex_row``.
    """
    df_mean = df.pivot_table(
        index="strategy_name",
        columns="model_name",
        values="objective_value",
        aggfunc="mean",
    )
    df_std = df.pivot_table(
        index="strategy_name",
        columns="model_name",
        values="objective_value",
        aggfunc="std",
    )
    df_count = df.pivot_table(
        index="strategy_name",
        columns="model_name",
        values="objective_value",
        aggfunc="count",
    )
    df_std = df_std.reindex_like(df_mean)
    df_count = df_count.reindex_like(df_mean)

    sqrt_n = np.sqrt(df_count.where(df_count > 1, 1))
    df_se = df_std.divide(sqrt_n)
    # n <= 1 fallback: use std as the displayed half-width.
    df_se = df_se.where(df_count > 1, df_std)
    return df_mean, df_se


# Computed at import against the latest parquet (static fallback when the
# results directory is absent). Each entry is
# ``(scope, dataset, model, objective, x_scale)``. Placed after
# ``pivot_dataset_objective`` because the resolver depends on it.
SHOWCASE_PLOTS: tuple[tuple[str, str, str, str, str], ...] = (
    resolve_showcase_plots()
)


def render_plot(
    *,
    dataset: str,
    objective: str,
    family: str,
    df_mean: pd.DataFrame,
    df_se: pd.DataFrame,
    x_scale: str = "linear",
) -> str:
    """Render one (dataset, model, objective) plot as a self-contained .tex.

    Single-model layout: each strategy occupies a labelled y-row; one black
    dot + horizontal $\\pm 1$ SE bar per strategy. Strategies are sorted by
    mean (best at top: lowest cost on cost objectives, highest score on
    maximization metrics). The winner and any strategy whose mean $\\pm$ SE
    interval overlaps the winner's interval are bolded, matching the
    per-dataset appendix table convention.

    The pivot's column axis must contain exactly one model. ``write_ci_plots``
    enforces this by filtering before calling.
    """
    if df_mean.shape[1] != 1:
        raise ValueError(
            "render_plot expects a single-model slice; got "
            f"{df_mean.shape[1]} models: {list(df_mean.columns)}"
        )
    if x_scale not in VALID_X_SCALES:
        raise ValueError(
            f"Unknown x_scale {x_scale!r}; expected one of {VALID_X_SCALES}"
        )
    model = df_mean.columns[0]
    direction = "higher is better" if family == "max" else "lower is better"
    is_max = family == "max"

    # Filter to strategies with a numeric mean for this model and sort them
    # by performance (best first).
    valid: list[tuple[str, float, float]] = []
    for strat in df_mean.index:
        m = df_mean.at[strat, model]
        if pd.isna(m):
            continue
        se = df_se.at[strat, model]
        se_val = 0.0 if pd.isna(se) else float(se)
        valid.append((strat, float(m), se_val))
    valid.sort(key=lambda t: -t[1] if is_max else t[1])
    if not valid:
        raise ValueError(
            f"No valid strategies with non-NaN means on {dataset} / {model} "
            f"/ {objective}"
        )

    # Identify the winner (rank 0) and any strategy whose mean $\pm$ SE
    # interval overlaps the winner's.
    win_strat, win_mean, win_se = valid[0]
    win_lo, win_hi = win_mean - win_se, win_mean + win_se
    tied: set[str] = set()
    for strat, m, se in valid[1:]:
        lo, hi = m - se, m + se
        if lo <= win_hi and win_lo <= hi:
            tied.add(strat)

    height_cm = max(5.5, 0.85 * len(valid) + 1.2)

    # ``\addplot`` (no trailing ``+``) opts out of pgfplots' cycle list so
    # ``color=black`` actually wins.
    plot_blocks: list[str] = []
    for strat, m_val, se_val in valid:
        coord = f"({m_val:.6g},{{{strat}}}) +- ({se_val:.6g},0)"
        plot_blocks.append(
            "\\addplot["
            "only marks, mark=*, mark size=2pt, color=black, "
            "error bars/x dir=both, error bars/x explicit, "
            "error bars/error bar style={color=black, line width=0.5pt}"
            "] coordinates { " + coord + " };"
        )

    # Brace-wrap each entry so strategy names with special characters parse
    # cleanly as symbolic y coords.
    strategy_list = ",".join(f"{{{s}}}" for s, _, _ in valid)

    def _decorate(strat: str) -> str:
        if strat == win_strat or strat in tied:
            return f"\\textit{{{strat}}}"
        return strat

    yticklabels_list = ",".join(_decorate(s) for s, _, _ in valid)

    scale_suffix = "" if x_scale == "linear" else f"-{x_scale}"
    label = (
        f"fig:ci-{_sanitize_filename(dataset)}"
        f"-{_sanitize_filename(objective)}{scale_suffix}"
    )
    ds_label = dataset.replace("_", "\\_")
    model_label = MODEL_LABELS.get(model, model)
    scale_caption = {
        "linear": "",
        "log": " The x axis is on a log scale.",
        "sqrt": " The x axis is on a square-root scale.",
    }[x_scale]
    if x_scale == "log":
        # Pure base-2 doubling ticks (..., 0.25, 0.5, 1, 2, 4, ...) so
        # the labels stay readable and uniformly spaced on the log axis,
        # rather than the auto log-tick decimals (10^-0.6 etc.).
        import math

        data_vals = [m for _, m, _ in valid]
        log_min = min(data_vals) / 1.2
        log_max = max(data_vals) * 1.5
        candidate_ticks: list[float] = []
        k = math.ceil(math.log2(log_max))
        while True:
            t = 2.0**k
            if t < log_min:
                break
            if log_min <= t <= log_max:
                candidate_ticks.append(t)
            k -= 1
        candidate_ticks.sort()
        # Drop trailing zeros so labels read 0.5 instead of 0.50.
        tick_str = ",".join(f"{t:g}" for t in candidate_ticks)
        scale_axis_options = (
            "    xmode=log, log basis x=10,\n"
            "    log ticks with fixed point,\n"
            f"    xtick={{{tick_str}}},\n"
        )
    elif x_scale == "sqrt":
        # Apply x = sqrt(value) when plotting; inverse for ticks. This
        # compresses the right tail (outliers) while expanding the left
        # cluster, without log's powers-of-10 tick labels.
        scale_axis_options = (
            "    x coord trafo/.code="
            "{\\pgfmathparse{sqrt(#1)}\\pgfmathresult},\n"
            "    x coord inv trafo/.code="
            "{\\pgfmathparse{(#1)*(#1)}\\pgfmathresult},\n"
        )
    else:
        scale_axis_options = ""

    return (
        "% Auto-generated by experiments.analysis.generate_ci_plots"
        " --- do not edit.\n"
        "\\begin{figure}[H]\n"
        "\\centering\n"
        # adjustbox scales the picture down to textwidth when long y-axis
        # labels would otherwise push it past the right margin. It never
        # scales up, so plots that already fit are unchanged.
        "\\adjustbox{max width=\\textwidth}{%\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        f"    width=0.85\\textwidth, height={height_cm:.2f}cm,\n"
        f"    xlabel={{Mean {objective} ({direction})}},\n"
        f"{scale_axis_options}"
        f"    symbolic y coords={{{strategy_list}}},\n"
        # ytick=data picks ticks from a single plot's coords; we have one
        # \\addplot per strategy so we must enumerate every strategy by name
        # to get all rows labelled.
        f"    ytick={{{strategy_list}}},\n"
        f"    yticklabels={{{yticklabels_list}}},\n"
        "    y dir=reverse,\n"
        "    enlarge y limits=0.08,\n"
        "    xmajorgrids=true,\n"
        "    major grid style={draw=black!15, line width=0.3pt},\n"
        "    scaled x ticks=false,\n"
        "    xticklabel style={/pgf/number format/fixed,"
        " /pgf/number format/precision=3},\n"
        "    tick label style={font=\\footnotesize},\n"
        "    label style={font=\\small},\n"
        "    every axis plot/.append style={thick},\n"
        "]\n" + "\n".join(plot_blocks) + "\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}%\n"
        "}\n"
        f"\\caption{{{ds_label}: {objective} on "
        f"{model_label} ({direction}). "
        "Each dot is a strategy's mean over ten seeds with a "
        "$\\pm 1$ SE horizontal bar; rows are sorted by mean, and the "
        "best strategy and any whose interval overlaps it are italicized."
        f"{scale_caption}}}%\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def write_ci_plots(
    parquet_path: Path | None = None,
    plots_dir: Path | None = None,
    scope: str = "binary",
    dataset_filter: tuple[str, ...] | None = None,
    objective_filter: tuple[str, ...] | None = None,
    model_filter: tuple[str, ...] | None = None,
    x_scale: str = "linear",
) -> list[Path]:
    """Generate per-(dataset, model, objective) CI plot files for one scope.

    Each (dataset, model, objective) cell present in the filtered data
    produces one ``.tex`` file. ``model_filter`` is required (or implied
    via ``SHOWCASE_PLOTS``) to restrict each cell to a single model so the
    forest-plot layout is well-defined.
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    cfg = SCOPE_CONFIG[scope]
    cost_set = set(cfg["cost_objectives"])
    max_set = set(cfg["max_metrics"])

    if parquet_path is None:
        parquet_path = _find_latest_parquet(cfg["parquet_pattern"])
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = _apply_legacy_strategy_renames(df)
    df = _normalize_balanced_error_objective(df)
    if dataset_filter is not None:
        df = df[df["dataset_name"].isin(dataset_filter)]
    if objective_filter is not None:
        df = df[df["optimization_objective_name"].isin(objective_filter)]
    if model_filter is not None:
        df = df[df["model_name"].isin(model_filter)]

    written: list[Path] = []
    for key, grp in df.groupby(
        ["dataset_name", "model_name", "optimization_objective_name"],
        sort=False,
    ):
        dataset_name = key[0]
        obj_name = key[2]
        if obj_name in cost_set:
            family = "cost"
        elif obj_name in max_set:
            family = "max"
        else:
            continue

        df_mean, df_se = pivot_dataset_objective(grp)
        # The groupby already restricts to one model per cell, so the
        # pivot is single-column by construction.
        tex = render_plot(
            dataset=dataset_name,
            objective=obj_name,
            family=family,
            df_mean=df_mean,
            df_se=df_se,
            x_scale=x_scale,
        )

        scale_suffix = "" if x_scale == "linear" else f"_{x_scale}"
        fname = (
            f"{_sanitize_filename(dataset_name)}_"
            f"{_sanitize_filename(obj_name)}{scale_suffix}.tex"
        )
        out = plots_dir / fname
        out.write_text(tex, encoding="utf-8")
        written.append(out)
        print(f"  Wrote {out}")

    return written


def write_showcase_plots(
    plots_dir: Path | None = None,
    showcase: tuple[tuple[str, str, str, str, str], ...] | None = None,
) -> list[Path]:
    """Generate exactly the curated showcase plots used by chapters 4 and 8.

    Each entry is ``(scope, dataset, model, objective, x_scale)``.
    """
    if showcase is None:
        showcase = SHOWCASE_PLOTS
    written: list[Path] = []
    for scope, dataset, model, objective, x_scale in showcase:
        written.extend(
            write_ci_plots(
                plots_dir=plots_dir,
                scope=scope,
                dataset_filter=(dataset,),
                model_filter=(model,),
                objective_filter=(objective,),
                x_scale=x_scale,
            )
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the curated chapter-4 / chapter-8 inline CI plots."
            " By default emits one .tex per entry in SHOWCASE_PLOTS."
            " Pass --datasets / --models / --objectives to bypass the"
            " showcase and use the general filter machinery instead."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("binary", "multiclass", "all"),
        default=None,
        help=(
            "Filter mode only: which experimental campaign to regenerate"
            " plots for. Default (showcase mode) ignores this flag."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Filter mode only: datasets to plot.",
    )
    parser.add_argument(
        "--objectives",
        nargs="*",
        default=None,
        help="Filter mode only: objectives to plot.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Filter mode only: base classifiers to include.",
    )
    parser.add_argument(
        "--print-selection",
        action="store_true",
        help=(
            "Print the deterministically resolved showcase cells (scope, "
            "dataset, model, objective, x_scale) with the .tex filename "
            "and \\label each will emit, then exit. Use this to keep the "
            "prose \\input / \\ref targets in sync."
        ),
    )
    args = parser.parse_args()

    if args.print_selection:
        for scope, dataset, model, objective, x_scale in SHOWCASE_PLOTS:
            fname, label = showcase_filename_and_label(
                dataset, objective, x_scale
            )
            print(
                f"{scope:10s} {dataset} / {model} / {objective} "
                f"[{x_scale}]\n"
                f"    file : figures/ci_plots/{fname}\n"
                f"    label: {label}"
            )
        return

    in_filter_mode = (
        args.scope is not None
        or args.datasets is not None
        or args.objectives is not None
        or args.models is not None
    )

    if not in_filter_mode:
        print(f"Generating {len(SHOWCASE_PLOTS)} showcase plots...")
        write_showcase_plots()
    else:
        scope = args.scope or "binary"
        dataset_filter = tuple(args.datasets) if args.datasets else None
        objective_filter = tuple(args.objectives) if args.objectives else None
        model_filter = tuple(args.models) if args.models else None
        scopes = ("binary", "multiclass") if scope == "all" else (scope,)
        for s in scopes:
            print(f"Generating CI plots for scope={s} (filter mode)...")
            write_ci_plots(
                scope=s,
                dataset_filter=dataset_filter,
                objective_filter=objective_filter,
                model_filter=model_filter,
            )

    print("Done.")


if __name__ == "__main__":
    main()
