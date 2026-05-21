"""Generate LaTeX tables from experiment results.

Per-dataset strategy tables are written to ``tex/supplement/tables/`` (used by
the standalone companion supplement, ``tex/supplement/supplement.tex``).
Summary tables (dataset stats, rotational-bias) are written to ``tex/tables/``
and inputted by the main dissertation.

Usage:
    uv run python -m experiments.analysis.generate_tables  # all tables
    uv run python -m experiments.analysis.generate_tables --only strategy
    uv run python -m experiments.analysis.generate_tables \\
        --only dataset-stats
    uv run python -m experiments.analysis.generate_tables \\
        --only rotational-bias
    uv run python -m experiments.analysis.generate_tables \\
        --only model-auc
    uv run python -m experiments.analysis.generate_tables \\
        --only model-ece
    uv run python -m experiments.analysis.generate_tables \\
        --only model-vertex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analysis.data_stats import (
    BINARY_DATASETS,
    MULTICLASS_DATASETS,
    build_dataset_stats_latex,
)
from experiments.analysis.latex_formatting import (
    build_latex_tables_per_dataset,
    format_latex_row,
)
from experiments.binary.config import MODELS as BINARY_MODELS
from experiments.multiclass.config import MODELS as MULTICLASS_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
TABLES_DIR = REPO_ROOT / "tex" / "tables"
SUPPLEMENT_TABLES_DIR = REPO_ROOT / "tex" / "supplement" / "tables"

SCOPE_CONFIG: dict[str, dict] = {
    "binary": {
        "parquet_pattern": "results_binary_*.parquet",
        "datasets": tuple(BINARY_DATASETS),
        "max_metrics": (
            "F1",
            "MCC",
            "F2",
            "F0.5",
            "GMean",
        ),
        "cost_objectives": (
            "Symmetric Cost",
            "Asymmetric 10x Cost",
            "Asymmetric 100x Cost",
            "Non-Zero Diagonal Cost",
            "Balanced Error",
        ),
    },
    "multiclass": {
        "parquet_pattern": "results_multiclass_*.parquet",
        "datasets": tuple(MULTICLASS_DATASETS),
        "max_metrics": (
            "F1 Macro",
            "F2 Macro",
            "F0.5 Macro",
            "MCC",
            "GMean",
        ),
        "cost_objectives": (
            "Symmetric Cost",
            "Asymmetric 10x Cost",
            "Asymmetric 100x Cost",
            "Custom Cost",
            "Ordinal Cost",
            "Quadratic Cost",
            "Balanced Error",
        ),
    },
}
# Backward-compat aliases; some callers (tests) import these directly.
MAXIMIZATION_METRICS = SCOPE_CONFIG["binary"]["max_metrics"]
COST_OBJECTIVES = SCOPE_CONFIG["binary"]["cost_objectives"]
SHORTEN_MODEL_NAMES = {
    "LGBMClassifier": "LGBM",
}

# Column order + display labels for the base-estimator ROC AUC table.
# Order follows each study config's ``MODELS`` insertion order (CPU
# models then GPU models); labels are the short names used in
# ``experiments/benchmark_validation_reference.md``. Unmapped keys fall
# back to themselves.
MODEL_DISPLAY_NAMES = {
    "LGBMClassifier()": "LGBM",
    "LogisticRegression()": "LogReg",
    "XGBoost": "XGB",
    "CatBoost": "CatBoost",
    "RandomForest": "RF",
    "ExtraTrees": "ET",
    "GaussianNB": "GaussNB",
    "TabM": "TabM",
    "TabPFN": "TabPFN",
    "TabICL": "TabICL",
}
MODEL_ORDER: dict[str, tuple[str, ...]] = {
    "binary": tuple(BINARY_MODELS),
    "multiclass": tuple(MULTICLASS_MODELS),
}

ROTATIONAL_BIAS_STRATEGIES = (
    "Argmax",
    "ThresholdRatio",
    "MultidimensionalThreshold",
    "VoronoiTuning",
    "FullMatrixPartitioning",
)
ROTATIONAL_BIAS_MAXIMIZATION = ("Macro-F1", "Accuracy")

# Legacy display labels written to ``results_rotational_bias_*.parquet``
# before the runner keys were aligned with the multiclass benchmark.
# Applied at load time so the table renders the new names without
# requiring a re-run of the rotational-bias study.
LEGACY_ROTATIONAL_BIAS_RENAMES = {
    "MultiDimThreshold": "MultidimensionalThreshold",
    "Voronoi": "VoronoiTuning",
    "Default": "Argmax",
}

# Legacy strategy names written to ``results_binary_*.parquet`` and
# ``results_multiclass_*.parquet`` before the Default-to-Argmax rename.
# Applied at load time so existing parquets render with the new names
# without re-running the campaigns.
LEGACY_STRATEGY_RENAMES = {
    "Default": "Argmax",
    "IsotonicDefault": "IsotonicArgmax",
    "PlattScalingDefault": "PlattScalingArgmax",
    "TemperatureScalingDefault": "TemperatureScalingArgmax",
    "DirichletCalibrationDefault": "DirichletCalibrationArgmax",
}


def _apply_legacy_strategy_renames(df: "pd.DataFrame") -> "pd.DataFrame":
    """Apply ``LEGACY_STRATEGY_RENAMES`` to the ``strategy_name`` column."""
    if "strategy_name" in df.columns:
        df = df.assign(
            strategy_name=df["strategy_name"].replace(LEGACY_STRATEGY_RENAMES)
        )
    return df


def _normalize_balanced_error_objective(df: "pd.DataFrame") -> "pd.DataFrame":
    """Collapse the two equivalent Balanced Error objectives into one.

    The binary and multiclass configs run both ``Balanced Error`` (a
    ``MetricObjective`` wrapping ``1 - balanced_accuracy_score``) and
    ``Balanced Error Cost`` (the prior-weighted cost matrix). In
    expectation they coincide, but only the cost-matrix form admits the
    Analytical / Calibrate-then-Analytical strategies. We drop the
    MetricObjective rows and relabel the cost-matrix rows to
    ``Balanced Error`` so downstream tables and plots show the full
    twelve-strategy column.
    """
    if "optimization_objective_name" not in df.columns:
        return df
    df = df[df["optimization_objective_name"] != "Balanced Error"].copy()
    df["optimization_objective_name"] = df[
        "optimization_objective_name"
    ].replace({"Balanced Error Cost": "Balanced Error"})
    return df


def _sanitize_filename(name: str) -> str:
    """Lowercase and replace non-alphanumeric chars with underscores."""
    return name.lower().replace(" ", "_")


def _find_latest_parquet(
    pattern: str = "results_binary_*.parquet",
) -> Path:
    # Search both top-level (legacy) and inside run_*/ subdirectories.
    files = [
        f
        for f in (
            *RESULTS_DIR.glob(pattern),
            *RESULTS_DIR.glob(f"run_*/{pattern}"),
        )
        if "profile" not in f.name
    ]
    if not files:
        raise FileNotFoundError(
            f"No parquet files matching {pattern} in {RESULTS_DIR}"
        )
    return max(files, key=lambda p: p.stat().st_mtime)


def write_strategy_tables(
    parquet_path: Path | None = None,
    tables_dir: Path | None = None,
    scope: str = "binary",
    datasets: tuple[str, ...] | None = None,
) -> list[Path]:
    """Generate per-dataset strategy result tables, split by objective type.

    Two files per dataset are produced: ``results_<dataset>_cost.tex`` (cost
    matrices; all strategies) and ``results_<dataset>_max.tex`` (maximization
    metrics; only strategies that apply to non-convex objectives). They are
    written to ``tex/supplement/tables/`` and inputted from the standalone
    companion supplement (``tex/supplement/supplement.tex``), not the main
    dissertation.

    ``scope`` selects the objective set and the default parquet pattern;
    supported values are ``"binary"`` and ``"multiclass"``. ``datasets``
    overrides the config-based dataset filter (mainly for tests with
    synthetic dataset names).
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    cfg = SCOPE_CONFIG[scope]
    max_metrics = cfg["max_metrics"]
    cost_objectives = cfg["cost_objectives"]
    if datasets is None:
        datasets = cfg["datasets"]

    if parquet_path is None:
        parquet_path = _find_latest_parquet(cfg["parquet_pattern"])
    if tables_dir is None:
        tables_dir = SUPPLEMENT_TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = _apply_legacy_strategy_renames(df)
    df = _normalize_balanced_error_objective(df)
    df = df[df["dataset_name"].isin(datasets)]

    specs = (
        (
            "cost",
            df[df["optimization_objective_name"].isin(cost_objectives)],
            "Cost-matrix results for {ds}.",
        ),
        (
            "max",
            df[df["optimization_objective_name"].isin(max_metrics)],
            "Maximization-metric results for {ds}.",
        ),
    )

    written: list[Path] = []
    for variant, df_variant, caption_tmpl in specs:
        if df_variant.empty:
            continue
        tables = build_latex_tables_per_dataset(
            df_variant,
            max_metrics,
            shorten_model_names=SHORTEN_MODEL_NAMES,
        )
        for ds_label, latex_code in tables.items():
            ds_name = ds_label.replace("\\_", "_")
            filename = f"results_{_sanitize_filename(ds_name)}_{variant}.tex"
            label = f"tab:results-{_sanitize_filename(ds_name)}-{variant}"
            out_path = tables_dir / filename

            content = (
                "\\begin{table}[H]\n"
                "    \\centering\n"
                "    \\tiny\n"
                f"    \\caption{{{caption_tmpl.format(ds=ds_label)}}}%\n"
                f"    \\label{{{label}}}\n"
                "    \\adjustbox{max width=\\linewidth}{%\n"
                f"{latex_code}"
                "    }\n"
                "\\end{table}\n"
            )

            out_path.write_text(content, encoding="utf-8")
            written.append(out_path)
            print(f"  Wrote {out_path}")

    return written


def write_dataset_stats(tables_dir: Path | None = None) -> list[Path]:
    """Generate the binary and multiclass dataset summary tables."""
    if tables_dir is None:
        tables_dir = TABLES_DIR
    specs = (
        (
            list(BINARY_DATASETS),
            "dataset-summary-binary",
            "Summary of binary classification datasets used in the"
            " experiments.",
            "dataset_summary_binary.tex",
            False,
        ),
        (
            list(MULTICLASS_DATASETS),
            "dataset-summary-multiclass",
            "Summary of multiclass classification datasets used in the"
            " experiments.",
            "dataset_summary_multiclass.tex",
            True,
        ),
    )
    written: list[Path] = []
    for datasets, label, caption, filename, include_class_count in specs:
        latex = build_dataset_stats_latex(
            datasets=datasets,
            label=label,
            caption=caption,
            include_class_count=include_class_count,
        )
        out_path = tables_dir / filename
        out_path.write_text(latex, encoding="utf-8")
        print(f"  Wrote {out_path}")
        written.append(out_path)
    return written


def build_rotational_bias_latex(
    df: pd.DataFrame,
    *,
    z: float = 1.97,
    decimals: int = 3,
) -> str:
    """Build the rotational-bias metric x strategy LaTeX table string."""
    df_mean = df.pivot_table(
        index="metric",
        columns="strategy",
        values="score",
        aggfunc="mean",
    )
    df_std = df.pivot_table(
        index="metric",
        columns="strategy",
        values="score",
        aggfunc="std",
    )
    df_count = df.pivot_table(
        index="metric",
        columns="strategy",
        values="score",
        aggfunc="count",
    )
    strategies = list(ROTATIONAL_BIAS_STRATEGIES)
    df_mean = df_mean[strategies]
    df_std = df_std.reindex(columns=df_mean.columns)
    df_count = df_count.reindex(columns=df_mean.columns)

    # Transposed layout: strategies as rows, the two metrics as columns.
    # Five strategy columns overflow the text margin; with metrics as the
    # (two) columns the table fits and the long strategy names get the
    # full width of the row-label column.
    metrics = list(df_mean.index)
    cells_by_metric: dict[str, list[str]] = {}
    for metric_name in metrics:
        mean_row = df_mean.loc[metric_name].copy()
        std_row = df_std.loc[metric_name]
        count_row = df_count.loc[metric_name]
        # format_latex_row expects .name to be a tuple with objective at idx 2.
        # Called per metric over all strategies, so it bolds the best
        # strategy within each metric, which lands in a column here.
        mean_row.name = ("_", "_", metric_name)
        cells_by_metric[metric_name] = format_latex_row(
            mean_row,
            std_row,
            count_row,
            df_mean.columns,
            ROTATIONAL_BIAS_MAXIMIZATION,
            z=z,
            decimals=decimals,
        )

    rows_tex = []
    for s_idx, strat in enumerate(strategies):
        cells = [cells_by_metric[m][s_idx] for m in metrics]
        rows_tex.append(f"    {strat} & " + " & ".join(cells) + r" \\")

    header = " & ".join(metrics)
    tabular = (
        "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        "    \\toprule\n"
        f"    & {header} \\\\\n"
        "    \\midrule\n" + "\n".join(rows_tex) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}"
    )

    return (
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\caption{Strategy performance on the rotational bias scenario "
        "($K{=}3$, correlation $0.8$, imbalanced heteroscedastic).}%\n"
        "    \\label{tab:results-rotational-bias}\n"
        f"    {tabular}\n"
        "\\end{table}\n"
    )


def write_rotational_bias_table(
    *,
    df: pd.DataFrame | None = None,
    parquet_path: Path | None = None,
    tables_dir: Path | None = None,
) -> Path:
    """Generate the rotational-bias metric x strategy table.

    If *df* is supplied it is used directly; otherwise the latest
    ``results_rotational_bias_*.parquet`` in ``RESULTS_DIR`` is read.
    """
    if df is None:
        if parquet_path is None:
            files = sorted(
                RESULTS_DIR.glob("results_rotational_bias_*.parquet")
            )
            if not files:
                raise FileNotFoundError(
                    "No results_rotational_bias_*.parquet files in "
                    f"{RESULTS_DIR}"
                )
            parquet_path = files[-1]
        print(f"Reading: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    if tables_dir is None:
        tables_dir = TABLES_DIR

    if "strategy" in df.columns:
        df = df.assign(
            strategy=df["strategy"].replace(LEGACY_ROTATIONAL_BIAS_RENAMES)
        )

    table_tex = build_rotational_bias_latex(df)
    out_path = tables_dir / "results_rotational_bias.tex"
    out_path.write_text(table_tex, encoding="utf-8")
    print(f"  Wrote {out_path}")
    return out_path


def _plain_cells(
    mean_row: pd.Series,
    std_row: pd.Series,
    count_row: pd.Series,
    columns: pd.Index,
    *,
    z: float,
    decimals: int,
) -> list[str]:
    """Format ``mean ± z·SE`` cells with no bolding.

    Mirrors :func:`format_latex_row`'s number formatting and SE
    fallback (``z·s`` when ``n <= 1``, else ``z·s/sqrt(n)``) but never
    bolds and renders absent cells as an em-dash. Used for diagnostics
    with no inherent better/worse direction.
    """
    means = mean_row[columns]
    stds = std_row[columns]
    counts = count_row[columns]
    fmt = f"{{:.{decimals}f}}"
    cells: list[str] = []
    for col in columns:
        m, s, n = means[col], stds[col], counts[col]
        if pd.isna(m):
            cells.append("—")
            continue
        half = z * s if (pd.isna(n) or n <= 1) else z * s / np.sqrt(n)
        cells.append(f"{fmt.format(m)} ± {fmt.format(half)}")
    return cells


def _build_model_diagnostic_latex(
    df: pd.DataFrame,
    scope: str,
    *,
    value_col: str,
    direction: str,
    caption: str,
    label: str,
    z: float = 1.0,
    decimals: int = 3,
    transpose: bool = False,
) -> str:
    """Build a models x datasets model-level diagnostic table.

    *value_col* is a model-level column duplicated across every
    strategy/objective row, so it is deduped to one value per
    ``(dataset, model, seed)`` before aggregating to mean
    :math:`\\pm` ``z`` SE over the seeds (``z = 1`` is 1 SE, matching
    the other results tables), following the snippet at the bottom of
    ``experiments/benchmark_validation_reference.md``.

    Scaled to the text width with ``adjustbox`` (so the page does not
    flip sideways). By default rows are models in the study config's
    ``MODELS`` order and columns are datasets in its size-tiered order.
    With ``transpose=True`` the axes swap (rows are datasets, columns
    are models): there are fewer models than datasets, so the table is
    narrower and ``adjustbox`` shrinks the text far less. Cells where
    the model was skipped by a documented size/feature/class gate are
    absent from the parquet and render as an em-dash.

    *direction* controls per-dataset bolding, the same overlap rule the
    per-dataset appendix tables use: ``"max"`` bolds the best (largest)
    model and every model within ``z`` SE of it, ``"min"`` does the
    same for the smallest, and ``"none"`` disables bolding (for a
    diagnostic with no inherent better/worse direction). Bolding is
    computed per dataset, so it is preserved under either orientation.
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    if direction not in ("max", "min", "none"):
        raise ValueError(
            f"Unknown direction {direction!r}; expected max, min, or none"
        )
    cfg = SCOPE_CONFIG[scope]
    present = set(df["dataset_name"].unique())
    datasets = [d for d in cfg["datasets"] if d in present]

    sub = (
        df[["dataset_name", "model_name", "random_seed", value_col]]
        .dropna()
        .drop_duplicates(["dataset_name", "model_name", "random_seed"])
    )

    pivot_kwargs = dict(
        index="dataset_name",
        columns="model_name",
        values=value_col,
    )
    df_mean = sub.pivot_table(aggfunc="mean", **pivot_kwargs)
    df_std = sub.pivot_table(aggfunc="std", **pivot_kwargs)
    df_count = sub.pivot_table(aggfunc="count", **pivot_kwargs)

    model_order = [m for m in MODEL_ORDER[scope] if m in df_mean.columns]
    df_mean = df_mean.reindex(index=datasets, columns=model_order)
    df_std = df_std.reindex(index=datasets, columns=model_order)
    df_count = df_count.reindex(index=datasets, columns=model_order)

    # Format cells per dataset so bolding marks the per-dataset best
    # (format_latex_row bolds the best in the values it is given).
    # cells_by_dataset[ds] is aligned with model_order; this holds
    # under either orientation rendered below, preserving the bolding.
    cells_by_dataset: dict[str, list[str]] = {}
    for ds in datasets:
        mean_row = df_mean.loc[ds].copy()
        if direction == "none":
            cells_by_dataset[ds] = _plain_cells(
                mean_row,
                df_std.loc[ds],
                df_count.loc[ds],
                df_mean.columns,
                z=z,
                decimals=decimals,
            )
            continue
        # format_latex_row keys "higher is better" off name[2]; the
        # sentinel objective is listed in maximization_metrics only
        # for "max", so "min" bolds the smallest value instead.
        mean_row.name = ("_", "_", "__diag__")
        max_metrics = ("__diag__",) if direction == "max" else ()
        cells_by_dataset[ds] = format_latex_row(
            mean_row,
            df_std.loc[ds],
            df_count.loc[ds],
            df_mean.columns,
            max_metrics,
            z=z,
            decimals=decimals,
        )

    rows_tex: list[str] = []
    if transpose:
        # Datasets as rows, models as columns. cells_by_dataset[ds] is
        # already in model_order, so a row is just that list.
        for ds in datasets:
            ds_label = ds.replace("_", "\\_")
            rows_tex.append(
                f"    {ds_label} & "
                + " & ".join(cells_by_dataset[ds])
                + r" \\"
            )
        header = " & ".join(MODEL_DISPLAY_NAMES.get(m, m) for m in model_order)
        tabular = (
            "\\begin{tabular}{l" + "c" * len(model_order) + "}\n"
            "    \\toprule\n"
            f"    Dataset & {header} \\\\\n"
            "    \\midrule\n" + "\n".join(rows_tex) + "\n"
            "    \\bottomrule\n"
            "\\end{tabular}"
        )
    else:
        for m_idx, model in enumerate(model_order):
            model_label = MODEL_DISPLAY_NAMES.get(model, model)
            row_cells = [cells_by_dataset[ds][m_idx] for ds in datasets]
            rows_tex.append(
                f"    {model_label} & " + " & ".join(row_cells) + r" \\"
            )
        header = " & ".join(ds.replace("_", "\\_") for ds in datasets)
        tabular = (
            "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"
            "    \\toprule\n"
            f"    Model & {header} \\\\\n"
            "    \\midrule\n" + "\n".join(rows_tex) + "\n"
            "    \\bottomrule\n"
            "\\end{tabular}"
        )

    return (
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\footnotesize\n"
        f"    \\caption{{{caption}}}%\n"
        f"    \\label{{{label}}}\n"
        "    \\adjustbox{max width=\\linewidth}{%\n"
        f"    {tabular}\n"
        "    }\n"
        "\\end{table}\n"
    )


def build_model_auc_latex(
    df: pd.DataFrame,
    scope: str,
    *,
    z: float = 1.0,
    decimals: int = 3,
) -> str:
    """Build the datasets x models base-estimator ROC AUC table.

    Reports the held-out test ROC AUC of the base estimator trained on
    the full 80\\% training pool (``model_roc_auc_full``). Higher is
    better, so the per-dataset bolding uses direction ``"max"``.
    Rendered with ``transpose=True`` (datasets as rows, models as
    columns) so the wide dataset suites do not force ``adjustbox`` to
    shrink the text; see :func:`_build_model_diagnostic_latex` for the
    dedup and layout. Inputted by the binary/multiclass experiment
    chapters of the main dissertation.
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    scope_word = "binary" if scope == "binary" else "multiclass"
    auc_desc = (
        "positive class"
        if scope == "binary"
        else "one-vs-rest, class-frequency-weighted"
    )
    caption = (
        "Held-out test ROC AUC of each base estimator trained on the "
        f"full 80\\% training pool ({auc_desc}) for the {scope_word} "
        "datasets, reported as mean $\\pm$ 1 SE over the ten seeds. "
        "Models skipped by a documented sample-size, feature-count, or "
        "class-count gate have no entry. Bold marks every model within "
        "1 SE of the per-dataset best."
    )
    return _build_model_diagnostic_latex(
        df,
        scope,
        value_col="model_roc_auc_full",
        direction="max",
        caption=caption,
        label=f"tab:model-auc-{scope}",
        z=z,
        decimals=decimals,
        transpose=True,
    )


def build_model_ece_latex(
    df: pd.DataFrame,
    scope: str,
    *,
    z: float = 1.0,
    decimals: int = 3,
) -> str:
    """Build the models x datasets base-estimator ECE table.

    Reports the held-out test Expected Calibration Error of the base
    estimator trained on the full 80\\% training pool (``model_ece_full``;
    top-label ECE, 15 bins). Lower is better, so the per-dataset
    bolding uses direction ``"min"``. See
    :func:`_build_model_diagnostic_latex` for the dedup and layout.
    Inputted by the companion supplement.
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    scope_word = "binary" if scope == "binary" else "multiclass"
    caption = (
        "Held-out test Expected Calibration Error (top-label ECE, 15 "
        "bins) of each base estimator trained on the full 80\\% "
        f"training pool for the {scope_word} datasets, reported as mean "
        "$\\pm$ 1 SE over the ten seeds. Lower is better. Models "
        "skipped by a documented sample-size, feature-count, or "
        "class-count gate have no entry. Bold marks every model within "
        "1 SE of the per-dataset lowest."
    )
    return _build_model_diagnostic_latex(
        df,
        scope,
        value_col="model_ece_full",
        direction="min",
        caption=caption,
        label=f"tab:model-ece-{scope}",
        z=z,
        decimals=decimals,
    )


def build_model_vertex_latex(
    df: pd.DataFrame,
    scope: str,
    *,
    z: float = 1.0,
    decimals: int = 3,
) -> str:
    """Build the models x datasets base-estimator vertex-distance table.

    Reports the mean Euclidean distance from each held-out test
    predicted probability vector to its nearest simplex vertex (the
    argmax vertex) for the base estimator trained on the full 80\\%
    training pool (``model_mean_vertex_distance_full``). This is a
    confidence-concentration diagnostic with no inherent better/worse
    direction, so no cell is bolded (direction ``"none"``). See
    :func:`_build_model_diagnostic_latex` for the dedup and layout.
    Inputted by the companion supplement.
    """
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    scope_word = "binary" if scope == "binary" else "multiclass"
    caption = (
        "Mean Euclidean distance from each held-out test predicted "
        "probability vector to its nearest simplex vertex (the argmax "
        "vertex) for each base estimator trained on the full 80\\% "
        f"training pool, for the {scope_word} datasets, reported as "
        "mean $\\pm$ 1 SE over the ten seeds. This diagnostic has no "
        "better/worse direction, so no cell is bolded. Models skipped by "
        "a documented sample-size, feature-count, or class-count gate "
        "have no entry."
    )
    return _build_model_diagnostic_latex(
        df,
        scope,
        value_col="model_mean_vertex_distance_full",
        direction="none",
        caption=caption,
        label=f"tab:model-vertex-{scope}",
        z=z,
        decimals=decimals,
    )


# kind -> (builder, output filename template, default output directory).
# "auc" lands in the main dissertation's tables dir (inputted by the
# experiment chapters); "ece"/"vertex" land in the supplement's tables
# dir (inputted by the companion supplement).
_MODEL_DIAGNOSTICS: dict[str, dict] = {
    "auc": {
        "builder": build_model_auc_latex,
        "filename": "model_auc_{scope}.tex",
        "tables_dir": TABLES_DIR,
    },
    "ece": {
        "builder": build_model_ece_latex,
        "filename": "model_ece_{scope}.tex",
        "tables_dir": SUPPLEMENT_TABLES_DIR,
    },
    "vertex": {
        "builder": build_model_vertex_latex,
        "filename": "model_vertex_distance_{scope}.tex",
        "tables_dir": SUPPLEMENT_TABLES_DIR,
    },
}


def write_model_diagnostic_tables(
    kind: str,
    scope: str = "binary",
    parquet_path: Path | None = None,
    tables_dir: Path | None = None,
) -> Path:
    """Generate a model-level diagnostic table (*kind*) for *scope*.

    *kind* is ``"auc"`` (written to ``tex/tables/``, inputted by the
    experiment chapters), ``"ece"``, or ``"vertex"`` (written to
    ``tex/supplement/tables/``, inputted by the companion supplement).
    """
    if kind not in _MODEL_DIAGNOSTICS:
        raise ValueError(
            f"Unknown kind {kind!r}; expected one of "
            f"{sorted(_MODEL_DIAGNOSTICS)}"
        )
    if scope not in SCOPE_CONFIG:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {sorted(SCOPE_CONFIG)}"
        )
    spec = _MODEL_DIAGNOSTICS[kind]
    cfg = SCOPE_CONFIG[scope]
    if parquet_path is None:
        parquet_path = _find_latest_parquet(cfg["parquet_pattern"])
    if tables_dir is None:
        tables_dir = spec["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = _apply_legacy_strategy_renames(df)
    df = _normalize_balanced_error_objective(df)
    table_tex = spec["builder"](df, scope)
    out_path = tables_dir / spec["filename"].format(scope=scope)
    out_path.write_text(table_tex, encoding="utf-8")
    print(f"  Wrote {out_path}")
    return out_path


def write_model_auc_tables(
    scope: str = "binary",
    parquet_path: Path | None = None,
    tables_dir: Path | None = None,
) -> Path:
    """Generate the base-estimator ROC AUC table for *scope*.

    Thin wrapper over :func:`write_model_diagnostic_tables` kept for the
    callers and tests that import it by name. Written to
    ``tex/tables/model_auc_<scope>.tex`` and inputted by the
    binary/multiclass experiment chapters of the main dissertation.
    """
    return write_model_diagnostic_tables(
        "auc", scope=scope, parquet_path=parquet_path, tables_dir=tables_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment results.",
    )
    parser.add_argument(
        "--only",
        choices=[
            "strategy",
            "dataset-stats",
            "rotational-bias",
            "model-auc",
            "model-ece",
            "model-vertex",
        ],
        default=None,
        help="Generate only a specific table type (default: all).",
    )
    parser.add_argument(
        "--scope",
        choices=["binary", "multiclass", "all"],
        default="all",
        help=(
            "Which experimental campaign to regenerate strategy tables"
            " for (default: all). Applies only to --only strategy."
        ),
    )
    args = parser.parse_args()

    implicit_all = args.only is None

    if args.only is None or args.only == "strategy":
        print("Generating strategy result tables...")
        scopes = (
            ("binary", "multiclass") if args.scope == "all" else (args.scope,)
        )
        for scope in scopes:
            print(f"  [{scope}]")
            try:
                write_strategy_tables(scope=scope)
            except FileNotFoundError as exc:
                if not implicit_all:
                    raise
                print(f"  Skipped {scope}: {exc}")

    if args.only is None or args.only == "dataset-stats":
        print("Generating dataset summary table...")
        write_dataset_stats()

    if args.only is None or args.only == "rotational-bias":
        print("Generating rotational-bias table...")
        try:
            write_rotational_bias_table()
        except FileNotFoundError as exc:
            if not implicit_all:
                raise
            print(f"  Skipped: {exc}")

    diagnostic_runs = (
        ("model-auc", "auc", "base-model ROC AUC"),
        ("model-ece", "ece", "base-model ECE (supplement)"),
        ("model-vertex", "vertex", "base-model vertex-distance (supplement)"),
    )
    for only_name, kind, desc in diagnostic_runs:
        if args.only is not None and args.only != only_name:
            continue
        print(f"Generating {desc} tables...")
        for scope in ("binary", "multiclass"):
            print(f"  [{scope}]")
            try:
                write_model_diagnostic_tables(kind, scope=scope)
            except FileNotFoundError as exc:
                if not implicit_all:
                    raise
                print(f"  Skipped {scope}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
