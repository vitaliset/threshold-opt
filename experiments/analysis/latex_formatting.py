"""LaTeX table formatting for experiment results.

Formats pivot tables of strategy performance (mean ± CI) with bold
highlighting for strategies whose confidence intervals overlap with
the best strategy's CI (direction-aware).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def format_latex_row(
    mean_row: pd.Series,
    std_row: pd.Series,
    count_row: pd.Series,
    strategy_columns: pd.Index,
    maximization_metrics: tuple[str, ...],
    *,
    z: float = 1.0,
    decimals: int = 3,
) -> list[str]:
    """Format one row of a results pivot table for LaTeX output.

    The displayed ± value is ``z * s / sqrt(n)``. With the default
    ``z = 1.0`` this is the standard error of the mean; callers who want
    CI half-widths (e.g. the rotational-bias table) pass ``z = 1.97``.

    A strategy is **bolded** when its ``mean ± z·SE`` interval overlaps
    with *any* best strategy's interval (i.e. it is statistically
    indistinguishable from the best at the chosen ``z``).

    Parameters
    ----------
    mean_row : pd.Series
        Mean objective values indexed by strategy name.  The Series
        ``name`` must be a tuple whose **third element** is the
        objective name (dataset, model, objective).
    std_row : pd.Series
        Sample standard deviations, same index as *mean_row*.
    count_row : pd.Series
        Number of observations per strategy, same index as *mean_row*.
        The half-width is ``z * s / sqrt(n)``. Cells with ``n <= 1``
        fall back to ``z * s`` since the SE is not defined.
    strategy_columns : pd.Index
        Which columns of *mean_row* / *std_row* are strategies.
    maximization_metrics : tuple[str, ...]
        Objective names where higher is better.
    z : float
        Multiplier for the SE (default 1.0 → display ± 1 SE).
    decimals : int
        Number of decimal places in the formatted numbers.

    Returns
    -------
    list[str]
        One LaTeX-formatted cell per strategy column.
    """
    objective_name = mean_row.name[2]
    is_maximization = objective_name in maximization_metrics

    means = mean_row[strategy_columns]
    stds = std_row[strategy_columns]
    counts = count_row[strategy_columns]

    means_no_na = means.dropna()
    if means_no_na.empty:
        return ["\u2014"] * len(strategy_columns)

    best_val = means_no_na.max() if is_maximization else means_no_na.min()

    def _ci_half(s: float, n: float) -> float:
        if pd.isna(n) or n <= 1:
            return z * s
        return z * s / np.sqrt(n)

    # Collect CIs of all strategies tied for best (within float tolerance)
    is_best_mask = np.isclose(means, best_val)
    best_cis: list[tuple[float, float]] = []
    for col in means[is_best_mask].index:
        m, s, n = means[col], stds[col], counts[col]
        if not pd.isna(m):
            h = _ci_half(s, n)
            best_cis.append((m - h, m + h))

    fmt = f"{{:.{decimals}f}}"
    cells: list[str] = []
    for col in strategy_columns:
        m, s, n = means[col], stds[col], counts[col]
        if pd.isna(m):
            cells.append("\u2014")
            continue

        ci_half = _ci_half(s, n)
        cell_str = f"{fmt.format(m)} ± {fmt.format(ci_half)}"

        lo, hi = m - ci_half, m + ci_half
        overlaps = any(lo <= bhi and blo <= hi for blo, bhi in best_cis)

        if overlaps:
            cells.append(f"\\textbf{{{cell_str}}}")
        else:
            cells.append(cell_str)

    return cells


def build_latex_table(
    df: pd.DataFrame,
    maximization_metrics: tuple[str, ...],
    *,
    z: float = 1.97,
    decimals: int = 3,
    shorten_model_names: dict[str, str] | None = None,
    escape_underscores: bool = True,
) -> str:
    """Build a full LaTeX table from raw experiment results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results with columns: dataset_name, model_name,
        optimization_objective_name, strategy_name, objective_value.
    maximization_metrics : tuple[str, ...]
        Objective names where higher is better.
    z : float
        CI multiplier (default 1.97).
    decimals : int
        Decimal places in formatted numbers.
    shorten_model_names : dict[str, str] | None
        Optional mapping of model name substrings to replacements.
    escape_underscores : bool
        Whether to escape ``_`` in dataset names for LaTeX.

    Returns
    -------
    str
        LaTeX table code.
    """
    group_idx = ["dataset_name", "model_name", "optimization_objective_name"]

    df_mean = df.pivot_table(
        index=group_idx,
        columns="strategy_name",
        values="objective_value",
        aggfunc="mean",
    )
    df_std = df.pivot_table(
        index=group_idx,
        columns="strategy_name",
        values="objective_value",
        aggfunc="std",
    )
    df_count = df.pivot_table(
        index=group_idx,
        columns="strategy_name",
        values="objective_value",
        aggfunc="count",
    )
    # Align indices
    df_std = df_std.reindex(df_mean.index)
    df_count = df_count.reindex(df_mean.index)

    strategy_cols = df_std.columns

    df_latex = df_mean.apply(
        lambda row: format_latex_row(
            row,
            df_std.loc[row.name],
            df_count.loc[row.name],
            strategy_cols,
            maximization_metrics,
            z=z,
            decimals=decimals,
        ),
        axis=1,
        result_type="expand",
    )
    df_latex.columns = strategy_cols
    df_latex.index.names = ["Dataset", "Model", "Obj."]

    # Shorten index values
    if shorten_model_names is None:
        shorten_model_names = {}
    new_index = []
    for ds, model, obj in df_latex.index:
        for old, new in shorten_model_names.items():
            model = model.replace(old, new)
        if escape_underscores:
            ds = ds.replace("_", "\\_")
        new_index.append((ds, model, obj))
    df_latex.index = pd.MultiIndex.from_tuples(
        new_index,
        names=df_latex.index.names,
    )

    latex_code = df_latex.to_latex(
        escape=False,
        multirow=True,
        column_format="lll" + "c" * len(df_latex.columns),
    )
    latex_code = latex_code.replace("strategy_name", "Strategy")
    return latex_code


def build_latex_tables_per_dataset(
    df: pd.DataFrame,
    maximization_metrics: tuple[str, ...],
    *,
    z: float = 1.0,
    decimals: int = 3,
    shorten_model_names: dict[str, str] | None = None,
    escape_underscores: bool = True,
) -> dict[str, str]:
    """Build one LaTeX table per dataset from raw experiment results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results with columns: dataset_name, model_name,
        optimization_objective_name, strategy_name, objective_value.
    maximization_metrics : tuple[str, ...]
        Objective names where higher is better.
    z : float
        CI multiplier (default 1.97).
    decimals : int
        Decimal places in formatted numbers.
    shorten_model_names : dict[str, str] | None
        Optional mapping of model name substrings to replacements.
    escape_underscores : bool
        Whether to escape ``_`` in dataset names for LaTeX.

    Returns
    -------
    dict[str, str]
        Mapping of dataset name to LaTeX table code.
    """
    if shorten_model_names is None:
        shorten_model_names = {}

    group_idx = ["dataset_name", "model_name", "optimization_objective_name"]

    df_mean = df.pivot_table(
        index=group_idx,
        columns="strategy_name",
        values="objective_value",
        aggfunc="mean",
    )
    df_std = df.pivot_table(
        index=group_idx,
        columns="strategy_name",
        values="objective_value",
        aggfunc="std",
    )
    df_count = df.pivot_table(
        index=group_idx,
        columns="strategy_name",
        values="objective_value",
        aggfunc="count",
    )
    df_std = df_std.reindex(df_mean.index)
    df_count = df_count.reindex(df_mean.index)

    strategy_cols = df_std.columns

    df_latex = df_mean.apply(
        lambda row: format_latex_row(
            row,
            df_std.loc[row.name],
            df_count.loc[row.name],
            strategy_cols,
            maximization_metrics,
            z=z,
            decimals=decimals,
        ),
        axis=1,
        result_type="expand",
    )
    df_latex.columns = strategy_cols

    tables: dict[str, str] = {}
    for dataset_name in df["dataset_name"].unique():
        # Filter to this dataset and drop the dataset level
        ds_slice = df_latex.loc[dataset_name]
        ds_slice.index.names = ["Model", "Obj."]

        # Shorten model names
        new_index = []
        for model, obj in ds_slice.index:
            for old, new in shorten_model_names.items():
                model = model.replace(old, new)
            new_index.append((model, obj))
        ds_slice.index = pd.MultiIndex.from_tuples(
            new_index,
            names=ds_slice.index.names,
        )

        latex_code = ds_slice.to_latex(
            escape=False,
            multirow=True,
            column_format="ll" + "c" * len(ds_slice.columns),
        )
        latex_code = latex_code.replace("strategy_name", "Strategy")

        label = dataset_name
        if escape_underscores:
            label = label.replace("_", "\\_")
        tables[label] = latex_code

    return tables
