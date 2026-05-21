"""Generate strategy win/tie-count summary figures from experiment results.

For every (scope, family) combination, count how many (dataset, model,
objective) cells each strategy ``wins or ties'' --- using the same
``mean $\\pm$ 1 SE intervals overlap with the best'' rule the per-dataset
appendix tables use to bold cells. Output one TikZ heatmap per (scope,
family), with strategies on the y-axis and column groups for the overall
total, per base classifier, per dataset-size bucket, and per
imbalance-ratio bucket. Counts shown in each cell; cell shading is the
win rate (count divided by the cells in that strategy $\\times$ column
slice).

Generates four files under ``tex/figures/win_counts/``:

    win_counts_binary_cost.tex
    win_counts_binary_max.tex
    win_counts_multiclass_cost.tex
    win_counts_multiclass_max.tex

Usage:
    uv run python -m experiments.analysis.generate_win_counts
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analysis.generate_tables import (
    SCOPE_CONFIG,
    _apply_legacy_strategy_renames,
    _find_latest_parquet,
    _normalize_balanced_error_objective,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WIN_COUNTS_DIR = REPO_ROOT / "tex" / "figures" / "win_counts"

# Display labels for base classifiers used in heatmap column headers.
MODEL_LABELS: dict[str, str] = {
    "LGBMClassifier()": "LGBM",
    "XGBoost": "XGB",
    "CatBoost": "Cat",
    "RandomForest": "RF",
    "ExtraTrees": "ET",
    "LogisticRegression()": "LogReg",
    "GaussianNB": "NB",
    "TabM": "TabM",
    "TabPFN": "TabPFN",
    "TabICL": "TabICL",
}

# Display order for base classifiers (left to right in the heatmap).
MODEL_ORDER: tuple[str, ...] = (
    "LGBMClassifier()",
    "XGBoost",
    "CatBoost",
    "RandomForest",
    "ExtraTrees",
    "LogisticRegression()",
    "GaussianNB",
    "TabM",
    "TabPFN",
    "TabICL",
)

# Dataset size buckets, shared across scopes. Half-open intervals [lo, hi).
SIZE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("small", 0.0, 1000.0),
    ("medium", 1000.0, 10000.0),
    ("large", 10000.0, float("inf")),
)

# Imbalance-ratio buckets per scope. Cutoffs differ because the binary
# benchmark spans IR 1.0--600 while the multiclass benchmark is bimodal
# (most datasets at IR$\approx$1, a handful $\gg$10), so a single shared
# moderate band would be near-empty on the multiclass side.
IR_BUCKETS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "binary": (
        ("low", 0.0, 2.0),
        ("moderate", 2.0, 10.0),
        ("high", 10.0, float("inf")),
    ),
    "multiclass": (
        ("balanced", 0.0, 2.0),
        ("imbalanced", 2.0, float("inf")),
    ),
}


def compute_dataset_metadata(
    datasets: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Return ``{name: {"n": rows, "ir": majority/minority}}`` for *datasets*.

    Loads each dataset via ``data_loader.load_raw_dataset`` to compute the
    raw row count and the majority/minority class ratio. This is slow on
    a fresh cache; callers in tests can pass a pre-built dict instead.
    """
    from experiments.common import data_loader

    out: dict[str, dict[str, float]] = {}
    for name in datasets:
        X, y = data_loader.load_raw_dataset(name)
        cnt = pd.Series(y).value_counts()
        out[name] = {
            "n": float(X.shape[0]),
            "ir": float(cnt.max() / cnt.min()),
        }
    return out


def _bucket(
    value: float,
    bins: tuple[tuple[str, float, float], ...],
) -> str:
    """Return the bucket whose half-open ``[lo, hi)`` range covers *value*."""
    for label, lo, hi in bins:
        if lo <= value < hi:
            return label
    return bins[-1][0]


def _ci_half(s: float, n: float) -> float:
    """Half-width $s / \\sqrt{n}$, falling back to $s$ when $n \\le 1$."""
    if pd.isna(s):
        return 0.0
    if pd.isna(n) or n <= 1:
        return float(s)
    return float(s) / np.sqrt(float(n))


def _winners_one_cell(
    means: pd.Series,
    stds: pd.Series,
    counts: pd.Series,
    is_max: bool,
) -> set[str]:
    """Return strategies whose $\\pm 1$ SE interval overlaps the best one.

    Mirrors ``latex_formatting.format_latex_row`` so the appendix tables
    and the win-count figure agree on which cells are bolded.
    """
    means_no_na = means.dropna()
    if means_no_na.empty:
        return set()
    best_val = means_no_na.max() if is_max else means_no_na.min()

    # ``np.isclose`` against best_val on an array that may contain NaN: the
    # fillna gives NaN entries a value that cannot be close to best_val, so
    # they are excluded from the best mask without raising.
    sentinel = best_val + 1.0 if not np.isclose(best_val, 0.0) else 1.0
    is_best_mask = np.isclose(means.fillna(sentinel), best_val)
    best_cis: list[tuple[float, float]] = []
    for col in means[is_best_mask].index:
        m, s, n = means[col], stds[col], counts[col]
        if pd.isna(m):
            continue
        h = _ci_half(s, n)
        best_cis.append((float(m) - h, float(m) + h))

    winners: set[str] = set()
    for col in means.index:
        m, s, n = means[col], stds[col], counts[col]
        if pd.isna(m):
            continue
        h = _ci_half(s, n)
        lo, hi = float(m) - h, float(m) + h
        if any(lo <= bhi and blo <= hi for blo, bhi in best_cis):
            winners.add(str(col))
    return winners


def compute_win_table(
    df: pd.DataFrame,
    max_metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Build the long ``(cell, strategy, is_winner)`` table.

    One row per (cell, strategy) pair where the strategy has a non-NaN
    mean. ``is_winner`` is True iff the strategy is the cell's winner or
    tied with it under the CI overlap rule.
    """
    rows: list[dict[str, object]] = []
    for (ds, model, obj), grp in df.groupby(
        ["dataset_name", "model_name", "optimization_objective_name"],
        sort=False,
    ):
        is_max = obj in max_metrics
        means = grp.groupby("strategy_name")["objective_value"].mean()
        stds = grp.groupby("strategy_name")["objective_value"].std()
        counts = grp.groupby("strategy_name")["objective_value"].count()
        winners = _winners_one_cell(means, stds, counts, is_max)
        for strat in means.index:
            if pd.isna(means[strat]):
                continue
            rows.append(
                {
                    "dataset_name": ds,
                    "model_name": model,
                    "optimization_objective_name": obj,
                    "strategy_name": strat,
                    "is_winner": strat in winners,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_breakdown(
    win_table: pd.DataFrame,
    label_col: str,
    label_order: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot ``(strategy x label)`` win counts and denominators."""
    df = win_table[win_table[label_col].isin(label_order)]
    count = df.pivot_table(
        index="strategy_name",
        columns=label_col,
        values="is_winner",
        aggfunc="sum",
        fill_value=0,
    ).astype(int)
    denom = df.pivot_table(
        index="strategy_name",
        columns=label_col,
        values="is_winner",
        aggfunc="count",
        fill_value=0,
    ).astype(int)
    count = count.reindex(columns=label_order, fill_value=0)
    denom = denom.reindex(columns=label_order, fill_value=0)
    return count, denom


def aggregate_win_counts(
    win_table: pd.DataFrame,
    dataset_metadata: dict[str, dict[str, float]],
    scope: str,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build the four breakdowns shown in the figure.

    Returns ``{key: (count_df, denom_df)}`` for keys ``"total"``,
    ``"by_model"``, ``"by_size"``, ``"by_ir"``. Each dataframe is indexed
    by ``strategy_name``; columns follow the canonical display order.
    """
    table = win_table.copy()
    table["size_bucket"] = table["dataset_name"].map(
        lambda d: _bucket(dataset_metadata[d]["n"], SIZE_BUCKETS)
    )
    table["ir_bucket"] = table["dataset_name"].map(
        lambda d: _bucket(dataset_metadata[d]["ir"], IR_BUCKETS[scope])
    )

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    total_count = (
        table.groupby("strategy_name")["is_winner"]
        .sum()
        .to_frame(name="all")
        .astype(int)
    )
    total_denom = (
        table.groupby("strategy_name")["is_winner"]
        .count()
        .to_frame(name="all")
        .astype(int)
    )
    out["total"] = (total_count, total_denom)

    models_present = tuple(
        m for m in MODEL_ORDER if m in set(table["model_name"])
    )
    out["by_model"] = _aggregate_breakdown(table, "model_name", models_present)
    out["by_size"] = _aggregate_breakdown(
        table, "size_bucket", tuple(b[0] for b in SIZE_BUCKETS)
    )
    out["by_ir"] = _aggregate_breakdown(
        table, "ir_bucket", tuple(b[0] for b in IR_BUCKETS[scope])
    )
    return out


def _slice_arrays(
    slice_table: pd.DataFrame,
    strats: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(W, A)`` matrices for one slice of the win table.

    Returns ``W`` (``n_cells``\\,$\\times$\\,``n_strats``) holding $1$ where
    the strategy was a winner / tied and $0$ otherwise, and ``A`` of the
    same shape masking the cells where the strategy was applicable.
    Cells are uniquely identified by ``(dataset, model, objective)``.
    """
    cell_keys = [
        "dataset_name",
        "model_name",
        "optimization_objective_name",
    ]
    cells = slice_table[cell_keys].drop_duplicates().reset_index(drop=True)
    n_cells = len(cells)
    n_strats = len(strats)
    if n_cells == 0:
        return (
            np.zeros((0, n_strats)),
            np.zeros((0, n_strats), dtype=bool),
        )

    cell_to_i = {
        (row.dataset_name, row.model_name, row.optimization_objective_name): i
        for i, row in cells.iterrows()
    }
    strat_to_j = {s: j for j, s in enumerate(strats)}

    W = np.zeros((n_cells, n_strats))
    A = np.zeros((n_cells, n_strats), dtype=bool)
    for row in slice_table.itertuples(index=False):
        key = (
            row.dataset_name,
            row.model_name,
            row.optimization_objective_name,
        )
        i = cell_to_i[key]
        j = strat_to_j[row.strategy_name]
        W[i, j] = 1.0 if row.is_winner else 0.0
        A[i, j] = True
    return W, A


def _bootstrap_lo_hi(
    W: np.ndarray,
    A: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster bootstrap over rows of ``W`` / ``A``.

    Returns ``(lo, hi)`` per-strategy arrays of shape ``(n_strats,)``
    holding the requested percentiles of the bootstrap rate distribution.
    """
    n_cells, n_strats = W.shape
    if n_cells == 0:
        return np.zeros(n_strats), np.zeros(n_strats)

    rates = np.empty((n_boot, n_strats))
    for b in range(n_boot):
        idx = rng.integers(0, n_cells, size=n_cells)
        numer = W[idx].sum(axis=0)
        denom = A[idx].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates[b] = np.where(denom > 0, numer / denom, 0.0)

    lo = np.quantile(rates, quantiles[0], axis=0)
    hi = np.quantile(rates, quantiles[1], axis=0)
    return lo, hi


def bootstrap_breakdown_cis(
    win_table: pd.DataFrame,
    dataset_metadata: dict[str, dict[str, float]],
    scope: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Cluster-bootstrap 95\\% CIs for every (strategy, slice) win rate.

    Resamples ``(dataset, model, objective)`` cells with replacement
    *within each slice* (so the per-classifier CI for LGBM is conditional
    on LGBM cells, not on a marginal resample of the whole table). The
    return shape mirrors :func:`aggregate_win_counts`: a dict with keys
    ``"total"``, ``"by_model"``, ``"by_size"``, ``"by_ir"``, each mapping
    to ``(lo_df, hi_df)`` indexed by ``strategy_name`` with one column
    per slice value.
    """
    rng = np.random.default_rng(seed)

    table = win_table.copy()
    table["size_bucket"] = table["dataset_name"].map(
        lambda d: _bucket(dataset_metadata[d]["n"], SIZE_BUCKETS)
    )
    table["ir_bucket"] = table["dataset_name"].map(
        lambda d: _bucket(dataset_metadata[d]["ir"], IR_BUCKETS[scope])
    )

    strats = sorted(table["strategy_name"].unique())

    def _slice_ci(slice_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        W, A = _slice_arrays(slice_table, strats)
        return _bootstrap_lo_hi(W, A, rng, n_boot)

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    lo, hi = _slice_ci(table)
    out["total"] = (
        pd.DataFrame({"all": lo}, index=strats),
        pd.DataFrame({"all": hi}, index=strats),
    )

    for breakdown_key, col_name, label_order in (
        (
            "by_model",
            "model_name",
            tuple(m for m in MODEL_ORDER if m in set(table["model_name"])),
        ),
        ("by_size", "size_bucket", tuple(b[0] for b in SIZE_BUCKETS)),
        (
            "by_ir",
            "ir_bucket",
            tuple(b[0] for b in IR_BUCKETS[scope]),
        ),
    ):
        lo_dict: dict[str, np.ndarray] = {}
        hi_dict: dict[str, np.ndarray] = {}
        for label in label_order:
            sub = table[table[col_name] == label]
            lo_arr, hi_arr = _slice_ci(sub)
            lo_dict[label] = lo_arr
            hi_dict[label] = hi_arr
        lo_df = pd.DataFrame(lo_dict, index=strats)
        hi_df = pd.DataFrame(hi_dict, index=strats)
        lo_df.index.name = "strategy_name"
        hi_df.index.name = "strategy_name"
        out[breakdown_key] = (lo_df, hi_df)

    out["total"][0].index.name = "strategy_name"
    out["total"][1].index.name = "strategy_name"
    return out


def _strategy_order(
    breakdowns: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[str]:
    """Sort strategies by overall win rate, descending.

    Sorting by rate (rather than raw count) keeps the ranking comparable
    across strategies with different denominators -- e.g.\\ CV variants
    that skip the largest datasets have a smaller denominator than their
    non-CV counterparts.
    """
    total_count = breakdowns["total"][0]["all"]
    total_denom = breakdowns["total"][1]["all"].replace(0, 1)
    rate = total_count / total_denom
    return list(rate.sort_values(ascending=False).index)


def _green_pct(rate: float) -> int:
    """Map a rate in $[0, 1]$ to a TikZ ``green!K`` percentage in $[0, 80]$.

    Uses a quadratic curve (``80 * rate**2``) so adjacent high-rate cells
    stay visibly distinguishable. Most cells in this benchmark sit
    between $40\\%$ and $100\\%$; a concave curve (sqrt or log) maps that
    band to a small slice of the green range and saturates the top, while
    the quadratic curve stretches the $80$--$100$ end so cells $5$--$10$
    percentage points apart become legible at a glance.
    """
    rate = max(0.0, min(1.0, rate))
    return round(80 * rate**2)


def render_win_count_figure(
    *,
    scope: str,
    family: str,
    breakdowns: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    ci_breakdowns: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> str:
    """Render one (scope, family) heatmap as a self-contained .tex figure.

    When *ci_breakdowns* is supplied (output of
    :func:`bootstrap_breakdown_cis`), each cell adds a $\\pm$\\,half-width
    in percentage points after the rate, e.g. ``89\\%\\pm 3``.
    """
    strategies = _strategy_order(breakdowns)
    if not strategies:
        raise ValueError(
            f"No strategies found for scope={scope!r}, family={family!r}."
        )

    # Build the column list in left-to-right order. Group labels are kept
    # short so they fit even when a group is just two columns wide. The
    # ``half_s`` series carries the bootstrap CI half-width (in
    # percentage points) for each strategy in the slice; it is None when
    # CIs were not requested.
    columns: list[tuple[str, str, pd.Series, pd.Series, pd.Series | None]] = []

    def _half_series(breakdown_key: str, slice_key: str) -> pd.Series | None:
        if ci_breakdowns is None:
            return None
        lo_df, hi_df = ci_breakdowns[breakdown_key]
        return ((hi_df[slice_key] - lo_df[slice_key]) / 2.0) * 100.0

    cnt_df, den_df = breakdowns["total"]
    columns.append(
        (
            "Total",
            "all",
            cnt_df["all"],
            den_df["all"],
            _half_series("total", "all"),
        )
    )
    cnt_df, den_df = breakdowns["by_model"]
    for col in cnt_df.columns:
        columns.append(
            (
                "Base classifier",
                MODEL_LABELS.get(col, col),
                cnt_df[col],
                den_df[col],
                _half_series("by_model", col),
            )
        )
    cnt_df, den_df = breakdowns["by_size"]
    for col in cnt_df.columns:
        columns.append(
            (
                "Size",
                col,
                cnt_df[col],
                den_df[col],
                _half_series("by_size", col),
            )
        )
    cnt_df, den_df = breakdowns["by_ir"]
    for col in cnt_df.columns:
        columns.append(
            ("IR", col, cnt_df[col], den_df[col], _half_series("by_ir", col))
        )

    n_cols = len(columns)
    n_rows = len(strategies)

    # Group spans (1-indexed inclusive bounds).
    group_spans: list[tuple[str, int, int]] = []
    cur_group: str | None = None
    cur_start = 1
    for i, (group, _, _, _, _) in enumerate(columns, start=1):
        if group != cur_group:
            if cur_group is not None:
                group_spans.append((cur_group, cur_start, i - 1))
            cur_group = group
            cur_start = i
    assert cur_group is not None
    group_spans.append((cur_group, cur_start, n_cols))

    # Cells get a hair wider when CIs are shown so the ``\pm H`` tail
    # fits cleanly; adjustbox still scales the figure to textwidth.
    cell_w = 0.95 if ci_breakdowns is not None else 0.85
    cell_h = 0.55

    lines: list[str] = []
    lines.append(
        "% Auto-generated by experiments.analysis.generate_win_counts"
        " --- do not edit."
    )
    lines.append("\\begin{figure}[H]")
    lines.append("\\centering")
    lines.append("\\adjustbox{max width=\\textwidth}{%")
    lines.append("\\begin{tikzpicture}[")
    lines.append(f"    x={cell_w}cm, y={cell_h}cm,")
    lines.append(
        "    cell/.style={rectangle, draw=black!25, line width=0.2pt,"
    )
    lines.append(
        f"        minimum width={cell_w}cm, minimum height={cell_h}cm,"
    )
    lines.append("        anchor=center, font=\\scriptsize, inner sep=0pt},")
    lines.append("    rowlbl/.style={font=\\scriptsize, anchor=east},")
    lines.append(
        "    collbl/.style={font=\\scriptsize, anchor=west, rotate=60},"
    )
    lines.append(
        "    grouplbl/.style={font=\\scriptsize\\bfseries, anchor=south},"
    )
    lines.append("    sep/.style={draw=black!55, line width=0.5pt},")
    lines.append("]")

    # Heatmap cells. Each cell shows the strategy's win rate on its
    # slice as an integer percentage (0--100) suffixed with ``%''. When
    # bootstrap CIs are supplied, the cell appends ``\pm H`` in
    # percentage points (half the 95\\% CI width). Fill uses
    # ``rate**2`` so the densely-populated 80--100\\% top tier stays
    # visibly differentiated.
    for ri, strat in enumerate(strategies):
        y = n_rows - ri  # top row = ri=0 -> highest y
        for ci, (_grp, _lbl, cnt_s, den_s, half_s) in enumerate(
            columns, start=1
        ):
            count = int(cnt_s.get(strat, 0))
            denom = int(den_s.get(strat, 0))
            rate = (count / denom) if denom > 0 else 0.0
            pct_display = round(rate * 100)
            pct_fill = _green_pct(rate)
            if half_s is not None:
                # Drop the ``%'' sign to keep the cell compact -- with
                # the CI half-width appended, "89%\pm 2" doesn't fit.
                # Caption notes that values are percentages.
                half_pp = round(float(half_s.get(strat, 0.0)))
                cell_text = f"${pct_display}{{\\pm}}{half_pp}$"
            else:
                cell_text = f"{pct_display}\\%"
            lines.append(
                f"\\node[cell, fill=green!{pct_fill}] at ({ci}, {y}) "
                f"{{{cell_text}}};"
            )

    # Row labels (strategy names).
    for ri, strat in enumerate(strategies):
        y = n_rows - ri
        lines.append(f"\\node[rowlbl] at ({0.4}, {y}) {{{strat}}};")

    # Per-column labels with representative denominator.
    y_collbl = n_rows + 0.6
    for ci, (_grp, lbl, _cnt_s, den_s, _half_s) in enumerate(columns, start=1):
        denom_repr = int(den_s.max()) if len(den_s) else 0
        lines.append(
            f"\\node[collbl] at ({ci}, {y_collbl}) "
            f"{{{lbl} ($n{{=}}{denom_repr}$)}};"
        )

    # Group labels above column labels, with an underline bracket.
    y_group_line = n_rows + 4.4
    y_group_lbl = y_group_line + 0.05
    for group_lbl, start, end in group_spans:
        x_lo = start - 0.45
        x_hi = end + 0.45
        x_center = (start + end) / 2.0
        lines.append(
            f"\\draw[black!55, line width=0.5pt] "
            f"({x_lo}, {y_group_line}) -- ({x_hi}, {y_group_line});"
        )
        lines.append(
            f"\\node[grouplbl] at ({x_center}, {y_group_lbl}) {{{group_lbl}}};"
        )

    # Vertical separators between groups.
    for _grp, _start, end in group_spans[:-1]:
        x_sep = end + 0.5
        lines.append(
            f"\\draw[sep] ({x_sep}, {0.5}) -- ({x_sep}, {n_rows + 0.5});"
        )

    lines.append("\\end{tikzpicture}%")
    lines.append("}")

    family_label = (
        "cost-matrix objectives"
        if family == "cost"
        else "maximization metrics"
    )
    label = f"fig:win-counts-{scope}-{family}"
    if ci_breakdowns is not None:
        cell_caption = (
            " Each cell reads ``$r{\\pm}H$'': $r$ is the percentage of"
            " (dataset, model, objective) cells in the slice where the"
            " strategy won or tied the per-row best under the $\\pm 1$ SE"
            " overlap rule, and $H$ is the half-width of a 95\\%"
            " cluster-bootstrap confidence interval for $r$ (percentage"
            " points)."
        )
    else:
        cell_caption = (
            " Each cell shows the percentage of (dataset, model,"
            " objective) cells in the slice where the strategy was the"
            " winner or tied with it under the $\\pm 1$ SE overlap rule"
            " the appendix tables use."
        )
    lines.append(
        f"\\caption{{Per-strategy win/tie rates on {scope} {family_label}."
        f"{cell_caption}"
        " Shading is a quadratic ($\\mathrm{rate}^2$) scale of $r$."
        " Strategies are sorted by overall win rate; ``$n$'' under each"
        " column header is the maximum applicable cell count for that"
        " slice.}%"
    )
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{figure}")

    return "\n".join(lines) + "\n"


def write_win_count_figures(
    parquet_paths: dict[str, Path] | None = None,
    plots_dir: Path | None = None,
    dataset_metadata: dict[str, dict[str, float]] | None = None,
    *,
    bootstrap: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[Path]:
    """Generate all four (scope, family) win-count figures.

    Parameters
    ----------
    parquet_paths
        Optional ``{scope: parquet_path}`` mapping. Missing scopes use
        ``_find_latest_parquet(SCOPE_CONFIG[scope]["parquet_pattern"])``.
    plots_dir
        Output directory. Defaults to ``WIN_COUNTS_DIR``.
    dataset_metadata
        Optional pre-built metadata. If absent, computed from the union
        of dataset names across both scopes via ``compute_dataset_metadata``.
    bootstrap
        When True (default), compute cluster bootstrap 95\\% CIs for
        every (strategy, slice) win rate and append the half-width to
        each cell. Set to False to suppress (faster generation, sparser
        figure).
    n_boot
        Number of bootstrap replicates per slice when ``bootstrap=True``.
    seed
        Bootstrap RNG seed (so re-running ``make win-counts`` gives
        identical figures).
    """
    if plots_dir is None:
        plots_dir = WIN_COUNTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = dict(parquet_paths or {})

    scope_dfs: dict[str, pd.DataFrame] = {}
    for scope in ("binary", "multiclass"):
        cfg = SCOPE_CONFIG[scope]
        path = parquet_paths.get(scope)
        if path is None:
            path = _find_latest_parquet(cfg["parquet_pattern"])
        print(f"Reading: {path}")
        df = pd.read_parquet(path)
        df = _apply_legacy_strategy_renames(df)
        df = _normalize_balanced_error_objective(df)
        scope_dfs[scope] = df[df["dataset_name"].isin(cfg["datasets"])]

    if dataset_metadata is None:
        all_datasets = tuple(
            sorted(
                set(scope_dfs["binary"]["dataset_name"]).union(
                    scope_dfs["multiclass"]["dataset_name"]
                )
            )
        )
        dataset_metadata = compute_dataset_metadata(all_datasets)

    written: list[Path] = []
    for scope in ("binary", "multiclass"):
        cfg = SCOPE_CONFIG[scope]
        df = scope_dfs[scope]
        max_metrics = cfg["max_metrics"]
        cost_objs = cfg["cost_objectives"]

        for family, objs in (("cost", cost_objs), ("max", max_metrics)):
            sub = df[df["optimization_objective_name"].isin(objs)]
            if sub.empty:
                continue
            win_table = compute_win_table(sub, max_metrics)
            breakdowns = aggregate_win_counts(
                win_table, dataset_metadata, scope
            )
            ci_breakdowns: (
                dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None
            ) = None
            if bootstrap:
                print(
                    f"  Bootstrapping CIs for {scope}/{family} "
                    f"(n_boot={n_boot})..."
                )
                ci_breakdowns = bootstrap_breakdown_cis(
                    win_table,
                    dataset_metadata,
                    scope,
                    n_boot=n_boot,
                    seed=seed,
                )
            tex = render_win_count_figure(
                scope=scope,
                family=family,
                breakdowns=breakdowns,
                ci_breakdowns=ci_breakdowns,
            )
            fname = f"win_counts_{scope}_{family}.tex"
            out = plots_dir / fname
            out.write_text(tex, encoding="utf-8")
            written.append(out)
            print(f"  Wrote {out}")
    return written


def main() -> None:
    write_win_count_figures()
    print("Done.")


if __name__ == "__main__":
    main()
