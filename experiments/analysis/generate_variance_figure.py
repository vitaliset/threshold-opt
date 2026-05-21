"""Generate the variance-experiment figure from the latest runner parquet.

Emits ``tex/figures/scaling_variance.tex`` as a two-panel pgfplots
figure showing the variance-scaling relative excess risk in both
log-log and linear scales for the Asymmetric 10x cost matrix. Each
per-K seed mean is drawn with a normal-approx 95% confidence band
(``mean +/- 1.96 * std / sqrt(n_seeds)``), matching the repo's
standard SE-based interval idiom.

Usage:
    uv run python -m experiments.analysis.generate_variance_figure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIG_DIR = REPO_ROOT / "tex" / "figures"

PARQUET_GLOB = "run_results_variance_*/results_variance_*.parquet"
OUTPUT_PATH = FIG_DIR / "scaling_variance.tex"

PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
)

Z_95 = 1.96


def _find_parquet() -> Path:
    files = sorted(RESULTS_DIR.glob(PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No KL-sensitivity parquet matching {PARQUET_GLOB} in "
            f"{RESULTS_DIR}"
        )
    return files[-1]


def _color_for(idx: int) -> str:
    r, g, b = PALETTE_RGB[idx % len(PALETTE_RGB)]
    return f"{{rgb,255:red,{r}; green,{g}; blue,{b}}}"


def _aggregate(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per-K seed mean and SE of the relative regret.

    Drops rows with ``c_opt <= 1/n_mc`` (the MC floor; relative-regret
    ratio is unreliable there).
    """
    out: dict[int, pd.DataFrame] = {}
    Ks = sorted(int(k) for k in df["K"].unique())
    for K in Ks:
        sub = df[df["K"] == K]
        if len(sub) == 0:
            continue
        n_mc = int(sub["n_mc"].iloc[0])
        sub = sub[sub["c_opt"] > (1.0 / n_mc)].copy()
        sub["rel"] = sub["regret"] / sub["c_opt"]
        agg = (
            sub.groupby("alpha")
            .agg(
                mean=("rel", "mean"),
                std=("rel", "std"),
                n=("rel", "count"),
            )
            .reset_index()
            .sort_values("alpha")
        )
        agg["se"] = agg["std"].fillna(0.0) / np.sqrt(agg["n"].astype(float))
        agg["lo"] = agg["mean"] - Z_95 * agg["se"]
        agg["hi"] = agg["mean"] + Z_95 * agg["se"]
        out[K] = agg
    return out


def _coords(xs: np.ndarray, ys: np.ndarray) -> str:
    return "\n".join(f"        ({x:.6g}, {y:.6g})" for x, y in zip(xs, ys))


def _curve_blocks(
    agg_by_K: dict[int, pd.DataFrame],
    *,
    log_scale: bool,
    tag: str,
    show_legend: bool,
) -> tuple[str, str, list[str]]:
    """Render per-K addplot blocks (CI band + line + markers)."""
    plots: list[str] = []
    legend: list[str] = []
    all_x: list[float] = []
    all_y_pos: list[float] = []
    all_y_full: list[float] = []
    Ks = sorted(agg_by_K.keys())
    pos_floor = 1e-6
    for idx, K in enumerate(Ks):
        agg = agg_by_K[K]
        x_raw = agg["alpha"].to_numpy(dtype=float)
        m_raw = agg["mean"].to_numpy(dtype=float)
        lo_raw = agg["lo"].to_numpy(dtype=float)
        hi_raw = agg["hi"].to_numpy(dtype=float)
        if log_scale:
            mask = (x_raw > 0) & (m_raw > 0)
        else:
            mask = x_raw >= 0
        x = x_raw[mask]
        m = m_raw[mask]
        lo = lo_raw[mask]
        hi = hi_raw[mask]
        if log_scale:
            lo = np.maximum(lo, pos_floor)
            hi = np.maximum(hi, pos_floor)
        all_x.extend(x.tolist())
        all_y_pos.extend(m[m > 0].tolist())
        all_y_full.extend(m.tolist())
        all_y_full.extend(hi.tolist())
        all_y_full.extend(lo.tolist())
        color = _color_for(idx)
        lo_name = f"lo_{tag}_K{K}"
        hi_name = f"hi_{tag}_K{K}"
        coords_pts = _coords(x, m)
        coords_lo = _coords(x, lo)
        coords_hi = _coords(x, hi)
        plots.append(
            f"\\addplot[name path={lo_name}, draw=none, forget plot]"
            f" coordinates {{\n{coords_lo}\n    }};\n"
            f"\\addplot[name path={hi_name}, draw=none, forget plot]"
            f" coordinates {{\n{coords_hi}\n    }};\n"
            f"\\addplot[color={color}, opacity=0.15, forget plot]"
            f" fill between [of={lo_name} and {hi_name}];\n"
            f"\\addplot[only marks, mark=*, mark size=1.2pt,"
            f" color={color}] coordinates {{\n{coords_pts}\n    }};\n"
            f"\\addplot[color={color}, line width=0.55pt, forget plot]"
            f" coordinates {{\n{coords_pts}\n    }};\n"
        )
        legend.append(f"$K={int(K)}$")
    pos_x = [v for v in all_x if v > 0]
    xmin = (min(pos_x) if pos_x else 1e-3) * 0.7
    xmax = (max(pos_x) if pos_x else 1.0) * 1.4
    if log_scale:
        y_pos = all_y_pos if all_y_pos else [1e-4]
        ymin = min(y_pos) * 0.3
        ymax = max(all_y_full) * 1.5
    else:
        ymin = 0.0
        ymax = max(all_y_full) * 1.05 if all_y_full else 1.0
    limits = (
        f"xmin={xmin:.6g}, xmax={xmax:.6g}, ymin={ymin:.6g}, ymax={ymax:.6g}"
    )
    if not show_legend:
        legend = []
    return "".join(plots), limits, legend


def _panel_block(
    agg_by_K: dict[int, pd.DataFrame],
    *,
    title: str,
    log_scale: bool,
    tag: str,
    show_legend: bool,
    show_ylabel: bool,
) -> str:
    plots, limits, legend = _curve_blocks(
        agg_by_K, log_scale=log_scale, tag=tag, show_legend=show_legend
    )
    legend_csv = ", ".join(legend)
    legend_block = (
        (
            "    legend style={font=\\scriptsize, fill opacity=0.85,"
            " draw=none, at={(0.02,0.98)}, anchor=north west},\n"
            "    legend cell align=left,\n"
        )
        if show_legend
        else ""
    )
    legend_call = f"\\legend{{{legend_csv}}}\n" if show_legend else ""
    ymode = "log" if log_scale else "normal"
    ylabel_line = (
        "    ylabel={$(R_\\lambda(h^{\\ast}_1)"
        " - R_\\lambda(h^{\\ast}_\\lambda))"
        " / R_\\lambda(h^{\\ast}_\\lambda)$},\n"
        if show_ylabel
        else ""
    )
    return (
        "\\nextgroupplot[\n"
        f"    title={{{title}}},\n"
        "    xlabel={Variance multiplier $\\lambda$},\n"
        f"{ylabel_line}"
        f"    ymode={ymode},\n"
        f"{legend_block}"
        f"    {limits},\n"
        "]\n"
        f"{plots}"
        f"{legend_call}"
    )


def build() -> Path:
    parquet = _find_parquet()
    print(f"Reading: {parquet}")
    df = pd.read_parquet(parquet)
    required = {"K", "kl", "regret", "c_opt", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Parquet {parquet} is missing columns: {missing}")
    agg_by_K = _aggregate(df)
    cost_name = str(np.atleast_1d(df["cost_name"].unique())[0])

    log_panel = _panel_block(
        agg_by_K,
        title=f"{cost_name} cost",
        log_scale=True,
        tag="log",
        show_legend=False,
        show_ylabel=True,
    )
    lin_panel = _panel_block(
        agg_by_K,
        title=f"{cost_name} cost",
        log_scale=False,
        tag="lin",
        show_legend=True,
        show_ylabel=False,
    )

    header = (
        "% Auto-generated by"
        " experiments.analysis.generate_variance_figure"
        " --- do not edit.\n"
        f"% Cost: {cost_name}. Uniform variance perturbation across"
        " classes, lambda >= 1.\n"
        "% Horizontal axis is the variance multiplier lambda.\n"
        "% Left log-log, right linear; bands are normal-approx 95%"
        " CIs (mean +/- 1.96 * SE).\n"
    )
    tex = (
        header
        + "\\noindent\\begin{tikzpicture}\n"
        + "\\begin{groupplot}[\n"
        + "    group style={group size=2 by 1, horizontal sep=1.6cm},\n"
        + "    width=0.46\\textwidth, height=6.0cm,\n"
        + "    grid=major, grid style={dotted, gray!40},\n"
        + "    xmode=log,\n"
        + "    tick label style={font=\\scriptsize},\n"
        + "    label style={font=\\footnotesize},\n"
        + "    title style={font=\\footnotesize},\n"
        + "]\n"
        + log_panel
        + lin_panel
        + "\\end{groupplot}\n"
        + "\\end{tikzpicture}%\n"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(tex, encoding="utf-8")
    print(f"  Wrote {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build()
