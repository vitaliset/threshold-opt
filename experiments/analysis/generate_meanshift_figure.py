"""Per-K curves figure for the mean-shift robustness experiment.

Emits ``tex/figures/scaling_meanshift.tex`` as a 2x2 pgfplots panel
showing the random per-class mean-shift relative excess risk in both
log-log and linear scales for the two cost matrices. Each per-K
seed-mean is drawn with a normal-approx 95% confidence band
(``mean +/- 1.96 * std / sqrt(n_seeds)``), matching the repo's
standard SE-based interval idiom (see ``latex_formatting.py`` and
``generate_ci_plots.py``).

Layout (groupplots ``2 by 2``):

    | Symmetric, log-log  | Asymmetric, log-log  |
    | Symmetric, linear   | Asymmetric, linear   |

Usage:
    uv run python -m experiments.analysis.generate_meanshift_figure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIG_DIR = REPO_ROOT / "tex" / "figures"

PARQUET_GLOB = "run_results_meanshift*/results_meanshift*.parquet"
OUTPUT_PATH = FIG_DIR / "scaling_meanshift.tex"

# Per-K curves figure uses a sparser K set to keep the panels readable;
# the full set of K values present in the parquet is used by the
# Lemma-test setup (which fits a K-decay exponent over all of them).
DISPLAY_KS: tuple[int, ...] = (2, 3, 4, 5, 10, 20)

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


def _latest_by_cost() -> dict[str, pd.DataFrame]:
    files = sorted(RESULTS_DIR.glob(PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No mean-shift parquet matching {PARQUET_GLOB} in {RESULTS_DIR}"
        )
    out: dict[str, pd.DataFrame] = {}
    for path in files:
        if "fixedpair" in path.parent.name:
            continue
        df = pd.read_parquet(path)
        if "r" not in df.columns:
            continue
        out[str(df["cost_name"].iloc[0])] = df
    return out


def _color_for(idx: int) -> str:
    r, g, b = PALETTE_RGB[idx % len(PALETTE_RGB)]
    return f"{{rgb,255:red,{r}; green,{g}; blue,{b}}}"


def _aggregate(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per-K seed mean and SE of the relative excess risk.

    Drops rows where ``c_opt <= 1/n_mc`` (the MC floor; relative-regret
    ratio is unreliable there).
    """
    out: dict[int, pd.DataFrame] = {}
    available = {int(k) for k in df["K"].unique()}
    Ks = [K for K in DISPLAY_KS if K in available]
    for K in Ks:
        sub = df[df["K"] == K]
        if len(sub) == 0:
            continue
        n_mc = int(sub["n_mc"].iloc[0])
        sub = sub[sub["c_opt"] > (1.0 / n_mc)].copy()
        sub["rel"] = sub["regret"] / sub["c_opt"]
        agg = (
            sub.groupby("r")
            .agg(
                kl=("kl", "first"),
                mean=("rel", "mean"),
                std=("rel", "std"),
                n=("rel", "count"),
            )
            .reset_index()
            .sort_values("r")
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
    """Render per-K addplot blocks (CI band + line + markers).

    Under log-y the lower band edge is clipped to a small positive
    floor so fillbetween still produces a valid path; pgfplots clips
    visually to ``ymin``.
    """
    plots: list[str] = []
    legend: list[str] = []
    all_x: list[float] = []
    all_y_pos: list[float] = []
    all_y_full: list[float] = []
    Ks = sorted(agg_by_K.keys())
    # A small positive floor to keep fillbetween paths valid under log-y.
    pos_floor = 1e-6
    for idx, K in enumerate(Ks):
        agg = agg_by_K[K]
        x_raw = agg["r"].to_numpy(dtype=float)
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
    show_xlabel: bool,
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
    xlabel_line = "    xlabel={$\\xi/\\sigma$},\n" if show_xlabel else ""
    ylabel_line = (
        "    ylabel={$(R_\\xi(h^{\\ast}_0)"
        " - R_\\xi(h^{\\ast}_\\xi))"
        " / R_\\xi(h^{\\ast}_\\xi)$},\n"
        if show_ylabel
        else ""
    )
    return (
        "\\nextgroupplot[\n"
        f"    title={{{title}}},\n"
        f"{xlabel_line}"
        f"{ylabel_line}"
        f"    ymode={ymode},\n"
        f"{legend_block}"
        f"    {limits},\n"
        "]\n"
        f"{plots}"
        f"{legend_call}"
    )


def build() -> Path:
    by_cost = _latest_by_cost()
    needed = {"Symmetric", "Asymmetric 10x"}
    missing = needed - set(by_cost.keys())
    if missing:
        raise KeyError(
            f"missing cost variants in mean-shift parquets: {missing}"
        )

    agg_sym = _aggregate(by_cost["Symmetric"])
    agg_asym = _aggregate(by_cost["Asymmetric 10x"])

    sym_log = _panel_block(
        agg_sym,
        title="Symmetric $0/1$ cost",
        log_scale=True,
        tag="sym_log",
        show_legend=False,
        show_xlabel=False,
        show_ylabel=True,
    )
    asym_log = _panel_block(
        agg_asym,
        title="Asymmetric $10\\times$ cost",
        log_scale=True,
        tag="asym_log",
        show_legend=False,
        show_xlabel=False,
        show_ylabel=False,
    )
    sym_lin = _panel_block(
        agg_sym,
        title="",
        log_scale=False,
        tag="sym_lin",
        show_legend=True,
        show_xlabel=True,
        show_ylabel=True,
    )
    asym_lin = _panel_block(
        agg_asym,
        title="",
        log_scale=False,
        tag="asym_lin",
        show_legend=False,
        show_xlabel=True,
        show_ylabel=False,
    )

    header = (
        "% Auto-generated by"
        " experiments.analysis.generate_meanshift_figure"
        " --- do not edit.\n"
        "% K mutually equidistant centroids in R^{K-1}; per-class iid"
        " unit shifts of magnitude xi (sigma = 1). Horizontal axis is\n"
        "% xi/sigma, the per-class Euclidean shift magnitude in units"
        " of class\n"
        "% noise. Top row log-log, bottom row linear; bands are"
        " normal-approx\n"
        "% 95% CIs from mean +/- 1.96 * SE across seeds.\n"
    )
    tex = (
        header
        + "\\noindent\\begin{tikzpicture}\n"
        + "\\begin{groupplot}[\n"
        + "    group style={group size=2 by 2,"
        " horizontal sep=1.6cm, vertical sep=1.6cm},\n"
        + "    width=0.46\\textwidth, height=5.0cm,\n"
        + "    grid=major, grid style={dotted, gray!40},\n"
        + "    xmode=log,\n"
        + "    tick label style={font=\\scriptsize},\n"
        + "    label style={font=\\footnotesize},\n"
        + "    title style={font=\\footnotesize},\n"
        + "]\n"
        + sym_log
        + asym_log
        + sym_lin
        + asym_lin
        + "\\end{groupplot}\n"
        + "\\end{tikzpicture}%\n"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(tex, encoding="utf-8")
    print(f"  Wrote {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build()
