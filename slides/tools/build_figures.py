"""Compile the static figures of the HTML deck to SVG.

Each figure is typeset alone, in the fonts of the deck, and converted with
``pdftocairo -svg``, which turns glyphs into paths so that the SVG needs no
font files. Figures shared with the dissertation are read from tex/;
figures made for the slides are in slides/figures/tex/. Nothing is written
to tex/: auxiliary files go to slides/.build/figures/.

The logos of the title slide are converted from tex/logos/*.pdf, which the
repository's *.pdf ignore rule keeps out of git; their SVGs are committed.

Usage: python slides/tools/build_figures.py [name ...]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "tex"
SLIDE_TEX = ROOT / "slides" / "figures" / "tex"
OUT = ROOT / "slides" / "figures" / "static"
BUILD = ROOT / "slides" / ".build" / "figures"

# Text width the figures were drawn for: 160mm paper with 3% and 5% margins.
PREAMBLE = r"""
\documentclass[12pt]{article}
\usepackage[T1]{fontenc}
\usepackage[mono=false]{libertinus}
\usepackage{libertinust1math}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath}
\usepackage[svgnames,x11names,table]{xcolor}
\usepackage{graphicx, booktabs, array, multirow, adjustbox}
\usepackage{tikz, pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes, arrows.meta, positioning, backgrounds, fit, calc, patterns}
\usepgfplotslibrary{fillbetween, groupplots}
\usepackage[subpreambles=false]{standalone}
\usepackage[active,tightpage]{preview}
\setlength{\PreviewBorder}{2pt}
\setlength{\textwidth}{419pt}
\setlength{\linewidth}{419pt}
\setlength{\hsize}{419pt}
\setlength{\parindent}{0pt}
\graphicspath{{figures/},{logos/}}
\begin{document}
\begin{preview}
%BODY%
\end{preview}
\end{document}
"""

# The thesis fragments carry their own float, caption and label; these are
# the redefinitions of \barefigure and \baretable in tex/slides.tex.
BARE = (
    r"\renewenvironment{figure}[1][]{}{}\renewenvironment{table}[1][]{}{}"
    r"\renewcommand{\caption}[1]{}\renewcommand{\label}[1]{}"
    r"\providecommand{\citep}[1]{}\renewcommand{\citep}[1]{}"
    r"\renewcommand{\centering}{}"
)


def slide(name: str) -> str:
    """A figure made for the slides, in slides/figures/tex/."""
    return inp((SLIDE_TEX / f"{name}.tex").as_posix())


def inp(path: str) -> str:
    return r"\input{%s}" % path


def bare(path: str) -> str:
    return BARE + inp(path)


FIGURES = {
    "pipeline_diagram": lambda: inp("figures/pipeline_diagram.tex"),
    "risk_lines": lambda: slide("risk_lines"),
    "calibration_sketch": lambda: slide("calibration_sketch"),
    "decision_learner": lambda: slide("decision_learner"),
    "split_bars": lambda: slide("split_bars"),
    "adult_mcc": lambda: bare("figures/ci_plots/adult_mcc.tex"),
    "win_bars_binary_cost": lambda: slide("win_bars_binary_cost"),
    "loss_gap": lambda: slide("loss_gap"),
    "win_bars_binary_max": lambda: slide("win_bars_binary_max"),
    "simplex_cost_partition": lambda: inp(
        "figures/simplex_cost_partition.tex"
    ),
    "simplex_distortions": lambda: inp("figures/simplex_distortions.tex"),
    "obrien_vs_column_scaling": lambda: slide("obrien_vs_column_scaling"),
    "partition_strategies": lambda: inp("figures/partition_strategies.tex"),
    "voronoi_witness": lambda: slide("voronoi_witness"),
    "simplex_regions": lambda: inp("figures/simplex_regions.tex"),
    "win_bars_multiclass_cost": lambda: slide("win_bars_multiclass_cost"),
    "win_bars_multiclass_max": lambda: slide("win_bars_multiclass_max"),
    "scaling_setup_geometry": lambda: inp(
        "figures/scaling_setup_geometry.tex"
    ),
    "scaling_variance": lambda: inp("figures/scaling_variance.tex"),
    "recipe_flowchart": lambda: slide("recipe_flowchart"),
    "dataset_summary_binary": lambda: bare(
        "tables/dataset_summary_binary.tex"
    ),
    "dataset_summary_multiclass": lambda: bare(
        "tables/dataset_summary_multiclass.tex"
    ),
    "win_counts_binary_cost": lambda: bare(
        "figures/win_counts/win_counts_binary_cost.tex"
    ),
    "win_counts_binary_max": lambda: bare(
        "figures/win_counts/win_counts_binary_max.tex"
    ),
    "win_counts_multiclass_cost": lambda: bare(
        "figures/win_counts/win_counts_multiclass_cost.tex"
    ),
    "win_counts_multiclass_max": lambda: bare(
        "figures/win_counts/win_counts_multiclass_max.tex"
    ),
    "scaling_meanshift": lambda: inp("figures/scaling_meanshift.tex"),
}

# Vector logos of the title slide and the one raster figure.
LOGOS = ["ime-logo", "usp-logo", "cc-by"]
COPIES = {
    "sampler_partition_overlay.png": TEX
    / "figures"
    / "sampler_partition_overlay.png",
    "bg-ime.svg": TEX / "logos" / "bg-ime.svg",
}


def to_svg(pdf: Path, svg: Path) -> None:
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)


def build(name: str) -> bool:
    tex = BUILD / f"{name}.tex"
    tex.write_text(
        PREAMBLE.replace("%BODY%", FIGURES[name]()), encoding="utf-8"
    )
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD}",
        str(tex),
    ]
    run = subprocess.run(
        cmd, cwd=TEX, capture_output=True, text=True, errors="replace"
    )
    pdf = BUILD / f"{name}.pdf"
    if run.returncode != 0 or not pdf.exists():
        tail = [
            ln
            for ln in run.stdout.splitlines()
            if ln.startswith("!") or ln.startswith("l.")
        ]
        print(f"FAILED {name}: {' | '.join(tail[:4])}")
        return False
    to_svg(pdf, OUT / f"{name}.svg")
    print(f"ok     {name}")
    return True


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(FIGURES)
    failed = [n for n in names if not build(n)]
    if not sys.argv[1:]:
        for logo in LOGOS:
            to_svg(TEX / "logos" / f"{logo}.pdf", OUT / f"{logo}.svg")
        for target, source in COPIES.items():
            shutil.copyfile(source, OUT / target)
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
