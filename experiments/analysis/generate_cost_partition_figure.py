"""Generate tex/figures/simplex_cost_partition.tex from tikz_simplex module.

Two-panel figure: symmetric vs asymmetric cost matrix decision regions.

Usage:
    uv run python -m experiments.analysis.generate_cost_partition_figure
"""

from pathlib import Path

import numpy as np

from experiments.analysis.tikz_simplex import (
    render_cost_simplex,
    render_multi_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tex" / "figures" / "simplex_cost_partition.tex"


def build() -> str:
    """Return the TikZ source for the cost-partition figure."""
    # Panel (a): Symmetric 0/1 loss — standard argmax
    # C = [[0,1,1],[1,0,1],[1,1,0]]
    panel_a = render_cost_simplex(
        cost_matrix=np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
        ),
        show_boundaries=True,
        show_triple_marker=False,
        show_region_labels=True,
        region_label_positions=[(0.0, 2.5), (2.0, -1.0), (-2.0, -1.0)],
    )

    # Panel (b): Asymmetric cost — C_{20}=5, C_{21}=3 in the dissertation
    # convention (row = truth, column = prediction). Misclassifying true
    # class 2 is costly (more so as class 0). The matrix below is in the
    # Elkan layout that tikz_simplex.render_cost_simplex expects (row =
    # prediction, column = truth) — i.e., the transpose of the dissertation
    # convention; see cost_boundary_intersections in tikz_simplex.py.
    # Breaks the 0/1 symmetry: boundary 0-1 is no longer a median.
    panel_b = render_cost_simplex(
        cost_matrix=np.array(
            [
                [0, 1, 5],
                [1, 0, 3],
                [1, 1, 0],
            ]
        ),
        show_boundaries=True,
        show_medians=True,
        show_triple_marker=False,
        show_region_labels=True,
        region_label_positions=[(0.3, 3.5), (2.8, -1.0), (-1.8, -0.5)],
        normal_annotations=[
            {
                "boundary": (0, 1),
                "position": 0.5,
                "length": 1.6,
                "label": r"$\mathbf{n}_{01}$",
                "label_anchor": "above left",
            },
            {
                "boundary": (2, 0),
                "position": 0.5,
                "length": 1.6,
                "label": r"$\mathbf{n}_{20}$",
                "label_anchor": "below right",
            },
            {
                "boundary": (1, 2),
                "position": 0.5,
                "length": 1.6,
                "label": r"$\mathbf{n}_{12}$",
                "label_anchor": "below right",
            },
        ],
    )

    return render_multi_panel(
        [panel_a, panel_b],
        panel_spacing_cm=8.0,
        scale=0.65,
        panel_captions=[
            r"(a) $C_{ij} = 1 \;\forall\, i \neq j$",
            r"(b) $C_{20}{=}5,\; C_{21}{=}3$",
        ],
    )


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
