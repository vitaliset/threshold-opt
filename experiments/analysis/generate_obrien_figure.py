"""Generate tex/figures/obrien_parallel_shifts.tex and
tex/figures/obrien_vs_column_scaling.tex from tikz_simplex module.

Figure A (4 panels): O'Brien's parallel-shift regularization under an
asymmetric cost matrix. The bias vector alpha is varied per panel to
expand each class region in turn, keeping boundaries parallel to the
baseline (dashed overlay).

Figure B (3 panels): parallel-shift versus column-scaling contrast on
the same cost matrix. Highlights that row-bias preserves boundary
normals (Prop. parallel_invariance), whereas column scaling rotates
them (Prop. column_rotation).

Usage:
    uv run python -m experiments.analysis.generate_obrien_figure
"""

from pathlib import Path

import numpy as np

from experiments.analysis.tikz_simplex import (
    render_cost_simplex,
    render_multi_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_A = REPO_ROOT / "tex" / "figures" / "obrien_parallel_shifts.tex"
OUT_B = REPO_ROOT / "tex" / "figures" / "obrien_vs_column_scaling.tex"

# Shared asymmetric cost matrix: reuses panel (b) of
# simplex_cost_partition.tex for visual continuity across the dissertation.
COST = np.array(
    [
        [0, 1, 5],
        [1, 0, 3],
        [1, 1, 0],
    ],
    dtype=float,
)


def row_shifted(C: np.ndarray, alpha) -> np.ndarray:
    """Return A = C + alpha per row (O'Brien's parallel shift)."""
    return C + np.asarray(alpha, dtype=float)[:, None]


def col_scaled(C: np.ndarray, beta) -> np.ndarray:
    """Return A = C @ diag(beta) (column scaling / adjusted priors)."""
    return C * np.asarray(beta, dtype=float)[None, :]


def _panel(
    A: np.ndarray,
    *,
    overlay: np.ndarray | None = None,
    region_label_positions=None,
) -> str:
    return render_cost_simplex(
        cost_matrix=A,
        show_boundaries=True,
        show_triple_marker=True,
        show_region_labels=True,
        region_label_positions=region_label_positions,
        region_label_font=r"\footnotesize\bfseries",
        overlay_cost_matrix=overlay,
        overlay_triple_marker=overlay is not None,
    )


def build_figure_a() -> str:
    # Baseline triple point is at (5/9, 3/9, 1/9) for this cost matrix:
    # class 0 dominates most of the simplex, class 2 is the smallest region.
    # Per-panel |alpha| is chosen so the triple point stays inside the simplex
    # in every panel (class 1 and class 2 baseline shares are small, so their
    # feasible shift magnitudes are correspondingly smaller than class 0's).
    panel_a = _panel(row_shifted(COST, [0.0, 0.0, 0.0]))
    panel_b = _panel(row_shifted(COST, [-0.4, 0.2, 0.2]), overlay=COST)
    panel_c = _panel(row_shifted(COST, [0.15, -0.3, 0.15]), overlay=COST)
    panel_d = _panel(row_shifted(COST, [0.1, 0.1, -0.2]), overlay=COST)

    return render_multi_panel(
        [panel_a, panel_b, panel_c, panel_d],
        panel_spacing_cm=5.2,
        scale=0.45,
        panel_captions=[
            r"(a) $\boldsymbol{\alpha} = (0,\,0,\,0)$",
            r"(b) $\boldsymbol{\alpha} = (-0.40,\,0.20,\,0.20)$",
            r"(c) $\boldsymbol{\alpha} = (0.15,\,-0.30,\,0.15)$",
            r"(d) $\boldsymbol{\alpha} = (0.10,\,0.10,\,-0.20)$",
        ],
    )


def build_figure_b() -> str:
    alpha = np.array([-0.3, 0.15, 0.15])
    beta = np.array([3.4, 5.0, 1.0])

    panel_a = _panel(COST)
    panel_b = _panel(row_shifted(COST, alpha), overlay=COST)
    panel_c = _panel(col_scaled(COST, beta), overlay=COST)

    return render_multi_panel(
        [panel_a, panel_b, panel_c],
        panel_spacing_cm=5.5,
        scale=0.5,
        panel_captions=[
            r"(a) $\mathbf{A} = \mathbf{C}$",
            (
                r"(b) $\mathbf{A} = \mathbf{C} + "
                r"\boldsymbol{\alpha}\mathbf{1}^\top$"
            ),
            (
                r"(c) $\mathbf{A} = \mathbf{C}\,"
                r"\mathrm{diag}(\boldsymbol{\beta})$"
            ),
        ],
    )


def main() -> None:
    OUT_A.write_text(build_figure_a(), encoding="utf-8")
    print(f"Wrote {OUT_A}")
    OUT_B.write_text(build_figure_b(), encoding="utf-8")
    print(f"Wrote {OUT_B}")


if __name__ == "__main__":
    main()
