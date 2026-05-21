"""Generate tex/figures/marchetti_thresholding.tex from tikz_simplex module.

Reproduces the 3-panel figure showing how the decision threshold tau
shifts the partition boundaries in the probability simplex.

Usage:
    uv run python -m experiments.analysis.generate_marchetti_figure
"""

from pathlib import Path

import numpy as np

from experiments.analysis.tikz_simplex import (
    render_affine_simplex,
    render_multi_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tex" / "figures" / "marchetti_thresholding.tex"

TAU_LABEL = r"$\boldsymbol{\tau}$"


def build() -> str:
    """Return the TikZ source for the Marchetti thresholding figure."""
    # Panel (a): tau = (1/3, 1/3, 1/3) — standard argmax (b = [0, 0, 0])
    panel_a = render_affine_simplex(
        params=np.array([0.0, 0.0, 0.0]),
        strategy_type="MultiDimThreshold",
        show_boundaries=True,
        show_triple_marker=True,
        triple_label=TAU_LABEL,
        triple_label_anchor="south",
        triple_label_offset=(0.0, 0.3),
        show_region_labels=True,
        region_label_positions=[(0.0, 2.5), (2.0, -1.0), (-2.0, -1.0)],
        show_vertex_labels=False,
        show_medians=True,
    )

    # Panel (b): tau = (0.5, 0.3, 0.2) — b = [0, 0.2, 0.3]
    panel_b = render_affine_simplex(
        params=np.array([0.0, 0.2, 0.3]),
        strategy_type="MultiDimThreshold",
        show_boundaries=True,
        show_triple_marker=True,
        triple_label=TAU_LABEL,
        triple_label_anchor="south",
        triple_label_offset=(0.0, 0.3),
        show_region_labels=True,
        region_label_positions=[(0.2, 3.2), (2.3, -0.5), (-2.0, -0.3)],
        show_vertex_labels=False,
        show_medians=True,
    )

    # Panel (c): tau = (0.15, 0.25, 0.60) — b = [0, -0.10, -0.45]
    panel_c = render_affine_simplex(
        params=np.array([0.0, -0.10, -0.45]),
        strategy_type="MultiDimThreshold",
        show_boundaries=True,
        show_triple_marker=True,
        triple_label=TAU_LABEL,
        triple_label_anchor="south",
        triple_label_offset=(0.0, 0.3),
        show_region_labels=True,
        region_label_positions=[(-0.3, 2.0), (2.0, -1.0), (-3.2, -1.8)],
        show_vertex_labels=False,
        show_medians=True,
    )

    return render_multi_panel(
        [panel_a, panel_b, panel_c],
        panel_spacing_cm=5.5,
        scale=0.5,
        panel_captions=[
            (
                r"(a) $\boldsymbol{\tau}"
                r" = (\tfrac{1}{3},\tfrac{1}{3},\tfrac{1}{3})$"
            ),
            r"(b) $\boldsymbol{\tau} = (0.5, 0.3, 0.2)$",
            r"(c) $\boldsymbol{\tau} = (0.15, 0.25, 0.60)$",
        ],
    )


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
