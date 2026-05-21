"""Generate tex/figures/partition_strategies.tex from tikz_simplex module.

Reproduces the 3-panel simplex figure with tested coordinate math
instead of hand-computed values.

Usage:
    uv run python -m experiments.analysis.generate_partition_figure
"""

from pathlib import Path

import numpy as np

from experiments.analysis.tikz_simplex import (
    render_affine_simplex,
    render_multi_panel,
    render_voronoi_simplex,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tex" / "figures" / "partition_strategies.tex"


def build() -> str:
    """Return the TikZ source for the partition-strategies figure."""
    # Panel (a): Parallel Shift (MultiDimThreshold)
    # Biases b=[0, 0.2, 0.3] boost classes 1 & 2, shifting the triple point
    # toward vertex 0 (Top) at barycentric (0.5, 0.3, 0.2).
    panel_a = render_affine_simplex(
        params=np.array([0.0, 0.2, 0.3]),
        strategy_type="MultiDimThreshold",
        show_boundaries=True,
        show_triple_marker=True,
        show_region_labels=True,
        region_label_positions=[(0.2, 3.0), (2.3, -0.5), (-2.0, -0.3)],
        show_vertex_labels=False,
        show_medians=True,
    )

    # Panel (b): Voronoi Tessellation
    panel_b = render_voronoi_simplex(
        centroids=np.array(
            [
                [0.7, 0.20, 0.10],
                [0.35, 0.45, 0.20],
                [0.2, 0.25, 0.55],
            ]
        ),
        show_region_labels=True,
        region_label_positions=[(0.0, 3.8), (3.0, -2.0), (-3.0, -2.0)],
        show_vertex_labels=False,
        show_medians=True,
        show_boundaries=True,
        show_prototypes=True,
        prototype_inner_sep="2pt",
        prototype_label_anchors=["east", "west", "east"],
        prototype_label_offsets=[(-0.3, 0.0), (0.3, 0.0), (-0.3, 0.0)],
    )

    return render_multi_panel(
        [panel_a, panel_b],
        panel_spacing_cm=5.5,
        scale=0.5,
        panel_labels=["(a)", "(b)"],
    )


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
