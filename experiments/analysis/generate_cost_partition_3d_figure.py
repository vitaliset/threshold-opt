"""Generate tex/figures/simplex_cost_partition_3d.png.

Two-panel 3D companion to Figure ``simplex_cost_partition``: each panel
shows the three conditional-risk linear functionals L_i(p) = c_i^T p as
planes over the 2-simplex, the lower envelope min_i L_i as opaque
facets, and the projection of the envelope onto the base (which
reproduces the 2D partition).

The matplotlib output is a high-DPI PNG matching the precedent set by
``sampler_partition_overlay.png`` (matplotlib-rendered figures are
tracked as PNG in this repo; ``*.pdf`` is globally gitignored). We use
matplotlib rather than pgfplots because matplotlib's
``Poly3DCollection`` does proper camera-relative depth sorting;
pgfplots' ``z buffer=sort`` only operates on patch/surf plots, not on
hand-drawn polygons via ``\\addplot3 coordinates``.

Cost-matrix layout: row = predicted class, column = true class (Elkan).
This is the transpose of the dissertation prose convention; with this
layout the risk of action i at p is ``(C @ p)[i]``. See
``tikz_simplex.cost_boundary_intersections`` for the same convention.

Usage:
    uv run python -m experiments.analysis.generate_cost_partition_3d_figure
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tex" / "figures" / "simplex_cost_partition_3d.png"

# Vertex positions of the 2-simplex in 2D Cartesian. Match the convention
# used by experiments.analysis.tikz_simplex (Top = e0, Right = e1, Left = e2),
# but at unit scale here — matplotlib handles aspect via box_aspect.
TOP = (0.0, 1.0)
RIGHT = (math.sqrt(3) / 2, -0.5)
LEFT = (-math.sqrt(3) / 2, -0.5)

# Palette matched to the 2D figure (xcolor mixes in tikz_simplex.py):
#   REGION_FILL_COLORS = ["green!15", "blue!15", "red!15"]
#   REGION_LABEL_COLORS = ["green!50!black", "blue!70!black", "red!70!black"]
# xcolor's `green!15` = 15% green (0,1,0) + 85% white → rgb(0.85,1,0.85).
# `green!50!black` = 50% green (0,1,0) + 50% black → rgb(0,0.5,0);
# analogously for blue!70!black and red!70!black.
REGION_FILL_LIGHT = [
    "#d8f5d8",
    "#d8d8f5",
    "#f5d8d8",
]  # green!15, blue!15, red!15
REGION_LABEL_COLORS = [
    "#008000",
    "#0000b3",
    "#b30000",
]  # !50!black, !70!black, !70!black
REGION_COLORS = (
    REGION_LABEL_COLORS  # back-compat alias for plane / edge colour
)


def bary_to_cart(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    x = p[..., 0] * TOP[0] + p[..., 1] * RIGHT[0] + p[..., 2] * LEFT[0]
    y = p[..., 0] * TOP[1] + p[..., 1] * RIGHT[1] + p[..., 2] * LEFT[1]
    return np.stack([x, y], axis=-1)


def simplex_grid(n: int = 60) -> tuple[np.ndarray, Triangulation]:
    coords = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            coords.append([i / n, j / n, k / n])
    bary = np.asarray(coords)
    cart = bary_to_cart(bary)
    triang = Triangulation(cart[:, 0], cart[:, 1])
    return bary, triang


def render_panel(
    ax,
    cost_matrix: np.ndarray,
    title: str,
    plane_label_positions_2d: dict[int, tuple[float, float]] | None = None,
    z_box_aspect: float = 1.4,
    region_label_offset_xy: dict[int, tuple[float, float]] | None = None,
) -> None:
    C = np.asarray(cost_matrix, dtype=float)
    bary, triang = simplex_grid(n=60)
    cart = bary_to_cart(bary)
    risks = C @ bary.T
    argmin = risks.argmin(axis=0)
    envelope = risks.min(axis=0)

    zmax = float(C.max())
    z_base = -0.25 * zmax

    # --- (1) Base projection: 2D partition shaded at z = z_base, with the
    # pastel ``green!15`` / ``blue!15`` / ``red!15`` palette of the 2D figure.
    base_polys: list[np.ndarray] = []
    base_colors: list[tuple[float, float, float, float]] = []
    for tri in triang.triangles:
        labels, counts = np.unique(argmin[tri], return_counts=True)
        if (counts.max() / counts.sum()) < 0.5:
            continue
        winner = int(labels[counts.argmax()])
        verts3d = np.column_stack(
            [cart[tri, 0], cart[tri, 1], np.full(3, z_base)]
        )
        rgba = list(plt.matplotlib.colors.to_rgba(REGION_FILL_LIGHT[winner]))
        rgba[3] = 0.9
        base_polys.append(verts3d)
        base_colors.append(tuple(rgba))
    ax.add_collection3d(
        Poly3DCollection(
            base_polys,
            facecolors=base_colors,
            edgecolor="none",
            linewidth=0,
        )
    )

    # Region labels on the base, placed just OUTSIDE the simplex past each
    # region's vertex. Pushing them into empty space avoids depth-occlusion by
    # the opaque envelope rising over the simplex; matplotlib 3D ``zorder``
    # alone doesn't override the depth comparison with polygon collections.
    vertex_xy = {0: TOP, 1: RIGHT, 2: LEFT}
    outward_t = 1.20  # >1 places the label past the vertex
    for k in range(3):
        mask = argmin == k
        if not mask.any():
            continue
        cx = float(cart[mask, 0].mean())
        cy = float(cart[mask, 1].mean())
        vx, vy = vertex_xy[k]
        lx = cx + outward_t * (vx - cx)
        ly = cy + outward_t * (vy - cy)
        if region_label_offset_xy is not None and k in region_label_offset_xy:
            dx, dy = region_label_offset_xy[k]
            lx += dx
            ly += dy
        ax.text(
            lx,
            ly,
            z_base + 0.05 * zmax,
            rf"$\mathcal{{P}}_{k}$",
            color=REGION_LABEL_COLORS[k],
            fontsize=18,
            ha="center",
            va="center",
            weight="bold",
            zorder=20,
        )

    # --- (2) Lower envelope as opaque triangulated facets, colored by argmin.
    # Use a medium-saturation tint of the 2D palette and ``edgecolor=none`` so
    # the per-facet edges don't create moiré specks visible in panel (a).
    env_polys: list[np.ndarray] = []
    env_colors: list[tuple[float, float, float, float]] = []
    for tri in triang.triangles:
        verts3d = np.column_stack([cart[tri, 0], cart[tri, 1], envelope[tri]])
        labels, counts = np.unique(argmin[tri], return_counts=True)
        winner = int(labels[counts.argmax()])
        rgba = list(plt.matplotlib.colors.to_rgba(REGION_FILL_LIGHT[winner]))
        # Mix the pastel toward the saturated colour so envelope sits visually
        # between the base and the planes.
        base_rgba = np.array(
            plt.matplotlib.colors.to_rgba(REGION_FILL_LIGHT[winner])
        )
        dark_rgba = np.array(
            plt.matplotlib.colors.to_rgba(REGION_LABEL_COLORS[winner])
        )
        mix = 0.35
        rgba = (1 - mix) * base_rgba + mix * dark_rgba
        rgba[3] = 0.95
        env_polys.append(verts3d)
        env_colors.append(tuple(rgba))
    ax.add_collection3d(
        Poly3DCollection(
            env_polys,
            facecolors=env_colors,
            edgecolor="none",
            linewidth=0,
        )
    )

    # --- (3) Three semi-transparent risk planes (light fill, darker edge).
    plane_verts_list: list[np.ndarray] = []
    for i in range(3):
        verts = np.array(
            [
                [TOP[0], TOP[1], C[i, 0]],
                [RIGHT[0], RIGHT[1], C[i, 1]],
                [LEFT[0], LEFT[1], C[i, 2]],
            ]
        )
        plane_verts_list.append(verts)
        poly = Poly3DCollection(
            [verts],
            facecolor=REGION_FILL_LIGHT[i],
            edgecolor=REGION_LABEL_COLORS[i],
            alpha=0.35,
            linewidth=1.0,
        )
        ax.add_collection3d(poly)

    # Plane labels — placed in 2D axes coordinates via ``text2D`` so they
    # never collide with plane edges (their position in the 3D scene depends
    # on cost-matrix-specific plane heights, which makes 3D placement
    # fragile). The screen positions below are tuned for view=(elev=24,
    # azim=-55) and are passed by the caller via ``plane_label_positions_2d``.
    if plane_label_positions_2d is not None:
        for i, (lx, ly) in plane_label_positions_2d.items():
            ax.text2D(
                lx,
                ly,
                rf"$\mathcal{{L}}_{i}$",
                color=REGION_LABEL_COLORS[i],
                fontsize=22,
                ha="center",
                va="center",
                weight="bold",
                transform=ax.transAxes,
                zorder=20,
            )

    # --- (4) Simplex base outline + vertex labels at z_base.
    base_x = [TOP[0], RIGHT[0], LEFT[0], TOP[0]]
    base_y = [TOP[1], RIGHT[1], LEFT[1], TOP[1]]
    ax.plot(base_x, base_y, [z_base] * 4, color="black", linewidth=0.8)
    for (vx, vy), lbl in [
        (TOP, r"$\mathbf{e}_0$"),
        (RIGHT, r"$\mathbf{e}_1$"),
        (LEFT, r"$\mathbf{e}_2$"),
    ]:
        ax.text(
            vx * 1.18,
            vy * 1.18,
            z_base - 0.04 * zmax,
            lbl,
            ha="center",
            va="top",
            fontsize=18,
            zorder=20,
        )

    ax.set_zlim(z_base - 0.05 * zmax, zmax * 1.10)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.70, 1.20)
    ax.set_box_aspect((1.8, 1.8, z_box_aspect))
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, weight="bold", y=0.96)
    ax.view_init(elev=24, azim=-55)


def _render_title_to_pil(
    title: str,
    dpi: int = 200,
    fontsize: int = 18,
) -> Image.Image:
    """Render only the panel caption as a tightly cropped PIL image.

    Captions go through their own one-line figure so the final composite
    can center each one exactly over its panel, independent of how
    matplotlib cropped the 3D scene.
    """
    fig = plt.figure(figsize=(8.0, 0.65))
    fig.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
    )
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _compose_panel_with_title(
    panel: Image.Image,
    title_img: Image.Image,
    gap_px: int = 12,
) -> Image.Image:
    """Stack title above panel, both horizontally centered in the result."""
    width = max(panel.width, title_img.width)
    height = title_img.height + gap_px + panel.height
    composed = Image.new("RGB", (width, height), "white")
    composed.paste(title_img, ((width - title_img.width) // 2, 0))
    composed.paste(
        panel, ((width - panel.width) // 2, title_img.height + gap_px)
    )
    return composed


def _render_panel_to_pil(
    cost_matrix: np.ndarray,
    title: str,
    plane_label_positions_2d: dict[int, tuple[float, float]],
    z_box_aspect: float = 1.4,
    dpi: int = 200,
    region_label_offset_xy: dict[int, tuple[float, float]] | None = None,
) -> Image.Image:
    """Render one 3D panel into a tightly-cropped PIL image.

    Each panel lives in its own one-axes figure so ``bbox_inches='tight'``
    crops cleanly to its content; with two axes in one figure the gap
    between them gets included in the tight bbox, which is what made the
    previous combined output look small. After matplotlib's crop, we run a
    second pass with PIL to trim any residual white border from 3D
    rendering padding.
    """
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    # Pass empty title — captions are rendered separately and composited
    # so they can be centered exactly over each panel.
    render_panel(
        ax,
        cost_matrix,
        title="",
        plane_label_positions_2d=plane_label_positions_2d,
        z_box_aspect=z_box_aspect,
        region_label_offset_xy=region_label_offset_xy,
    )

    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")

    # Second-pass crop: find the bounding box of non-white pixels and
    # tighten to it, keeping a small padding margin.
    rgb = np.asarray(img)
    nonwhite = np.any(rgb < 250, axis=-1)
    rows = np.any(nonwhite, axis=1)
    cols = np.any(nonwhite, axis=0)
    if rows.any() and cols.any():
        top, bottom = (
            int(np.argmax(rows)),
            int(len(rows) - 1 - np.argmax(rows[::-1])),
        )
        left, right = (
            int(np.argmax(cols)),
            int(len(cols) - 1 - np.argmax(cols[::-1])),
        )
        pad = 8
        img = img.crop(
            (
                max(0, left - pad),
                max(0, top - pad),
                min(img.width, right + 1 + pad),
                min(img.height, bottom + 1 + pad),
            )
        )

    return img


def _stitch_horizontally(
    images: list[Image.Image],
    gap_px: int = 40,
    background: str = "white",
) -> Image.Image:
    """Combine images side-by-side with a fixed pixel gap, top-aligned.

    The two panels have the same render size (same figsize/dpi) but after
    content cropping their heights can differ by a few pixels. Pad
    shorter panels at the bottom so they align cleanly.
    """
    max_h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + gap_px * (len(images) - 1)
    canvas = Image.new("RGB", (total_w, max_h), background)
    x = 0
    for im in images:
        y = (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap_px
    return canvas


def main() -> None:
    # Match the dissertation body: serif text + Computer Modern math glyphs
    # for $\mathcal{P}_k$ / $\mathcal{L}_i$ / $\mathbf{e}_j$ labels.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
        }
    )

    title_a = r"(a) $C_{ij} = 1\ \forall\, i \neq j$"
    title_b = r"(b) $C_{20}{=}5,\ C_{21}{=}3$"

    panel_a = _render_panel_to_pil(
        cost_matrix=np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
        ),
        title=title_a,
        plane_label_positions_2d={
            0: (0.70, 0.60),
            1: (0.22, 0.75),
            2: (0.56, 0.65),
        },
    )

    panel_b = _render_panel_to_pil(
        cost_matrix=np.array(
            [
                [0, 1, 5],
                [1, 0, 3],
                [1, 1, 0],
            ]
        ),
        title=title_b,
        plane_label_positions_2d={
            0: (0.50, 0.60),
            1: (0.22, 0.45),
            2: (0.70, 0.55),
        },
        region_label_offset_xy={1: (0.15, 0.0)},
    )

    panel_a = _compose_panel_with_title(panel_a, _render_title_to_pil(title_a))
    panel_b = _compose_panel_with_title(panel_b, _render_title_to_pil(title_b))

    combined = _stitch_horizontally([panel_a, panel_b], gap_px=40)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT} ({combined.width}x{combined.height})")


if __name__ == "__main__":
    main()
