"""Generate TikZ simplex figures from fitted strategy parameters.

Produces standalone .tex files following the drawing conventions
established in tex/figures/partition_strategies.tex and
tex/figures/voronoi_vertical_stack.tex.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants — match the coordinate system used in existing TikZ figures
# ---------------------------------------------------------------------------
# Vertices of the equilateral triangle (simplex) in Cartesian coords.
# Top = class 0 (p0=1), Right = class 1 (p1=1), Left = class 2 (p2=1).
SIMPLEX_SCALE = 5.0
_TOP = (0.0, SIMPLEX_SCALE)
_RIGHT = (SIMPLEX_SCALE * math.sqrt(3) / 2, -SIMPLEX_SCALE / 2)
_LEFT = (-SIMPLEX_SCALE * math.sqrt(3) / 2, -SIMPLEX_SCALE / 2)

REGION_COLORS = ["green", "blue", "red"]
REGION_FILL_COLORS = ["green!15", "blue!15", "red!15"]
REGION_LABEL_COLORS = ["green!50!black", "blue!70!black", "red!70!black"]
POINT_COLORS = ["green!60!black", "blue!60", "red!60"]
PROTOTYPE_FILL_COLORS = ["green!60!black", "blue!80", "red!80"]


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def bary_to_cart(p: np.ndarray | Sequence[float]) -> tuple[float, float]:
    """Convert barycentric coordinates (p0, p1, p2) to 2-D Cartesian.

    Uses the convention: Top = e0, Right = e1, Left = e2.
    Coordinates are scaled to match the TikZ simplex size.
    """
    p = np.asarray(p, dtype=float)
    x = p[..., 0] * _TOP[0] + p[..., 1] * _RIGHT[0] + p[..., 2] * _LEFT[0]
    y = p[..., 0] * _TOP[1] + p[..., 1] * _RIGHT[1] + p[..., 2] * _LEFT[1]
    return float(x), float(y)


def bary_to_cart_array(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised version of :func:`bary_to_cart` for arrays of points."""
    p = np.asarray(p, dtype=float)
    x = p[..., 0] * _TOP[0] + p[..., 1] * _RIGHT[0] + p[..., 2] * _LEFT[0]
    y = p[..., 0] * _TOP[1] + p[..., 1] * _RIGHT[1] + p[..., 2] * _LEFT[1]
    return x, y


def cart_to_bary(x: float, y: float) -> tuple[float, float, float]:
    """Inverse of :func:`bary_to_cart`."""
    denom = (_RIGHT[1] - _LEFT[1]) * (_TOP[0] - _LEFT[0]) + (
        _LEFT[0] - _RIGHT[0]
    ) * (_TOP[1] - _LEFT[1])
    p0 = (
        (_RIGHT[1] - _LEFT[1]) * (x - _LEFT[0])
        + (_LEFT[0] - _RIGHT[0]) * (y - _LEFT[1])
    ) / denom
    p1 = (
        (_LEFT[1] - _TOP[1]) * (x - _LEFT[0])
        + (_TOP[0] - _LEFT[0]) * (y - _LEFT[1])
    ) / denom
    p2 = 1.0 - p0 - p1
    return float(p0), float(p1), float(p2)


# ---------------------------------------------------------------------------
# Boundary intersection computation
# ---------------------------------------------------------------------------


def affine_boundary_intersections(
    params: np.ndarray,
    strategy_type: str,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Compute Cartesian intersection points of pairwise boundaries with
    the simplex edges.

    Parameters
    ----------
    params : array, shape (3,)
        Thresholds (for ThresholdRatio) or biases (for MultiDimThreshold).
    strategy_type : str
        One of ``"Argmax"``, ``"ThresholdRatio"``, ``"MultiDimThreshold"``.

    Returns
    -------
    dict mapping ``(i, j)`` class pairs to a list of Cartesian ``(x, y)``
    intersection points.
    """
    params = np.asarray(params, dtype=float)
    result: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for i, j in combinations(range(3), 2):
        pts: list[tuple[float, float]] = []
        # Check all 3 edges (p_k = 0 for k in {0,1,2})
        for k in range(3):
            bary = _intersection_on_edge_full(i, j, k, params, strategy_type)
            if bary is None:
                continue
            cart = bary_to_cart(bary)
            if not any(
                abs(cart[0] - u[0]) < 1e-9 and abs(cart[1] - u[1]) < 1e-9
                for u in pts
            ):
                pts.append(cart)
        result[(i, j)] = pts
    return result


def _intersection_on_edge_full(
    i: int,
    j: int,
    k: int,
    params: np.ndarray,
    strategy_type: str,
) -> tuple[float, float, float] | None:
    """Find where boundary (i,j) crosses edge k (where p_k = 0).

    On edge k, p_k = 0 and the other two coordinates (call them a, b)
    satisfy p_a + p_b = 1. We solve the boundary equation for p_a, p_b.

    Returns barycentric coords or None if intersection is outside the edge.
    """
    # The two classes on this edge
    others = [c for c in range(3) if c != k]
    _a, _b = others  # p_a + p_b = 1 on this edge

    if strategy_type == "ThresholdRatio":
        # Boundary: p_i / t_i = p_j / t_j
        t = params
        if i == k:
            # p_i = 0 => 0/t_i = p_j/t_j => p_j = 0 => degenerate corner
            return None
        if j == k:
            # p_j = 0 => p_i/t_i = 0 => p_i = 0 => degenerate corner
            return None
        # Both i,j are on this edge: p_i + p_j = 1
        p_i = t[i] / (t[i] + t[j])
        p_j = t[j] / (t[i] + t[j])

    elif strategy_type == "MultiDimThreshold":
        # Boundary: p_i + b_i = p_j + b_j => p_i - p_j = b_j - b_i
        bi, bj = params[i], params[j]
        if i == k:
            # p_i = 0 => -p_j = b_j - b_i => p_j = b_i - b_j
            p_j = bi - bj
            if p_j < -1e-12 or p_j > 1.0 + 1e-12:
                return None
            p_i = 0.0
        elif j == k:
            # p_j = 0 => p_i = b_j - b_i
            p_i = bj - bi
            if p_i < -1e-12 or p_i > 1.0 + 1e-12:
                return None
            p_j = 0.0
        else:
            # Both i,j on this edge: p_i + p_j = 1
            p_i = (1.0 + bj - bi) / 2.0
            p_j = (1.0 + bi - bj) / 2.0
    else:
        # Argmax: boundary is p_i = p_j
        if i == k or j == k:
            return None  # degenerate corner
        p_i = 0.5
        p_j = 0.5

    if p_i < -1e-12 or p_j < -1e-12 or p_i > 1.0 + 1e-12 or p_j > 1.0 + 1e-12:
        return None

    # Exclude degenerate corner points (vertex of the simplex)
    n_zero = (abs(p_i) < 1e-12) + (abs(p_j) < 1e-12)
    if n_zero >= 1 and (k != i and k != j):
        # One of i,j is zero on edge k — this is a simplex vertex
        return None
    if abs(p_i) < 1e-12 and abs(p_j) < 1e-12:
        return None

    bary = [0.0, 0.0, 0.0]
    bary[i] = float(np.clip(p_i, 0.0, 1.0))
    bary[j] = float(np.clip(p_j, 0.0, 1.0))
    # Compute the third coordinate (may differ from k when k == i or k == j)
    third = 3 - i - j
    bary[third] = 1.0 - bary[i] - bary[j]
    if bary[third] < -1e-12:
        return None
    bary[third] = max(0.0, bary[third])
    return tuple(bary)


def _compute_triple_point(
    params: np.ndarray,
    strategy_type: str,
) -> tuple[float, float] | None:
    """Compute the triple point where all three boundaries meet.

    For affine strategies, this is where all three score functions are equal.
    """
    if strategy_type == "ThresholdRatio":
        # p_0/t_0 = p_1/t_1 = p_2/t_2 and p_0+p_1+p_2 = 1
        t = params
        s = t.sum()
        bary = (t[0] / s, t[1] / s, t[2] / s)
    elif strategy_type == "MultiDimThreshold":
        # p_0+b_0 = p_1+b_1 = p_2+b_2 and p_0+p_1+p_2 = 1
        b = params
        # p_i + b_i = c for all i => p_i = c - b_i, sum = 3c - sum(b) = 1
        # c = (1 + sum(b))/3, p_i = (1 + sum(b))/3 - b_i
        c = (1.0 + b.sum()) / 3.0
        bary = tuple(c - b[k] for k in range(3))
    else:
        bary = (1 / 3, 1 / 3, 1 / 3)

    if any(v < -1e-12 for v in bary):
        return None
    return bary_to_cart(bary)


# ---------------------------------------------------------------------------
# Cost-matrix boundary computation
# ---------------------------------------------------------------------------


def cost_boundary_intersections(
    cost_matrix: np.ndarray,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Compute edge intersections of cost-sensitive decision boundaries.

    Parameters
    ----------
    cost_matrix : array, shape (3, 3)
        ``cost_matrix[i][j]`` = cost of predicting class *i* when the true
        class is *j* (row = predicted class). Note this is the transpose of
        the sklearn convention used elsewhere in the codebase and in the
        dissertation prose; figure-generation scripts pass matrices already
        in this Elkan-style layout. The boundary between classes *i* and *j* is
        defined by ``sum_k p_k (C_ik - C_jk) = 0``.

    Returns
    -------
    dict mapping ``(i, j)`` class pairs to a list of Cartesian ``(x, y)``
    intersection points (excluding simplex vertices).
    """
    C = np.asarray(cost_matrix, dtype=float)
    result: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for i, j in combinations(range(3), 2):
        pts: list[tuple[float, float]] = []
        a = C[i] - C[j]  # boundary coefficients
        for k in range(3):
            bary = _cost_intersection_on_edge(a, k)
            if bary is None:
                continue
            cart = bary_to_cart(bary)
            if not any(
                abs(cart[0] - u[0]) < 1e-9 and abs(cart[1] - u[1]) < 1e-9
                for u in pts
            ):
                pts.append(cart)
        result[(i, j)] = pts
    return result


def _cost_intersection_on_edge(
    a: np.ndarray,
    k: int,
) -> tuple[float, float, float] | None:
    """Find where boundary ``sum_l p_l a_l = 0`` crosses edge *k*.

    On edge *k*, ``p_k = 0`` and ``p_m + p_n = 1`` for the other two
    indices *m*, *n*.  Returns barycentric coords or *None*.
    """
    others = [c for c in range(3) if c != k]
    m, n = others  # p_m + p_n = 1

    denom = a[n] - a[m]
    if abs(denom) < 1e-12:
        return None  # parallel to this edge

    p_m = a[n] / denom
    p_n = 1.0 - p_m

    if p_m < 1e-12 or p_n < 1e-12 or p_m > 1.0 - 1e-12 or p_n > 1.0 - 1e-12:
        return None  # outside edge interior (vertex or beyond)

    bary = [0.0, 0.0, 0.0]
    bary[m] = float(p_m)
    bary[n] = float(p_n)
    return tuple(bary)


def cost_boundary_endpoints(
    cost_matrix: np.ndarray,
    i: int,
    j: int,
) -> list[tuple[float, float]]:
    """Full boundary endpoints (including vertices) for boundary (i, j).

    Returns 0-2 Cartesian points.  Used for drawing full-length boundaries.
    """
    C = np.asarray(cost_matrix, dtype=float)
    a = C[i] - C[j]
    points: list[tuple[float, float]] = []

    # Check vertices (p_v = 1, others = 0)
    vertices = [_TOP, _RIGHT, _LEFT]
    for v in range(3):
        if abs(a[v]) < 1e-12:
            points.append(vertices[v])

    # Check edge interiors
    for k in range(3):
        bary = _cost_intersection_on_edge(a, k)
        if bary is None:
            continue
        points.append(bary_to_cart(bary))

    unique: list[tuple[float, float]] = []
    for pt in points:
        if not any(
            abs(pt[0] - u[0]) < 1e-9 and abs(pt[1] - u[1]) < 1e-9
            for u in unique
        ):
            unique.append(pt)
    return unique


def _compute_cost_triple_point(
    cost_matrix: np.ndarray,
) -> tuple[float, float] | None:
    """Triple point where all three cost-sensitive boundaries meet."""
    C = np.asarray(cost_matrix, dtype=float)
    A = np.array([C[0] - C[1], C[0] - C[2], [1, 1, 1]])
    b = np.array([0.0, 0.0, 1.0])
    try:
        bary = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if any(v < -1e-12 for v in bary):
        return None
    return bary_to_cart(bary)


# ---------------------------------------------------------------------------
# Region polygon computation (for affine strategies)
# ---------------------------------------------------------------------------


def _affine_region_polygon(
    class_idx: int,
    triple_point: tuple[float, float],
    intersections: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Build the polygon vertices for the decision region of *class_idx*.

    The region is bounded by the simplex vertex, two boundary segments
    (each going from the triple point to a simplex edge), and the simplex
    edges between them.
    """
    vertex = [_TOP, _RIGHT, _LEFT][class_idx]

    # Clockwise vertex ordering: 0(Top) -> 1(Right) -> 2(Left) -> 0
    cw1 = (class_idx + 1) % 3  # first clockwise neighbour
    cw2 = (class_idx + 2) % 3  # second clockwise neighbour

    # First edge (CW): vertex_k -> vertex_cw1, where p_{cw2} = 0.
    # Boundary crossing it separates class_idx from cw1.
    pair1 = tuple(sorted([class_idx, cw1]))
    edge1_zero = cw2

    # Second edge (CCW): vertex_k -> vertex_cw2, where p_{cw1} = 0.
    # Boundary crossing it separates class_idx from cw2.
    pair2 = tuple(sorted([class_idx, cw2]))
    edge2_zero = cw1

    def find_on_edge(pair, zero_coord):
        """Return the intersection point lying on the edge where
        barycentric coordinate *zero_coord* is approximately zero."""
        pts = intersections.get(pair, [])
        for pt in pts:
            bary = cart_to_bary(pt[0], pt[1])
            if abs(bary[zero_coord]) < 0.05:
                return pt
        return pts[0] if pts else triple_point

    pt1 = find_on_edge(pair1, edge1_zero)
    pt2 = find_on_edge(pair2, edge2_zero)

    # Polygon: vertex -> pt1 (CW edge) -> triple -> pt2 (CCW edge)
    return [vertex, pt1, triple_point, pt2]


# ---------------------------------------------------------------------------
# TikZ rendering
# ---------------------------------------------------------------------------

_PREAMBLE = r"""\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usepackage{amsmath}
\usetikzlibrary{shapes, calc, backgrounds, arrows.meta}
"""


def _fmt(v: float) -> str:
    """Format a coordinate value for TikZ."""
    return f"{v:.3f}"


def _render_data_points(
    data_points: list[tuple[float, float, int]],
) -> str:
    """Render subsampled test data points as small TikZ circles."""
    lines = ["\n    % --- Data points ---"]
    for x, y, label in data_points:
        color = POINT_COLORS[label]
        lines.append(
            f"    \\node[circle, fill={color}, fill opacity=0.6, "
            f"inner sep=0.8pt, draw=none] "
            f"at ({_fmt(x)}, {_fmt(y)}) {{}};"
        )
    return "\n".join(lines)


def _clip_polygon_halfplane(
    poly_bary: list[np.ndarray],
    c: np.ndarray,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """Sutherland-Hodgman clip of a barycentric polygon by ``c . p <= 0``."""
    out: list[np.ndarray] = []
    n = len(poly_bary)
    for idx in range(n):
        a = poly_bary[idx]
        b = poly_bary[(idx + 1) % n]
        sa = float(np.dot(c, a))
        sb = float(np.dot(c, b))
        a_in = sa <= tol
        b_in = sb <= tol
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = sa / (sa - sb)
            out.append(a + t * (b - a))
    return out


def _matrix_region_polygons(
    matrix: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """Cartesian polygons of the ``argmin_i (p^T A)_i`` simplex regions.

    Robust for any 3x3 ``A``: each region is the triangle clipped by the
    half-planes ``(A[:, i] - A[:, j]) . p <= 0``, so it does not depend on
    the three score-equality lines meeting at an interior triple point
    (they generally do not for a freely-learned partition matrix).
    """
    A = np.asarray(matrix, dtype=float).reshape(3, 3)
    triangle = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    polys: list[list[tuple[float, float]]] = []
    for i in range(3):
        poly = triangle
        for j in range(3):
            if j == i:
                continue
            poly = _clip_polygon_halfplane(poly, A[:, i] - A[:, j])
            if not poly:
                break
        polys.append([bary_to_cart(p) for p in poly])
    return polys


def render_affine_simplex(
    params: np.ndarray,
    strategy_type: str,
    *,
    title: str | None = None,
    data_points: list[tuple[float, float, int]] | None = None,
    show_medians: bool = True,
    show_boundaries: bool = False,
    show_triple_marker: bool = False,
    show_region_labels: bool = False,
    region_label_positions: list[tuple[float, float]] | None = None,
    show_vertex_labels: bool = True,
    triple_label: str | None = None,
    triple_label_anchor: str = "south",
    triple_label_offset: tuple[float, float] = (0.0, 0.3),
) -> str:
    """Generate TikZ code for an affine partition panel.

    Parameters
    ----------
    params : array, shape (3,)
        Thresholds or biases depending on *strategy_type*.
    strategy_type : str
        ``"Argmax"``, ``"ThresholdRatio"``, or ``"MultiDimThreshold"``.
    title : str, optional
        Panel title.
    data_points : list of (x, y, class_label), optional
        Subsampled NB posteriors in Cartesian coords.
    show_medians : bool
        Whether to draw the default argmax medians as dashed reference lines.
    show_boundaries : bool
        Whether to draw boundary line segments from the triple point to each
        edge intersection.
    show_triple_marker : bool
        Whether to draw a small filled circle at the triple point.
    show_region_labels : bool
        Whether to place :math:`\\mathcal{P}_k` labels in each region.
    region_label_positions : list of (x, y), optional
        Manual Cartesian positions for the three region labels.  When *None*
        and *show_region_labels* is True, positions are computed as polygon
        centroids.
    show_vertex_labels : bool
        Whether to draw vertex labels (:math:`p_0`, :math:`p_1`, :math:`p_2`).
    triple_label : str, optional
        LaTeX label to place near the triple point (e.g.
        ``r"$\\boldsymbol{\\tau}$"``).
    triple_label_anchor : str
        TikZ anchor for the triple-point label (default ``"south"``).
    triple_label_offset : (dx, dy)
        Offset from the triple point for the label (default ``(0.0, 0.3)``).
    """
    params = np.asarray(params, dtype=float)
    if strategy_type == "FullMatrix":
        # argmin_i (p^T A)_i is a linear partition: boundary (i, j) is
        # sum_k p_k (A_ki - A_kj) = 0, identical to a cost-matrix
        # partition with C = A^T. Reuse the cost-boundary helpers so the
        # region fills, scatter, and medians below are unchanged.
        cost_like = params.reshape(3, 3).T
        intersections = cost_boundary_intersections(cost_like)
        triple = _compute_cost_triple_point(cost_like)
    else:
        intersections = affine_boundary_intersections(params, strategy_type)
        triple = _compute_triple_point(params, strategy_type)

    lines = []

    # Vertices
    lines.append(
        f"    \\coordinate (Top) at ({_fmt(_TOP[0])}, {_fmt(_TOP[1])});"
    )
    lines.append(
        f"    \\coordinate (Right) at ({_fmt(_RIGHT[0])}, {_fmt(_RIGHT[1])});"
    )
    lines.append(
        f"    \\coordinate (Left) at ({_fmt(_LEFT[0])}, {_fmt(_LEFT[1])});"
    )
    lines.append("")

    # Region fills
    polys: list[list[tuple[float, float]]] = []
    if strategy_type == "FullMatrix":
        for k, poly in enumerate(_matrix_region_polygons(params)):
            polys.append(poly)
            if len(poly) >= 3:
                coords = " -- ".join(
                    f"({_fmt(pt[0])}, {_fmt(pt[1])})" for pt in poly
                )
                lines.append(
                    f"    \\fill[{REGION_FILL_COLORS[k]}] {coords} -- cycle;"
                )
    elif triple is not None:
        tx, ty = _fmt(triple[0]), _fmt(triple[1])
        lines.append(f"    \\coordinate (Int) at ({tx}, {ty});")
        lines.append("")
        for k in range(3):
            poly = _affine_region_polygon(k, triple, intersections)
            polys.append(poly)
            coords = " -- ".join(
                f"({_fmt(pt[0])}, {_fmt(pt[1])})" for pt in poly
            )
            lines.append(
                f"    \\fill[{REGION_FILL_COLORS[k]}] {coords} -- cycle;"
            )
    lines.append("")

    # Medians (dashed reference)
    if show_medians:
        mid_tr = ((_TOP[0] + _RIGHT[0]) / 2, (_TOP[1] + _RIGHT[1]) / 2)
        mid_tl = ((_TOP[0] + _LEFT[0]) / 2, (_TOP[1] + _LEFT[1]) / 2)
        mid_rl = ((_RIGHT[0] + _LEFT[0]) / 2, (_RIGHT[1] + _LEFT[1]) / 2)
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Left) -- "
            f"({_fmt(mid_tr[0])}, {_fmt(mid_tr[1])});"
        )
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Right) -- "
            f"({_fmt(mid_tl[0])}, {_fmt(mid_tl[1])});"
        )
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Top) -- "
            f"({_fmt(mid_rl[0])}, {_fmt(mid_rl[1])});"
        )
        lines.append("")

    # Boundary lines (from triple point to edge intersections)
    if show_boundaries and triple is not None:
        for (bi, bj), pts in intersections.items():
            # Draw from Int to the intersection on the edge opposite
            # the third class (p_k = 0 where k = 3 - i - j).
            bk = 3 - bi - bj
            best = None
            for pt in pts:
                bary = cart_to_bary(pt[0], pt[1])
                if best is None or abs(bary[bk]) < abs(
                    cart_to_bary(*best)[bk]
                ):
                    best = pt
            if best is None:
                best = pts[0]
            lines.append(
                f"    \\draw[thick, black!60] (Int) -- "
                f"({_fmt(best[0])}, {_fmt(best[1])});"
            )
        lines.append("")

    # Triple point marker
    if show_triple_marker and triple is not None:
        lines.append(
            "    \\node[circle, fill=black, inner sep=1.2pt] at (Int) {};"
        )
        if triple_label:
            dx, dy = triple_label_offset
            lines.append(
                f"    \\node[font=\\scriptsize, anchor={triple_label_anchor}] "
                f"at ({_fmt(triple[0] + dx)}, {_fmt(triple[1] + dy)}) "
                f"{{{triple_label}}};"
            )
        lines.append("")

    # Data points (clipped to triangle interior)
    if data_points:
        lines.append("    \\begin{scope}")
        lines.append("        \\clip (Top) -- (Right) -- (Left) -- cycle;")
        lines.append(_render_data_points(data_points))
        lines.append("    \\end{scope}")
        lines.append("")

    # Triangle frame (after data points so edges are on top)
    lines.append("    \\draw[thick] (Top) -- (Right) -- (Left) -- cycle;")
    lines.append("")

    # Region labels
    if show_region_labels:
        positions = region_label_positions
        if positions is None and polys:
            positions = [
                (
                    sum(pt[0] for pt in poly) / len(poly),
                    sum(pt[1] for pt in poly) / len(poly),
                )
                for poly in polys
            ]
        if positions:
            for k, (lx, ly) in enumerate(positions):
                lines.append(
                    f"    \\node[{REGION_LABEL_COLORS[k]}, "
                    f"font=\\footnotesize\\bfseries] "
                    f"at ({_fmt(lx)}, {_fmt(ly)}) "
                    f"{{$\\mathcal{{P}}_{k}$}};"
                )
            lines.append("")

    # Vertex labels
    if show_vertex_labels:
        lines.append(
            "    \\node[above, font=\\small] at (Top) {$\\mathbf{e}_0$};"
        )
        lines.append(
            "    \\node[right, font=\\small] at (Right) {$\\mathbf{e}_1$};"
        )
        lines.append(
            "    \\node[left, font=\\small] at (Left) {$\\mathbf{e}_2$};"
        )

    # Title
    if title:
        lines.append(
            f"    \\node[font=\\small\\bfseries, anchor=south east] "
            f"at ({_fmt(-2.5)}, {_fmt(_TOP[1] + 0.3)}) {{{title}}};"
        )

    return "\n".join(lines)


def render_cost_simplex(
    cost_matrix: np.ndarray,
    *,
    show_medians: bool = False,
    show_boundaries: bool = True,
    boundary_style: str = "thick, black!50",
    show_triple_marker: bool = True,
    triple_marker_sep: str = "1.5pt",
    show_region_labels: bool = True,
    region_label_positions: list[tuple[float, float]] | None = None,
    region_label_font: str = r"\small\bfseries",
    vertex_labels: list[str] | None = None,
    normal_annotations: list[dict] | None = None,
    overlay_cost_matrix: np.ndarray | None = None,
    overlay_boundary_style: str = "dashed, gray!60, thin",
    overlay_triple_marker: bool = False,
    overlay_triple_marker_sep: str = "0.8pt",
) -> str:
    """Generate TikZ code for a cost-sensitive partition panel.

    Parameters
    ----------
    cost_matrix : array, shape (3, 3)
        ``cost_matrix[i][j]`` = cost of predicting *i* when true class is *j*
        (row = predicted class; transpose of the sklearn convention used
        elsewhere — see ``cost_boundary_intersections``).
    show_boundaries : bool
        Draw full-length boundary lines (edge-to-edge or vertex-to-edge).
    boundary_style : str
        TikZ style for boundary lines.
    show_triple_marker : bool
        Draw a filled circle at the triple point.
    triple_marker_sep : str
        Inner sep for the triple-point marker.
    show_region_labels : bool
        Place region labels.
    region_label_positions : list of (x, y), optional
        Manual positions; auto-computed from polygon centroids if *None*.
    region_label_font : str
        Font command for region labels.
    vertex_labels : list of str, optional
        Labels for the three vertices.  Defaults to
        ``["$\\mathbf{e}_0$", "$\\mathbf{e}_1$", "$\\mathbf{e}_2$"]``.
    normal_annotations : list of dict, optional
        Each dict: ``{"boundary": (i, j), "position": float,
        "length": float, "label": str, "label_anchor": str}``.
        Draws a normal-vector arrow on the boundary.
    overlay_cost_matrix : array, shape (3, 3), optional
        A second cost matrix whose boundaries are rendered as dashed
        reference lines underneath the main boundaries.  Useful for
        showing how a parameterized partition matrix *A* shifts away
        from a baseline *C*.  Only the triple-point-to-edge segments
        are drawn (no region fills, no region labels).
    overlay_boundary_style : str
        TikZ style for the overlay segments.
    overlay_triple_marker : bool
        Whether to draw a small marker at the overlay's triple point.
    overlay_triple_marker_sep : str
        Inner sep for the overlay triple-point marker.
    """
    C = np.asarray(cost_matrix, dtype=float)
    intersections = cost_boundary_intersections(C)
    triple = _compute_cost_triple_point(C)

    lines = []

    # Vertices
    lines.append(
        f"    \\coordinate (Top) at ({_fmt(_TOP[0])}, {_fmt(_TOP[1])});"
    )
    lines.append(
        f"    \\coordinate (Right) at ({_fmt(_RIGHT[0])}, {_fmt(_RIGHT[1])});"
    )
    lines.append(
        f"    \\coordinate (Left) at ({_fmt(_LEFT[0])}, {_fmt(_LEFT[1])});"
    )
    lines.append("")

    # Region fills
    polys: list[list[tuple[float, float]]] = []
    if triple is not None:
        lines.append(
            f"    \\coordinate (Int) at "
            f"({_fmt(triple[0])}, {_fmt(triple[1])});"
        )
        lines.append("")
        for k in range(3):
            poly = _affine_region_polygon(k, triple, intersections)
            polys.append(poly)
            coords = " -- ".join(
                f"({_fmt(pt[0])}, {_fmt(pt[1])})" for pt in poly
            )
            lines.append(
                f"    \\fill[{REGION_FILL_COLORS[k]}] {coords} -- cycle;"
            )
    lines.append("")

    # Compute drawn boundary segments (triple → edge intersection)
    drawn_segments: dict[
        tuple[int, int],
        tuple[
            tuple[float, float],
            tuple[float, float],
        ],
    ] = {}
    if triple is not None:
        for (bi, bj), pts in intersections.items():
            bk = 3 - bi - bj
            best = None
            for pt in pts:
                bary = cart_to_bary(pt[0], pt[1])
                if best is None or abs(bary[bk]) < abs(
                    cart_to_bary(*best)[bk]
                ):
                    best = pt
            if best is None and pts:
                best = pts[0]
            if best is not None:
                drawn_segments[(bi, bj)] = (triple, best)

    # Overlay boundaries (dashed reference for a baseline cost matrix)
    if overlay_cost_matrix is not None:
        ov = np.asarray(overlay_cost_matrix, dtype=float)
        ov_inters = cost_boundary_intersections(ov)
        ov_triple = _compute_cost_triple_point(ov)
        if ov_triple is not None:
            lines.append(
                f"    \\coordinate (OvInt) at "
                f"({_fmt(ov_triple[0])}, {_fmt(ov_triple[1])});"
            )
            for (bi, bj), pts in ov_inters.items():
                bk = 3 - bi - bj
                best = None
                for pt in pts:
                    bary = cart_to_bary(pt[0], pt[1])
                    if best is None or abs(bary[bk]) < abs(
                        cart_to_bary(*best)[bk]
                    ):
                        best = pt
                if best is None and pts:
                    best = pts[0]
                if best is not None:
                    lines.append(
                        f"    \\draw[{overlay_boundary_style}] (OvInt) -- "
                        f"({_fmt(best[0])}, {_fmt(best[1])});"
                    )
            if overlay_triple_marker:
                lines.append(
                    f"    \\node[circle, fill=gray!60, "
                    f"inner sep={overlay_triple_marker_sep}] "
                    f"at (OvInt) {{}};"
                )
            lines.append("")

    if show_boundaries:
        for seg_end in drawn_segments.values():
            lines.append(
                f"    \\draw[{boundary_style}] (Int) -- "
                f"({_fmt(seg_end[1][0])}, {_fmt(seg_end[1][1])});"
            )
        lines.append("")

    # Medians (dashed reference)
    if show_medians:
        mid_tr = ((_TOP[0] + _RIGHT[0]) / 2, (_TOP[1] + _RIGHT[1]) / 2)
        mid_tl = ((_TOP[0] + _LEFT[0]) / 2, (_TOP[1] + _LEFT[1]) / 2)
        mid_rl = ((_RIGHT[0] + _LEFT[0]) / 2, (_RIGHT[1] + _LEFT[1]) / 2)
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Left) -- "
            f"({_fmt(mid_tr[0])}, {_fmt(mid_tr[1])});"
        )
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Right) -- "
            f"({_fmt(mid_tl[0])}, {_fmt(mid_tl[1])});"
        )
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Top) -- "
            f"({_fmt(mid_rl[0])}, {_fmt(mid_rl[1])});"
        )
        lines.append("")

    # Triangle frame
    lines.append("    \\draw[thick] (Top) -- (Right) -- (Left) -- cycle;")
    lines.append("")

    # Triple point marker
    if show_triple_marker and triple is not None:
        lines.append(
            f"    \\node[circle, fill=black, inner sep={triple_marker_sep}]"
            f" at (Int) {{}};"
        )
        lines.append("")

    # Normal vector annotations (positioned along drawn boundary segments)
    if normal_annotations and drawn_segments:
        lines.append("    % Normal vector annotations")
        for ann in normal_annotations:
            bi, bj = ann["boundary"]
            key = tuple(sorted([bi, bj]))
            seg = drawn_segments.get(key)
            if seg is None:
                continue
            start, end = seg
            pos = ann.get("position", 0.5)
            length = ann.get("length", 1.0)
            label = ann.get("label", "")
            label_anchor = ann.get("label_anchor", "above right")
            # Boundary direction and perpendicular (normal)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            norm = math.sqrt(dx * dx + dy * dy)
            if norm < 1e-12:
                continue
            # Perpendicular pointing toward class bi's region
            nx, ny = -dy / norm, dx / norm
            # Check direction: should point toward vertex of class bi
            vertex_i = [_TOP, _RIGHT, _LEFT][bi]
            mid = (
                start[0] + pos * dx,
                start[1] + pos * dy,
            )
            to_vertex = (
                vertex_i[0] - mid[0],
                vertex_i[1] - mid[1],
            )
            if nx * to_vertex[0] + ny * to_vertex[1] < 0:
                nx, ny = -nx, -ny
            arrow_dx = nx * length
            arrow_dy = ny * length
            lines.append(
                f"    \\coordinate (NormStart) at "
                f"({_fmt(mid[0])}, {_fmt(mid[1])});"
            )
            lines.append(
                f"    \\draw[->, >=Stealth, thick, black!70] (NormStart) "
                f"-- ++({_fmt(arrow_dx)}, {_fmt(arrow_dy)})"
            )
            lines.append(
                f"        node[{label_anchor}, font=\\footnotesize] "
                f"{{{label}}};"
            )
        lines.append("")

    # Region labels
    if show_region_labels:
        positions = region_label_positions
        if positions is None and polys:
            positions = [
                (
                    sum(pt[0] for pt in poly) / len(poly),
                    sum(pt[1] for pt in poly) / len(poly),
                )
                for poly in polys
            ]
        if positions:
            for k, (lx, ly) in enumerate(positions):
                lines.append(
                    f"    \\node[{REGION_LABEL_COLORS[k]}, "
                    f"font={region_label_font}] "
                    f"at ({_fmt(lx)}, {_fmt(ly)}) "
                    f"{{$\\mathcal{{P}}_{k}$}};"
                )
            lines.append("")

    # Vertex labels
    if vertex_labels is None:
        vertex_labels = [
            r"$\mathbf{e}_0$",
            r"$\mathbf{e}_1$",
            r"$\mathbf{e}_2$",
        ]
    anchors = ["above", "right", "left"]
    names = ["Top", "Right", "Left"]
    for k in range(3):
        if vertex_labels[k]:
            lines.append(
                f"    \\node[{anchors[k]}, font=\\small] "
                f"at ({names[k]}) {{{vertex_labels[k]}}};"
            )

    return "\n".join(lines)


def render_voronoi_simplex(
    centroids: np.ndarray,
    *,
    title: str | None = None,
    data_points: list[tuple[float, float, int]] | None = None,
    show_medians: bool = True,
    show_prototypes: bool = True,
    show_boundaries: bool = False,
    show_region_labels: bool = False,
    region_label_positions: list[tuple[float, float]] | None = None,
    show_vertex_labels: bool = True,
    prototype_inner_sep: str = "1pt",
    prototype_label_anchors: list[str] | None = None,
    prototype_label_offsets: list[tuple[float, float]] | None = None,
) -> str:
    """Generate TikZ code for a Voronoi partition panel.

    Parameters
    ----------
    centroids : array, shape (3, 3)
        Prototype positions in barycentric coordinates.
    title : str, optional
        Panel title.
    data_points : list of (x, y, class_label), optional
        Subsampled NB posteriors in Cartesian coords.
    show_medians : bool
        Whether to draw the default argmax medians as dashed reference lines.
    show_prototypes : bool
        Whether to draw star markers at prototype positions.
    show_boundaries : bool
        Whether to draw boundary lines (perpendicular bisectors) between
        Voronoi regions, clipped to the simplex.
    show_region_labels : bool
        Whether to place :math:`\\mathcal{P}_k` labels in each region.
    region_label_positions : list of (x, y), optional
        Manual Cartesian positions for the three region labels.
    show_vertex_labels : bool
        Whether to draw vertex labels.
    prototype_inner_sep : str
        Inner sep for prototype star markers (e.g. ``"1pt"``, ``"2pt"``).
    prototype_label_anchors : list of str, optional
        Per-prototype anchor for the :math:`\\mu_k` label (length 3).
        Defaults to ``["west", "west", "west"]``.
    prototype_label_offsets : list of (dx, dy), optional
        Per-prototype offset for the label position (length 3).
        Defaults to ``(0.3, 0)`` for west anchors and ``(-0.3, 0)`` for east.
    """
    centroids = np.asarray(centroids, dtype=float)
    cart = [bary_to_cart(centroids[k]) for k in range(3)]
    anchors = prototype_label_anchors or ["west", "west", "west"]
    offsets = prototype_label_offsets

    lines = []

    # Vertices
    lines.append(
        f"    \\coordinate (Top) at ({_fmt(_TOP[0])}, {_fmt(_TOP[1])});"
    )
    lines.append(
        f"    \\coordinate (Right) at ({_fmt(_RIGHT[0])}, {_fmt(_RIGHT[1])});"
    )
    lines.append(
        f"    \\coordinate (Left) at ({_fmt(_LEFT[0])}, {_fmt(_LEFT[1])});"
    )
    lines.append("")

    # Prototypes
    for k in range(3):
        mx, my = _fmt(cart[k][0]), _fmt(cart[k][1])
        lines.append(f"    \\coordinate (Mu{k}) at ({mx}, {my});")
    lines.append("")

    # Voronoi regions via clipping
    for k in range(3):
        lines.append(f"    % Region {k} ({REGION_COLORS[k].title()})")
        lines.append("    \\begin{scope}")
        lines.append("        \\clip (Top) -- (Right) -- (Left) -- cycle;")
        others = [c for c in range(3) if c != k]
        for other in others:
            mid_name = f"M{k}{other}"
            lines.append(
                f"        \\path (Mu{k}) -- (Mu{other}) "
                f"coordinate[midway] ({mid_name});"
            )
            lines.append(
                f"        \\clip ($({mid_name})!20cm!90:(Mu{other})$) -- "
                f"($({mid_name})!20cm!-90:(Mu{other})$) -- "
                f"($({mid_name})!20cm!180:(Mu{other})$) -- cycle;"
            )
        lines.append(
            f"        \\fill[{REGION_FILL_COLORS[k]}] (-6,-6) rectangle (6,6);"
        )
        lines.append("    \\end{scope}")
        lines.append("")

    # Medians (dashed reference)
    if show_medians:
        mid_tr = ((_TOP[0] + _RIGHT[0]) / 2, (_TOP[1] + _RIGHT[1]) / 2)
        mid_tl = ((_TOP[0] + _LEFT[0]) / 2, (_TOP[1] + _LEFT[1]) / 2)
        mid_rl = ((_RIGHT[0] + _LEFT[0]) / 2, (_RIGHT[1] + _LEFT[1]) / 2)
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Left) -- "
            f"({_fmt(mid_tr[0])}, {_fmt(mid_tr[1])});"
        )
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Right) -- "
            f"({_fmt(mid_tl[0])}, {_fmt(mid_tl[1])});"
        )
        lines.append(
            f"    \\draw[dashed, gray!50, thin] (Top) -- "
            f"({_fmt(mid_rl[0])}, {_fmt(mid_rl[1])});"
        )
        lines.append("")

    # Voronoi boundary lines (perpendicular bisectors clipped to regions)
    if show_boundaries:
        for k, other in [(0, 1), (0, 2), (1, 2)]:
            # Clip the bisector between k and other to region k,
            # so it only extends from the Voronoi vertex to the simplex edge.
            third = 3 - k - other
            lines.append("    \\begin{scope}")
            lines.append("        \\clip (Top) -- (Right) -- (Left) -- cycle;")
            # Also clip to region k (away from the third prototype)
            mid_third = f"Mc{k}{third}"
            lines.append(
                f"        \\path (Mu{k}) -- (Mu{third}) "
                f"coordinate[midway] ({mid_third});"
            )
            lines.append(
                f"        \\clip ($({mid_third})!20cm!90:(Mu{third})$) -- "
                f"($({mid_third})!20cm!-90:(Mu{third})$) -- "
                f"($({mid_third})!20cm!180:(Mu{third})$) -- cycle;"
            )
            mid_name = f"Mb{k}{other}"
            lines.append(
                f"        \\path (Mu{k}) -- (Mu{other}) "
                f"coordinate[midway] ({mid_name});"
            )
            lines.append(
                f"        \\draw[thick, black!60] "
                f"($({mid_name})!20cm!90:(Mu{other})$) -- "
                f"($({mid_name})!20cm!-90:(Mu{other})$);"
            )
            lines.append("    \\end{scope}")
        lines.append("")

    # Data points (clipped to triangle interior)
    if data_points:
        lines.append("    \\begin{scope}")
        lines.append("        \\clip (Top) -- (Right) -- (Left) -- cycle;")
        lines.append(_render_data_points(data_points))
        lines.append("    \\end{scope}")
        lines.append("")

    # Triangle frame (after data points so edges are on top)
    lines.append("    \\draw[thick] (Top) -- (Right) -- (Left) -- cycle;")
    lines.append("")

    # Prototypes (stars, drawn on top of everything)
    if show_prototypes:
        for k in range(3):
            lines.append(
                f"    \\node[star, star points=5, star point ratio=2.25, "
                f"fill={PROTOTYPE_FILL_COLORS[k]}, draw=black, "
                f"inner sep={prototype_inner_sep}] "
                f"at (Mu{k}) {{}};"
            )
            anchor = anchors[k]
            if offsets and k < len(offsets):
                dx, dy = offsets[k]
            elif anchor == "east":
                dx, dy = -0.3, 0.0
            else:
                dx, dy = 0.3, 0.0
            lines.append(
                f"    \\node[font=\\scriptsize, {REGION_LABEL_COLORS[k]}, "
                f"anchor={anchor}] at ($(Mu{k})+({_fmt(dx)},{_fmt(dy)})$) "
                f"{{$\\boldsymbol{{\\mu}}_{k}$}};"
            )
        lines.append("")

    # Region labels
    if show_region_labels and region_label_positions:
        for k, (lx, ly) in enumerate(region_label_positions):
            lines.append(
                f"    \\node[{REGION_LABEL_COLORS[k]}, "
                f"font=\\footnotesize\\bfseries] "
                f"at ({_fmt(lx)}, {_fmt(ly)}) "
                f"{{$\\mathcal{{P}}_{k}$}};"
            )
        lines.append("")

    # Vertex labels
    if show_vertex_labels:
        lines.append(
            "    \\node[above, font=\\small] at (Top) {$\\mathbf{e}_0$};"
        )
        lines.append(
            "    \\node[right, font=\\small] at (Right) {$\\mathbf{e}_1$};"
        )
        lines.append(
            "    \\node[left, font=\\small] at (Left) {$\\mathbf{e}_2$};"
        )

    # Title
    if title:
        lines.append(
            f"    \\node[font=\\small\\bfseries, anchor=south east] "
            f"at ({_fmt(-2.5)}, {_fmt(_TOP[1] + 0.3)}) {{{title}}};"
        )

    return "\n".join(lines)


def render_simplex_panel(
    strategy_name: str,
    params: np.ndarray,
    *,
    title: str | None = None,
    data_points: list[tuple[float, float, int]] | None = None,
    show_medians: bool = True,
    show_vertex_labels: bool = True,
    show_region_labels: bool = False,
    region_label_positions: list[tuple[float, float]] | None = None,
    # Affine-only options
    show_boundaries: bool = False,
    show_triple_marker: bool = False,
    triple_label: str | None = None,
    triple_label_anchor: str = "south",
    triple_label_offset: tuple[float, float] = (0.0, 0.3),
    # Voronoi-only options
    show_prototypes: bool = True,
    prototype_inner_sep: str = "1pt",
    prototype_label_anchors: list[str] | None = None,
) -> str:
    """High-level: render a single simplex panel for the given strategy.

    Parameters
    ----------
    strategy_name : str
        One of ``"Argmax"``, ``"ThresholdRatio"``,
        ``"MultiDimThreshold"``, ``"Voronoi"``.
    params : array
        Strategy parameters: shape (3,) for affine, shape (3,3) for Voronoi.
    title : str, optional
        Panel title (defaults to *strategy_name*).
    data_points : list of (x, y, class_label), optional
        Subsampled NB posteriors in Cartesian coords.
    show_medians : bool
        Show dashed argmax reference lines.
    show_vertex_labels : bool
        Whether to draw vertex labels.
    show_region_labels : bool
        Whether to place region labels.
    region_label_positions : list of (x, y), optional
        Manual positions for region labels.
    show_boundaries : bool
        (Affine only) Draw boundary line segments.
    show_triple_marker : bool
        (Affine only) Draw a marker at the triple point.
    show_prototypes : bool
        (Voronoi only) Draw prototype star markers.
    prototype_inner_sep : str
        (Voronoi only) Inner sep for prototype stars.
    prototype_label_anchors : list of str, optional
        (Voronoi only) Per-prototype anchor for labels.
    """
    if title is None:
        title = strategy_name

    if strategy_name == "Voronoi":
        return render_voronoi_simplex(
            params,
            title=title,
            data_points=data_points,
            show_medians=show_medians,
            show_boundaries=show_boundaries,
            show_vertex_labels=show_vertex_labels,
            show_region_labels=show_region_labels,
            region_label_positions=region_label_positions,
            show_prototypes=show_prototypes,
            prototype_inner_sep=prototype_inner_sep,
            prototype_label_anchors=prototype_label_anchors,
        )
    else:
        return render_affine_simplex(
            params,
            strategy_name,
            title=title,
            data_points=data_points,
            show_medians=show_medians,
            show_vertex_labels=show_vertex_labels,
            show_boundaries=show_boundaries,
            show_triple_marker=show_triple_marker,
            triple_label=triple_label,
            triple_label_anchor=triple_label_anchor,
            triple_label_offset=triple_label_offset,
            show_region_labels=show_region_labels,
            region_label_positions=region_label_positions,
        )


def render_multi_panel(
    panels: list[str],
    *,
    panel_spacing_cm: float = 6.0,
    scale: float = 0.5,
    panel_labels: list[str] | None = None,
    panel_captions: list[str] | None = None,
    caption_y: float | None = None,
) -> str:
    """Combine multiple TikZ panels into a standalone multi-panel figure.

    Parameters
    ----------
    panels : list of str
        TikZ code for each panel (from :func:`render_simplex_panel`).
    panel_spacing_cm : float
        Horizontal spacing between panels in cm.
    scale : float
        Scale factor for each panel.
    panel_labels : list of str, optional
        Labels placed at the top-left of each panel (e.g. ``["(a)", "(b)"]``).
    panel_captions : list of str, optional
        Captions placed below each panel (e.g.
        ``[r"(a) $\\tau = (1/3,1/3,1/3)$", ...]``).
    caption_y : float, optional
        Y-coordinate of the caption's top edge (it is north-anchored) in the
        panel's local coordinate system. Defaults to ``_LEFT[1] - 0.25``, a
        small fixed gap below the triangle base that is independent of the
        caption's font height.
    """
    body_lines = [
        _PREAMBLE,
        r"\begin{document}",
        r"\begin{tikzpicture}[font=\sffamily]",
    ]

    for idx, panel in enumerate(panels):
        xshift = idx * panel_spacing_cm
        body_lines.append("")
        body_lines.append(
            f"    \\begin{{scope}}[xshift={xshift}cm, scale={scale}]"
        )
        if panel_labels and idx < len(panel_labels):
            body_lines.append(
                f"    \\node[font=\\small\\bfseries, anchor=north west] "
                f"at ({_fmt(_LEFT[0])}, {_fmt(_TOP[1] + 0.5)}) "
                f"{{{panel_labels[idx]}}};"
            )
        body_lines.append(panel)
        if panel_captions and idx < len(panel_captions):
            cy = caption_y if caption_y is not None else _LEFT[1] - 0.25
            body_lines.append(
                f"    \\node[font=\\small\\bfseries, anchor=north, "
                f"align=center] at (0, {_fmt(cy)}) "
                f"{{{panel_captions[idx]}}};"
            )
        body_lines.append("    \\end{scope}")

    body_lines.append("")
    body_lines.append(r"\end{tikzpicture}")
    body_lines.append(r"\end{document}")
    body_lines.append("")

    return "\n".join(body_lines)
