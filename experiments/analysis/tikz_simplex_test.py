import math
import re
import typing

import numpy as np
import pytest

from experiments.analysis.tikz_simplex import (
    _LEFT,
    _RIGHT,
    _TOP,
    _affine_region_polygon,
    _compute_cost_triple_point,
    _compute_triple_point,
    _cost_intersection_on_edge,
    _intersection_on_edge_full,
    affine_boundary_intersections,
    bary_to_cart,
    bary_to_cart_array,
    cart_to_bary,
    cost_boundary_endpoints,
    cost_boundary_intersections,
    render_affine_simplex,
    render_cost_simplex,
    render_multi_panel,
    render_simplex_panel,
    render_voronoi_simplex,
)

# Coordinate transform tests


class TestBaryToCart:
    """Tests for barycentric <-> Cartesian coordinate transforms."""

    def test_vertex_0(self):
        """e0 = (1,0,0) maps to Top vertex."""
        x, y = bary_to_cart([1, 0, 0])
        assert x == pytest.approx(_TOP[0], abs=1e-9)
        assert y == pytest.approx(_TOP[1], abs=1e-9)

    def test_vertex_1(self):
        """e1 = (0,1,0) maps to Right vertex."""
        x, y = bary_to_cart([0, 1, 0])
        assert x == pytest.approx(_RIGHT[0], abs=1e-9)
        assert y == pytest.approx(_RIGHT[1], abs=1e-9)

    def test_vertex_2(self):
        """e2 = (0,0,1) maps to Left vertex."""
        x, y = bary_to_cart([0, 0, 1])
        assert x == pytest.approx(_LEFT[0], abs=1e-9)
        assert y == pytest.approx(_LEFT[1], abs=1e-9)

    def test_centroid(self):
        """(1/3, 1/3, 1/3) maps to the geometric centroid."""
        x, y = bary_to_cart([1 / 3, 1 / 3, 1 / 3])
        expected_x = (_TOP[0] + _RIGHT[0] + _LEFT[0]) / 3
        expected_y = (_TOP[1] + _RIGHT[1] + _LEFT[1]) / 3
        assert x == pytest.approx(expected_x, abs=1e-9)
        assert y == pytest.approx(expected_y, abs=1e-9)

    def test_edge_midpoint_01(self):
        """(0.5, 0.5, 0) maps to the midpoint of Top-Right edge."""
        x, y = bary_to_cart([0.5, 0.5, 0])
        expected_x = (_TOP[0] + _RIGHT[0]) / 2
        expected_y = (_TOP[1] + _RIGHT[1]) / 2
        assert x == pytest.approx(expected_x, abs=1e-9)
        assert y == pytest.approx(expected_y, abs=1e-9)

    def test_roundtrip(self):
        """bary -> cart -> bary is identity."""
        bary_orig = (0.3, 0.5, 0.2)
        x, y = bary_to_cart(bary_orig)
        p0, p1, p2 = cart_to_bary(x, y)
        assert p0 == pytest.approx(bary_orig[0], abs=1e-9)
        assert p1 == pytest.approx(bary_orig[1], abs=1e-9)
        assert p2 == pytest.approx(bary_orig[2], abs=1e-9)

    @pytest.mark.parametrize(
        "bary",
        [
            (0.7, 0.2, 0.1),
            (0.1, 0.1, 0.8),
            (0.0, 0.5, 0.5),
            (1.0, 0.0, 0.0),
        ],
    )
    def test_roundtrip_parametrized(self, bary):
        """Roundtrip works for various points on the simplex."""
        x, y = bary_to_cart(bary)
        result = cart_to_bary(x, y)
        np.testing.assert_allclose(result, bary, atol=1e-9)


class TestBaryToCartArray:
    """Tests for the vectorised bary_to_cart_array."""

    def test_vertices(self):
        """All three vertices at once."""
        pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        xs, ys = bary_to_cart_array(pts)
        np.testing.assert_allclose(
            xs, [_TOP[0], _RIGHT[0], _LEFT[0]], atol=1e-9
        )
        np.testing.assert_allclose(
            ys, [_TOP[1], _RIGHT[1], _LEFT[1]], atol=1e-9
        )

    def test_matches_scalar(self):
        """Vectorised version matches scalar version."""
        pts = np.array([[0.3, 0.5, 0.2], [0.1, 0.1, 0.8]])
        xs, ys = bary_to_cart_array(pts)
        for i in range(len(pts)):
            sx, sy = bary_to_cart(pts[i])
            assert xs[i] == pytest.approx(sx, abs=1e-12)
            assert ys[i] == pytest.approx(sy, abs=1e-12)


# Boundary intersection tests


class TestAffineIntersections:
    """Tests for boundary intersection computation."""

    def test_default_produces_edge_midpoints(self):
        """Argmax boundaries pass through edge midpoints."""
        params = np.array([1.0, 1.0, 1.0])
        result = affine_boundary_intersections(params, "Argmax")
        # Each boundary (i,j) intersects the opposite edge midpoint
        for i, j in result:
            pts = result[(i, j)]
            assert len(pts) >= 1, f"No intersection for ({i},{j})"
            # Check that at least one point is an edge midpoint
            k = 3 - i - j
            expected_bary = [0.0, 0.0, 0.0]
            expected_bary[i] = 0.5
            expected_bary[j] = 0.5
            expected_bary[k] = 0.0
            ex, ey = bary_to_cart(expected_bary)
            found = any(
                abs(pt[0] - ex) < 1e-6 and abs(pt[1] - ey) < 1e-6 for pt in pts
            )
            assert found, (
                f"Boundary ({i},{j}) doesn't pass through "
                f"midpoint of edge opposite {k}"
            )

    def test_threshold_ratio_uniform_same_as_default(self):
        """ThresholdRatio with t=[1,1,1] is the same as Argmax."""
        default = affine_boundary_intersections(
            np.array([1.0, 1.0, 1.0]), "Argmax"
        )
        ratio = affine_boundary_intersections(
            np.array([1.0, 1.0, 1.0]), "ThresholdRatio"
        )
        for key in default:
            for pt_d, pt_r in zip(sorted(default[key]), sorted(ratio[key])):
                assert pt_d[0] == pytest.approx(pt_r[0], abs=1e-9)
                assert pt_d[1] == pytest.approx(pt_r[1], abs=1e-9)

    def test_multidim_zero_biases_same_as_default(self):
        """MultiDimThreshold with b=[0,0,0] is the same as Argmax."""
        default = affine_boundary_intersections(
            np.array([1.0, 1.0, 1.0]), "Argmax"
        )
        multidim = affine_boundary_intersections(
            np.array([0.0, 0.0, 0.0]), "MultiDimThreshold"
        )
        for key in default:
            for pt_d, pt_m in zip(sorted(default[key]), sorted(multidim[key])):
                assert pt_d[0] == pytest.approx(pt_m[0], abs=1e-9)
                assert pt_d[1] == pytest.approx(pt_m[1], abs=1e-9)

    def test_intersections_on_simplex(self):
        """All intersection points lie on the simplex boundary."""
        params = np.array([1.0, 0.8, 0.6])
        result = affine_boundary_intersections(params, "ThresholdRatio")
        for (i, j), pts in result.items():
            for x, y in pts:
                p0, p1, p2 = cart_to_bary(x, y)
                # Should sum to 1
                assert p0 + p1 + p2 == pytest.approx(1.0, abs=1e-6)
                # All coords >= 0
                assert p0 >= -1e-6, f"p0={p0} for boundary ({i},{j})"
                assert p1 >= -1e-6, f"p1={p1} for boundary ({i},{j})"
                assert p2 >= -1e-6, f"p2={p2} for boundary ({i},{j})"
                # At least one coord should be ~0 (on an edge)
                assert min(p0, p1, p2) < 1e-6, (
                    f"Not on edge: ({p0:.4f}, {p1:.4f}, {p2:.4f})"
                )

    @pytest.mark.parametrize(
        "biases",
        [
            np.array([0.0, 0.13, 0.12]),  # similar to fitted experiment
            np.array([0.0, -0.2, 0.1]),
            np.array([0.1, 0.0, -0.1]),
        ],
    )
    def test_multidim_intersections_on_correct_edge(self, biases):
        """Each intersection point lies on the correct simplex edge
        (exactly one barycentric coordinate is ~0)."""
        result = affine_boundary_intersections(biases, "MultiDimThreshold")
        for (i, j), pts in result.items():
            for x, y in pts:
                bary = cart_to_bary(x, y)
                assert sum(bary) == pytest.approx(1.0, abs=1e-6)
                assert all(b >= -1e-6 for b in bary), (
                    f"Negative bary coord for ({i},{j}): {bary}"
                )
                n_zero = sum(abs(b) < 1e-6 for b in bary)
                assert n_zero >= 1, (
                    f"Point not on edge for ({i},{j}): bary={bary}"
                )

    @pytest.mark.parametrize(
        "biases",
        [
            np.array([0.0, 0.13, 0.12]),
            np.array([0.0, -0.2, 0.1]),
        ],
    )
    def test_multidim_regions_tile_simplex(self, biases):
        """The three affine region polygons should tile the simplex
        without gaps or overlaps (areas sum to simplex area)."""
        from experiments.analysis.tikz_simplex import (
            _affine_region_polygon,
            _compute_triple_point,
        )

        intersections = affine_boundary_intersections(
            biases, "MultiDimThreshold"
        )
        triple = _compute_triple_point(biases, "MultiDimThreshold")
        assert triple is not None

        total_area = 0.0
        for k in range(3):
            poly = _affine_region_polygon(k, triple, intersections)
            # Shoelace formula
            n = len(poly)
            area = 0.0
            for idx in range(n):
                x1, y1 = poly[idx]
                x2, y2 = poly[(idx + 1) % n]
                area += x1 * y2 - x2 * y1
            total_area += abs(area) / 2.0

        # Simplex area = (sqrt(3)/4) * side^2, side = distance between vertices
        side = math.sqrt(
            (_RIGHT[0] - _LEFT[0]) ** 2 + (_RIGHT[1] - _LEFT[1]) ** 2
        )
        expected_area = (math.sqrt(3) / 4) * side**2
        assert total_area == pytest.approx(expected_area, rel=0.01), (
            f"Areas {total_area:.4f} != {expected_area:.4f}"
        )


class TestTriplePoint:
    """Tests for triple point computation."""

    def test_default_at_centroid(self):
        """Argmax strategy triple point is at the centroid."""
        tp = _compute_triple_point(np.array([1.0, 1.0, 1.0]), "Argmax")
        cx = (_TOP[0] + _RIGHT[0] + _LEFT[0]) / 3
        cy = (_TOP[1] + _RIGHT[1] + _LEFT[1]) / 3
        assert tp[0] == pytest.approx(cx, abs=1e-9)
        assert tp[1] == pytest.approx(cy, abs=1e-9)

    def test_threshold_ratio_triple(self):
        """ThresholdRatio triple point is at (t0/S, t1/S, t2/S)."""
        t = np.array([3.0, 2.0, 1.0])
        tp = _compute_triple_point(t, "ThresholdRatio")
        expected = bary_to_cart(t / t.sum())
        assert tp[0] == pytest.approx(expected[0], abs=1e-9)
        assert tp[1] == pytest.approx(expected[1], abs=1e-9)

    def test_multidim_triple(self):
        """MultiDimThreshold triple point satisfies p_i + b_i = const."""
        b = np.array([0.0, 0.1, -0.1])
        tp = _compute_triple_point(b, "MultiDimThreshold")
        assert tp is not None
        p0, p1, p2 = cart_to_bary(tp[0], tp[1])
        # p_i + b_i should be equal for all i
        scores = [p0 + b[0], p1 + b[1], p2 + b[2]]
        assert scores[0] == pytest.approx(scores[1], abs=1e-9)
        assert scores[1] == pytest.approx(scores[2], abs=1e-9)


# TikZ output validation tests


class TestRenderAffine:
    """Tests for render_affine_simplex output."""

    def test_contains_fill_regions(self):
        """Output contains fill commands for all 3 regions."""
        tex = render_affine_simplex(np.array([1.0, 1.0, 1.0]), "Argmax")
        assert "\\fill[green!15]" in tex
        assert "\\fill[blue!15]" in tex
        assert "\\fill[red!15]" in tex

    def test_contains_triangle_frame(self):
        """Output draws the simplex triangle."""
        tex = render_affine_simplex(np.array([1.0, 1.0, 1.0]), "Argmax")
        assert "\\draw[thick] (Top) -- (Right) -- (Left) -- cycle;" in tex

    def test_contains_vertex_labels(self):
        """Output labels the simplex vertices."""
        tex = render_affine_simplex(np.array([1.0, 1.0, 1.0]), "Argmax")
        assert "$\\mathbf{e}_0$" in tex
        assert "$\\mathbf{e}_1$" in tex
        assert "$\\mathbf{e}_2$" in tex

    def test_contains_title(self):
        """Title appears in output when provided."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]), "Argmax", title="My Title"
        )
        assert "My Title" in tex

    def test_medians_when_requested(self):
        """Dashed median lines appear when show_medians=True."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            show_medians=True,
        )
        assert "dashed, gray!50" in tex

    def test_no_medians_when_disabled(self):
        """No median lines when show_medians=False."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            show_medians=False,
        )
        assert "dashed, gray!50" not in tex


class TestRenderVoronoi:
    """Tests for render_voronoi_simplex output."""

    @pytest.fixture
    def centroids(self):
        return np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )

    def test_contains_centroids(self, centroids):
        """Output contains coordinate definitions for all prototypes."""
        tex = render_voronoi_simplex(centroids)
        assert "\\coordinate (Mu0)" in tex
        assert "\\coordinate (Mu1)" in tex
        assert "\\coordinate (Mu2)" in tex

    def test_centroid_coordinates_correct(self, centroids):
        """Centroid Cartesian coordinates match bary_to_cart."""
        tex = render_voronoi_simplex(centroids)
        for k in range(3):
            ex, ey = bary_to_cart(centroids[k])
            # Check the coordinate appears in the output
            coord_str = f"({ex:.3f}, {ey:.3f})"
            assert coord_str in tex, (
                f"Mu{k} coordinate {coord_str} not found in output"
            )

    def test_contains_clipping_regions(self, centroids):
        """Output uses clipping-based Voronoi region rendering."""
        tex = render_voronoi_simplex(centroids)
        assert "\\clip (Top) -- (Right) -- (Left) -- cycle;" in tex
        assert "\\fill[green!15]" in tex
        assert "\\fill[blue!15]" in tex
        assert "\\fill[red!15]" in tex

    def test_contains_prototype_stars(self, centroids):
        """Prototype star markers appear in output."""
        tex = render_voronoi_simplex(centroids, show_prototypes=True)
        assert "star, star points=5" in tex
        assert "\\boldsymbol{\\mu}_0" in tex
        assert "\\boldsymbol{\\mu}_1" in tex
        assert "\\boldsymbol{\\mu}_2" in tex

    def test_no_prototypes_when_disabled(self, centroids):
        """No prototype markers when show_prototypes=False."""
        tex = render_voronoi_simplex(centroids, show_prototypes=False)
        assert "star, star points=5" not in tex


class TestRenderDataPoints:
    """Tests for data point rendering."""

    def test_data_points_rendered(self):
        """Data points appear as circle nodes in TikZ output."""
        data_pts = [
            (1.0, 2.0, 0),
            (-1.0, -1.0, 1),
            (0.5, 0.0, 2),
        ]
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=data_pts,
        )
        assert "\\node[circle" in tex
        # Check all three points are present
        assert "(1.000, 2.000)" in tex
        assert "(-1.000, -1.000)" in tex
        assert "(0.500, 0.000)" in tex

    def test_data_point_colors(self):
        """Each data point uses the color of its true class."""
        data_pts = [(0.0, 0.0, 0), (1.0, 1.0, 1), (2.0, 2.0, 2)]
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=data_pts,
        )
        assert "green!60!black" in tex
        assert "blue!60" in tex
        assert "red!60" in tex

    def test_no_data_points_section_when_none(self):
        """No data points section when data_points is None."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=None,
        )
        assert "Data points" not in tex


# High-level render tests


class TestRenderSimplexPanel:
    """Tests for the high-level render_simplex_panel dispatcher."""

    @pytest.mark.parametrize(
        "strategy", ["Argmax", "ThresholdRatio", "MultiDimThreshold"]
    )
    def test_affine_strategies(self, strategy):
        """Affine strategies produce valid TikZ with region fills."""
        params = {
            "Argmax": np.array([1.0, 1.0, 1.0]),
            "ThresholdRatio": np.array([1.0, 0.8, 0.6]),
            "MultiDimThreshold": np.array([0.0, 0.1, -0.1]),
        }[strategy]
        tex = render_simplex_panel(strategy, params)
        assert "\\fill[green!15]" in tex
        assert "\\draw[thick] (Top) -- (Right) -- (Left) -- cycle;" in tex
        assert strategy in tex  # title

    def test_voronoi_strategy(self):
        """Voronoi strategy produces TikZ with clipping and prototypes."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_simplex_panel("Voronoi", centroids)
        assert "\\coordinate (Mu0)" in tex
        assert "\\clip" in tex
        assert "Voronoi" in tex  # title


class TestRenderMultiPanel:
    """Tests for multi-panel standalone document rendering."""

    @pytest.fixture
    def four_panels(self):
        panels = []
        for strategy, params in [
            ("Argmax", np.array([1.0, 1.0, 1.0])),
            ("ThresholdRatio", np.array([1.0, 0.8, 0.6])),
            ("MultiDimThreshold", np.array([0.0, 0.1, -0.1])),
        ]:
            panels.append(render_simplex_panel(strategy, params))
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        panels.append(render_simplex_panel("Voronoi", centroids))
        return panels

    def test_standalone_document_structure(self, four_panels):
        """Output is a valid standalone LaTeX document."""
        tex = render_multi_panel(four_panels)
        assert tex.startswith("\\documentclass")
        assert "\\begin{tikzpicture}" in tex
        assert "\\end{tikzpicture}" in tex
        assert "\\end{document}" in tex

    def test_contains_all_panels(self, four_panels):
        """All four panels are present with xshift scoping."""
        tex = render_multi_panel(four_panels)
        assert tex.count("\\begin{scope}[xshift=") == 4

    def test_panel_spacing(self, four_panels):
        """Panels are spaced according to panel_spacing_cm."""
        tex = render_multi_panel(four_panels, panel_spacing_cm=8.0)
        assert "xshift=0.0cm" in tex or "xshift=0cm" in tex
        assert "xshift=8.0cm" in tex
        assert "xshift=16.0cm" in tex
        assert "xshift=24.0cm" in tex

    def test_uses_tikz_libraries(self, four_panels):
        """Output loads required TikZ libraries."""
        tex = render_multi_panel(four_panels)
        assert (
            "\\usetikzlibrary{shapes, calc, backgrounds, arrows.meta}" in tex
        )

    def test_panel_labels(self, four_panels):
        """Panel labels appear in output when provided."""
        tex = render_multi_panel(
            four_panels, panel_labels=["(a)", "(b)", "(c)", "(d)"]
        )
        assert "(a)" in tex
        assert "(b)" in tex
        assert "(c)" in tex
        assert "(d)" in tex

    def test_no_panel_labels_by_default(self, four_panels):
        """No panel label nodes when panel_labels is None."""
        tex = render_multi_panel(four_panels)
        assert "anchor=north west" not in tex


# New rendering options tests


class TestBoundaryLines:
    """Tests for show_boundaries option."""

    def test_show_boundaries_emits_draw_lines(self):
        """Three boundary lines from Int (one per class pair)."""
        tex = render_affine_simplex(
            np.array([0.0, 0.2, 0.3]),
            "MultiDimThreshold",
            show_boundaries=True,
        )
        count = tex.count("\\draw[thick, black!60] (Int)")
        assert count == 3, f"Expected 3 boundary lines, got {count}"

    def test_show_boundaries_default_off(self):
        """No boundary lines by default."""
        tex = render_affine_simplex(
            np.array([0.5, 0.3, 0.2]),
            "MultiDimThreshold",
        )
        assert "\\draw[thick, black!60]" not in tex

    def test_threshold_ratio_boundaries(self):
        """ThresholdRatio also produces 3 boundary lines."""
        tex = render_affine_simplex(
            np.array([1.0, 0.8, 0.6]),
            "ThresholdRatio",
            show_boundaries=True,
        )
        count = tex.count("\\draw[thick, black!60]")
        assert count == 3


class TestTripleMarker:
    """Tests for show_triple_marker option."""

    def test_marker_present(self):
        tex = render_affine_simplex(
            np.array([1.0, 0.8, 0.6]),
            "ThresholdRatio",
            show_triple_marker=True,
        )
        assert "\\node[circle, fill=black, inner sep=1.2pt] at (Int)" in tex

    def test_marker_absent_by_default(self):
        tex = render_affine_simplex(
            np.array([1.0, 0.8, 0.6]),
            "ThresholdRatio",
        )
        assert "fill=black, inner sep=1.2pt" not in tex


class TestRegionLabels:
    """Tests for show_region_labels option."""

    def test_manual_positions(self):
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            show_region_labels=True,
            region_label_positions=[(0.0, 2.5), (2.0, -1.0), (-2.0, -1.0)],
        )
        assert "$\\mathcal{P}_0$" in tex
        assert "$\\mathcal{P}_1$" in tex
        assert "$\\mathcal{P}_2$" in tex

    def test_auto_positions(self):
        """Region labels use heuristic positions by default."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            show_region_labels=True,
        )
        assert "$\\mathcal{P}_0$" in tex
        assert "$\\mathcal{P}_1$" in tex
        assert "$\\mathcal{P}_2$" in tex

    def test_no_labels_by_default(self):
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
        )
        assert "\\mathcal{P}" not in tex

    def test_voronoi_region_labels(self):
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(
            centroids,
            show_region_labels=True,
            region_label_positions=[(-0.8, 3.8), (3.0, -2.0), (-3.0, -2.0)],
        )
        assert "$\\mathcal{P}_0$" in tex
        assert "$\\mathcal{P}_2$" in tex


class TestVertexLabelsToggle:
    """Tests for show_vertex_labels option."""

    def test_vertex_labels_on_by_default(self):
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
        )
        assert "$\\mathbf{e}_0$" in tex

    def test_vertex_labels_off(self):
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            show_vertex_labels=False,
        )
        assert "$\\mathbf{e}_0$" not in tex
        assert "$\\mathbf{e}_1$" not in tex
        assert "$\\mathbf{e}_2$" not in tex

    def test_voronoi_vertex_labels_off(self):
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids, show_vertex_labels=False)
        assert "$\\mathbf{e}_0$" not in tex


class TestPrototypeOptions:
    """Tests for Voronoi prototype customization."""

    def test_prototype_inner_sep(self):
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids, prototype_inner_sep="2pt")
        assert "inner sep=2pt" in tex

    def test_prototype_label_anchors(self):
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(
            centroids,
            prototype_label_anchors=["west", "west", "east"],
        )
        assert "anchor=east" in tex

    def test_voronoi_show_boundaries(self):
        """Voronoi boundary lines are drawn when show_boundaries=True."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids, show_boundaries=True)
        # 3 perpendicular bisectors clipped to simplex
        assert tex.count("\\draw[thick, black!60]") == 3

    def test_voronoi_no_boundaries_by_default(self):
        """No boundary lines by default."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids)
        assert "\\draw[thick, black!60]" not in tex

    def test_prototype_fill_colors(self):
        """Prototype stars use distinct fill colors per class."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids)
        assert "fill=green!60!black" in tex
        assert "fill=blue!80" in tex
        assert "fill=red!80" in tex

    def test_prototype_label_offsets_custom(self):
        """Custom offsets appear in the TikZ output."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(
            centroids,
            prototype_label_offsets=[(0.4, 0.0), (0.3, 0.5), (-0.3, -0.2)],
        )
        assert "+(0.400,0.000)" in tex  # μ₀
        assert "+(0.300,0.500)" in tex  # μ₁ nudged up
        assert "+(-0.300,-0.200)" in tex  # μ₂

    def test_prototype_label_offsets_default_west(self):
        """Without offsets, west-anchored labels get (0.3, 0) offset."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids)
        assert "+(0.300,0.000)" in tex

    def test_prototype_label_offsets_default_east(self):
        """Without offsets, east-anchored labels get (-0.3, 0) offset."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(
            centroids,
            prototype_label_anchors=["west", "west", "east"],
        )
        assert "+(-0.300,0.000)" in tex  # μ₂ east anchor default

    def test_prototype_label_offsets_override_anchor_default(self):
        """Custom offsets override the anchor-based default direction."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        # east anchor would default to (-0.3, 0), but we override to (0.5, 0.1)
        tex = render_voronoi_simplex(
            centroids,
            prototype_label_anchors=["west", "west", "east"],
            prototype_label_offsets=[(0.3, 0.0), (0.3, 0.0), (0.5, 0.1)],
        )
        assert "anchor=east" in tex
        assert "+(0.500,0.100)" in tex

    def test_prototype_label_offsets_all_labels_present(self):
        """All three μ labels are still emitted with custom offsets."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(
            centroids,
            prototype_label_offsets=[(0.1, 0.2), (0.3, 0.4), (-0.5, -0.6)],
        )
        assert "$\\boldsymbol{\\mu}_0$" in tex
        assert "$\\boldsymbol{\\mu}_1$" in tex
        assert "$\\boldsymbol{\\mu}_2$" in tex


class TestGeneratedMatchesHandCoded:
    """Compare generated coords against hand-coded values."""

    def test_threshold_ratio_triple_point(self):
        """ThresholdRatio triple point matches (0.361, 0.625)."""
        tp = _compute_triple_point(np.array([1.0, 0.8, 0.6]), "ThresholdRatio")
        assert tp[0] == pytest.approx(0.361, abs=0.001)
        assert tp[1] == pytest.approx(0.625, abs=0.001)

    def test_multidim_triple_point(self):
        """MultiDimThreshold triple at bary (1/6, 11/30, 14/30)."""
        tp = _compute_triple_point(
            np.array([0.5, 0.3, 0.2]), "MultiDimThreshold"
        )
        bary = cart_to_bary(tp[0], tp[1])
        np.testing.assert_allclose(
            bary,
            [1 / 6, 11 / 30, 14 / 30],
            atol=1e-6,
        )

    def test_voronoi_centroids_cartesian(self):
        """Voronoi centroids match hand-coded Cartesian values."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        expected = [(0.0, 2.75), (1.516, -1.375), (-1.299, -1.0)]
        for k in range(3):
            x, y = bary_to_cart(centroids[k])
            assert x == pytest.approx(expected[k][0], abs=0.001)
            assert y == pytest.approx(expected[k][1], abs=0.001)

    def test_panel_dispatch_passes_new_kwargs(self):
        """render_simplex_panel correctly forwards new kwargs."""
        tex = render_simplex_panel(
            "ThresholdRatio",
            np.array([1.0, 0.8, 0.6]),
            show_boundaries=True,
            show_triple_marker=True,
            show_region_labels=True,
            region_label_positions=[(0, 3), (2.5, -0.5), (-2, -0.5)],
            show_vertex_labels=False,
        )
        assert "\\draw[thick, black!60]" in tex
        assert "fill=black, inner sep=1.2pt" in tex
        assert "$\\mathcal{P}_0$" in tex
        assert "$\\mathbf{e}_0$" not in tex


# cart_to_bary direct tests


class TestCartToBary:
    """Direct tests for the inverse transform (not just roundtrip)."""

    def test_top_vertex(self):
        p0, p1, p2 = cart_to_bary(_TOP[0], _TOP[1])
        np.testing.assert_allclose([p0, p1, p2], [1, 0, 0], atol=1e-12)

    def test_right_vertex(self):
        p0, p1, p2 = cart_to_bary(_RIGHT[0], _RIGHT[1])
        np.testing.assert_allclose([p0, p1, p2], [0, 1, 0], atol=1e-12)

    def test_left_vertex(self):
        p0, p1, p2 = cart_to_bary(_LEFT[0], _LEFT[1])
        np.testing.assert_allclose([p0, p1, p2], [0, 0, 1], atol=1e-12)

    def test_centroid(self):
        cx = (_TOP[0] + _RIGHT[0] + _LEFT[0]) / 3
        cy = (_TOP[1] + _RIGHT[1] + _LEFT[1]) / 3
        p0, p1, p2 = cart_to_bary(cx, cy)
        np.testing.assert_allclose(
            [p0, p1, p2], [1 / 3, 1 / 3, 1 / 3], atol=1e-12
        )

    def test_sum_to_one_for_arbitrary_interior_point(self):
        """Interior point bary coords sum to 1."""
        x, y = bary_to_cart([0.15, 0.35, 0.50])
        p0, p1, p2 = cart_to_bary(x, y)
        assert p0 + p1 + p2 == pytest.approx(1.0, abs=1e-12)


# Intersection equation verification tests


class TestIntersectionsSatisfyBoundaryEquation:
    """Verify intersection points actually lie on the boundary equation,
    not just on the simplex edge."""

    @pytest.mark.parametrize(
        "thresholds",
        [
            np.array([1.0, 0.8, 0.6]),
            np.array([3.0, 2.0, 1.0]),
            np.array([1.0, 1.0, 5.0]),
            np.array([0.5, 0.5, 0.5]),
        ],
    )
    def test_threshold_ratio_boundary_equation(self, thresholds):
        """Each intersection satisfies p_i/t_i = p_j/t_j."""
        result = affine_boundary_intersections(thresholds, "ThresholdRatio")
        for (i, j), pts in result.items():
            for x, y in pts:
                bary = cart_to_bary(x, y)
                score_i = bary[i] / thresholds[i]
                score_j = bary[j] / thresholds[j]
                assert score_i == pytest.approx(score_j, abs=1e-6), (
                    f"Boundary ({i},{j}): p_{i}/t_{i}={score_i:.6f} != "
                    f"p_{j}/t_{j}={score_j:.6f}, bary={bary}"
                )

    @pytest.mark.parametrize(
        "biases",
        [
            np.array([0.0, 0.13, 0.12]),
            np.array([0.0, -0.2, 0.1]),
            np.array([0.1, 0.0, -0.1]),
            np.array([0.05, 0.05, 0.05]),
        ],
    )
    def test_multidim_boundary_equation(self, biases):
        """Each intersection satisfies p_i + b_i = p_j + b_j."""
        result = affine_boundary_intersections(biases, "MultiDimThreshold")
        for (i, j), pts in result.items():
            for x, y in pts:
                bary = cart_to_bary(x, y)
                score_i = bary[i] + biases[i]
                score_j = bary[j] + biases[j]
                assert score_i == pytest.approx(score_j, abs=1e-6), (
                    f"Boundary ({i},{j}): p_{i}+b_{i}={score_i:.6f} != "
                    f"p_{j}+b_{j}={score_j:.6f}, bary={bary}"
                )


# Triple point lies on all three boundaries


class TestTriplePointOnAllBoundaries:
    """The triple point must satisfy all three pairwise boundary equations."""

    @pytest.mark.parametrize(
        "thresholds",
        [
            np.array([1.0, 1.0, 1.0]),
            np.array([3.0, 2.0, 1.0]),
            np.array([1.0, 0.5, 2.0]),
        ],
    )
    def test_threshold_ratio_triple_on_all_boundaries(self, thresholds):
        tp = _compute_triple_point(thresholds, "ThresholdRatio")
        assert tp is not None
        bary = cart_to_bary(tp[0], tp[1])
        scores = [bary[k] / thresholds[k] for k in range(3)]
        assert scores[0] == pytest.approx(scores[1], abs=1e-9)
        assert scores[1] == pytest.approx(scores[2], abs=1e-9)

    @pytest.mark.parametrize(
        "biases",
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.13, 0.12]),
            np.array([0.1, -0.2, 0.1]),
        ],
    )
    def test_multidim_triple_on_all_boundaries(self, biases):
        tp = _compute_triple_point(biases, "MultiDimThreshold")
        assert tp is not None
        bary = cart_to_bary(tp[0], tp[1])
        scores = [bary[k] + biases[k] for k in range(3)]
        assert scores[0] == pytest.approx(scores[1], abs=1e-9)
        assert scores[1] == pytest.approx(scores[2], abs=1e-9)

    def test_triple_inside_simplex(self):
        """Triple point for moderate params must lie inside the simplex."""
        tp = _compute_triple_point(np.array([1.0, 0.8, 1.2]), "ThresholdRatio")
        bary = cart_to_bary(tp[0], tp[1])
        assert all(b >= -1e-9 for b in bary), (
            f"Triple point outside simplex: {bary}"
        )
        assert sum(bary) == pytest.approx(1.0, abs=1e-9)


# Region tiling tests (extended to all affine strategies)


def _simplex_area():
    """Exact simplex area from vertex coordinates (shoelace)."""
    side = math.sqrt((_RIGHT[0] - _LEFT[0]) ** 2 + (_RIGHT[1] - _LEFT[1]) ** 2)
    return (math.sqrt(3) / 4) * side**2


def _shoelace(poly):
    """Signed area of polygon via shoelace formula."""
    n = len(poly)
    area = 0.0
    for idx in range(n):
        x1, y1 = poly[idx]
        x2, y2 = poly[(idx + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


class TestRegionTilingExtended:
    """All affine strategy regions must tile the simplex exactly."""

    @pytest.mark.parametrize(
        "params, strategy_type",
        [
            (np.array([1.0, 1.0, 1.0]), "Argmax"),
            (np.array([3.0, 2.0, 1.0]), "ThresholdRatio"),
            (np.array([1.0, 0.5, 2.0]), "ThresholdRatio"),
            (np.array([0.0, 0.0, 0.0]), "MultiDimThreshold"),
            (np.array([0.0, 0.13, 0.12]), "MultiDimThreshold"),
            (np.array([0.1, -0.2, 0.1]), "MultiDimThreshold"),
        ],
    )
    def test_regions_tile_simplex(self, params, strategy_type):
        """Three region polygons sum to exactly the simplex area."""
        intersections = affine_boundary_intersections(params, strategy_type)
        triple = _compute_triple_point(params, strategy_type)
        assert triple is not None

        total_area = 0.0
        for k in range(3):
            poly = _affine_region_polygon(k, triple, intersections)
            total_area += abs(_shoelace(poly))

        assert total_area == pytest.approx(_simplex_area(), rel=0.01)

    @pytest.mark.parametrize(
        "params, strategy_type",
        [
            (np.array([1.0, 1.0, 1.0]), "Argmax"),
            (np.array([3.0, 2.0, 1.0]), "ThresholdRatio"),
            (np.array([0.0, 0.13, 0.12]), "MultiDimThreshold"),
        ],
    )
    def test_all_regions_have_positive_area(self, params, strategy_type):
        """Each of the three regions has non-trivial area."""
        intersections = affine_boundary_intersections(params, strategy_type)
        triple = _compute_triple_point(params, strategy_type)
        assert triple is not None

        for k in range(3):
            poly = _affine_region_polygon(k, triple, intersections)
            area = abs(_shoelace(poly))
            assert area > 0.1, f"Region {k} has near-zero area: {area}"


# Region containment: vertex neighbourhood belongs to correct class


class TestRegionContainment:
    """Points near a simplex vertex should be classified into that class."""

    @pytest.mark.parametrize(
        "params, strategy_type",
        [
            (np.array([1.0, 1.0, 1.0]), "Argmax"),
            (np.array([1.0, 0.8, 0.6]), "ThresholdRatio"),
            (np.array([0.0, 0.1, -0.1]), "MultiDimThreshold"),
        ],
    )
    def test_vertex_neighbourhood_in_correct_region(
        self, params, strategy_type
    ):
        """A point 90% toward vertex k should be in region k."""
        intersections = affine_boundary_intersections(params, strategy_type)
        triple = _compute_triple_point(params, strategy_type)
        assert triple is not None

        for k in range(3):
            # Point very close to vertex k: (0.9 at k, 0.05, 0.05)
            bary = [0.05, 0.05, 0.05]
            bary[k] = 0.9
            test_x, test_y = bary_to_cart(bary)

            # Check which region contains this point using point-in-polygon
            poly = _affine_region_polygon(k, triple, intersections)
            assert _point_in_polygon(test_x, test_y, poly), (
                f"Point near vertex {k} (bary={bary}) not in region {k}"
            )


def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


# _intersection_on_edge_full direct tests


class TestIntersectionOnEdgeFull:
    """Direct tests for the low-level intersection function."""

    def test_default_on_edge_01(self):
        """Argmax boundary (0,1) on edge k=2 (p2=0) gives midpoint."""
        bary = _intersection_on_edge_full(
            0, 1, 2, np.array([1.0, 1.0, 1.0]), "Argmax"
        )
        assert bary is not None
        np.testing.assert_allclose(bary, [0.5, 0.5, 0.0], atol=1e-12)

    def test_default_on_wrong_edge_returns_none(self):
        """Argmax boundary (0,1) on edge k=0 (p0=0) is degenerate => None."""
        bary = _intersection_on_edge_full(
            0, 1, 0, np.array([1.0, 1.0, 1.0]), "Argmax"
        )
        assert bary is None

    def test_multidim_known_value(self):
        """MultiDimThreshold with b=[0, 0.2, 0]: boundary (0,1) on edge k=2.
        p0 - p1 = b1 - b0 = 0.2 and p0 + p1 = 1 => p0=0.6, p1=0.4."""
        biases = np.array([0.0, 0.2, 0.0])
        bary = _intersection_on_edge_full(0, 1, 2, biases, "MultiDimThreshold")
        assert bary is not None
        np.testing.assert_allclose(bary, [0.6, 0.4, 0.0], atol=1e-12)

    def test_threshold_ratio_known_value(self):
        """ThresholdRatio with t=[2, 1, 1]: boundary (0,1) on edge k=2.
        p0/2 = p1/1 and p0+p1=1 => p0=2/3, p1=1/3."""
        thresholds = np.array([2.0, 1.0, 1.0])
        bary = _intersection_on_edge_full(
            0, 1, 2, thresholds, "ThresholdRatio"
        )
        assert bary is not None
        np.testing.assert_allclose(bary, [2 / 3, 1 / 3, 0.0], atol=1e-12)

    def test_multidim_on_edge_where_i_is_zero(self):
        """MultiDimThreshold boundary (0,1) on edge k=0 (p0=0).
        p0=0 => -p1 = b1-b0 => p1 = b0-b1. With b=[0.3,0.1,0]: p1=0.2."""
        biases = np.array([0.3, 0.1, 0.0])
        bary = _intersection_on_edge_full(0, 1, 0, biases, "MultiDimThreshold")
        assert bary is not None
        assert bary[0] == pytest.approx(0.0, abs=1e-12)
        assert bary[1] == pytest.approx(0.2, abs=1e-12)
        assert bary[2] == pytest.approx(0.8, abs=1e-12)

    def test_out_of_range_returns_none(self):
        """MultiDimThreshold with large bias pushes intersection off-edge."""
        biases = np.array([0.0, 0.0, 2.0])  # b2 huge
        bary = _intersection_on_edge_full(0, 2, 1, biases, "MultiDimThreshold")
        # p0 + b0 = p2 + b2 on edge k=1: p0 - p2 = 2, p0 + p2 = 1 => p0 = 1.5
        assert bary is None


# Randomized property tests


def _multidim_biases_with_inside_triple(n_seeds):
    """Return the subset of seed indices for which random MultiDimThreshold
    biases produce a triple point strictly inside the simplex.

    Computed at collection time so `test_random_multidim_tiling` can
    parametrize only over valid seeds — avoids `pytest.skip` branches that
    would otherwise hide how many seeds actually ran.
    """
    valid = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        biases = rng.uniform(-0.15, 0.15, size=3)
        triple = _compute_triple_point(biases, "MultiDimThreshold")
        if triple is None:
            continue
        bary = cart_to_bary(triple[0], triple[1])
        if any(b < -1e-6 for b in bary):
            continue
        valid.append(seed)
    return valid


_MULTIDIM_TILING_SEED_COUNT = 20
_MULTIDIM_TILING_VALID_SEEDS = _multidim_biases_with_inside_triple(
    _MULTIDIM_TILING_SEED_COUNT
)


class TestRandomizedProperties:
    """Stress-test invariants with random parameter values."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_threshold_ratio_tiling(self, seed):
        """Random ThresholdRatio params always tile the simplex."""
        rng = np.random.default_rng(seed)
        thresholds = rng.uniform(0.1, 5.0, size=3)
        intersections = affine_boundary_intersections(
            thresholds, "ThresholdRatio"
        )
        triple = _compute_triple_point(thresholds, "ThresholdRatio")
        assert triple is not None

        total_area = sum(
            abs(_shoelace(_affine_region_polygon(k, triple, intersections)))
            for k in range(3)
        )
        assert total_area == pytest.approx(_simplex_area(), rel=0.01)

    def test_multidim_bias_range_exercises_inside_triple_case(self):
        """Guards the parametrize filter below: a healthy fraction of
        random seeds in [-0.15, 0.15] must yield triples inside the simplex,
        otherwise `test_random_multidim_tiling` would collect no cases."""
        assert (
            len(_MULTIDIM_TILING_VALID_SEEDS)
            >= _MULTIDIM_TILING_SEED_COUNT // 2
        )

    @pytest.mark.parametrize("seed", _MULTIDIM_TILING_VALID_SEEDS)
    def test_random_multidim_tiling(self, seed):
        """Random MultiDimThreshold biases (small enough for triple inside)
        always tile the simplex."""
        rng = np.random.default_rng(seed)
        biases = rng.uniform(-0.15, 0.15, size=3)
        triple = _compute_triple_point(biases, "MultiDimThreshold")
        assert triple is not None

        intersections = affine_boundary_intersections(
            biases, "MultiDimThreshold"
        )
        total_area = sum(
            abs(_shoelace(_affine_region_polygon(k, triple, intersections)))
            for k in range(3)
        )
        assert total_area == pytest.approx(_simplex_area(), rel=0.01)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_intersections_satisfy_equation(self, seed):
        """Random ThresholdRatio intersections satisfy p_i/t_i = p_j/t_j."""
        rng = np.random.default_rng(seed + 100)
        thresholds = rng.uniform(0.1, 5.0, size=3)
        result = affine_boundary_intersections(thresholds, "ThresholdRatio")
        for (i, j), pts in result.items():
            for x, y in pts:
                bary = cart_to_bary(x, y)
                assert bary[i] / thresholds[i] == pytest.approx(
                    bary[j] / thresholds[j], abs=1e-6
                )

    @pytest.mark.parametrize("seed", range(10))
    def test_random_vertex_containment(self, seed):
        """Points near each vertex are always in the correct region."""
        rng = np.random.default_rng(seed + 200)
        thresholds = rng.uniform(0.3, 3.0, size=3)
        intersections = affine_boundary_intersections(
            thresholds, "ThresholdRatio"
        )
        triple = _compute_triple_point(thresholds, "ThresholdRatio")
        assert triple is not None

        for k in range(3):
            bary = [0.03, 0.03, 0.03]
            bary[k] = 0.94
            tx, ty = bary_to_cart(bary)
            poly = _affine_region_polygon(k, triple, intersections)
            assert _point_in_polygon(tx, ty, poly), (
                f"seed={seed}, vertex {k}: point not in region"
            )


# Boundary count invariants


class TestBoundaryCountInvariants:
    """Each pairwise boundary should cross exactly 1 simplex edge
    for Argmax/ThresholdRatio, and 1-3 for MultiDimThreshold."""

    @pytest.mark.parametrize(
        "thresholds",
        [
            np.array([1.0, 1.0, 1.0]),
            np.array([3.0, 2.0, 1.0]),
            np.array([1.0, 0.5, 2.0]),
        ],
    )
    def test_threshold_ratio_one_intersection_per_boundary(self, thresholds):
        """ThresholdRatio: each boundary crosses exactly 1 edge
        (the edge opposite to the missing class)."""
        result = affine_boundary_intersections(thresholds, "ThresholdRatio")
        for (i, j), pts in result.items():
            assert len(pts) == 1, (
                f"Boundary ({i},{j}) has {len(pts)} intersections, expected 1"
            )

    def test_default_one_intersection_per_boundary(self):
        result = affine_boundary_intersections(
            np.array([1.0, 1.0, 1.0]), "Argmax"
        )
        for (i, j), pts in result.items():
            assert len(pts) == 1

    @pytest.mark.parametrize(
        "biases",
        [
            np.array([0.0, 0.13, 0.12]),
            np.array([0.0, -0.2, 0.1]),
        ],
    )
    def test_multidim_at_least_one_intersection(self, biases):
        """MultiDimThreshold: each boundary has at least 1 intersection."""
        result = affine_boundary_intersections(biases, "MultiDimThreshold")
        for (i, j), pts in result.items():
            assert len(pts) >= 1, f"Boundary ({i},{j}) has no intersections"


# TikZ structural validation (deeper than string matching)


class TestTikZStructuralValidation:
    """Deeper checks on the generated TikZ output structure."""

    def test_affine_region_count_matches_classes(self):
        """Exactly 3 \\fill commands for 3 classes."""
        tex = render_affine_simplex(
            np.array([1.0, 0.8, 0.6]), "ThresholdRatio"
        )
        fill_count = tex.count("\\fill[")
        assert fill_count == 3, f"Expected 3 fills, got {fill_count}"

    def test_voronoi_scope_count_matches_classes(self):
        """Exactly 3 clipping scopes for 3 Voronoi regions."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids)
        scope_count = tex.count("\\begin{scope}")
        assert scope_count == 3, f"Expected 3 scopes, got {scope_count}"

    def test_voronoi_centroid_coordinates_are_numerical(self):
        """Centroid coordinates in TikZ output parse as valid floats."""
        centroids = np.array(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.6, 0.25],
                [0.2, 0.25, 0.55],
            ]
        )
        tex = render_voronoi_simplex(centroids)
        for k in range(3):
            match = re.search(
                rf"\\coordinate \(Mu{k}\) at \(([^,]+), ([^)]+)\)", tex
            )
            assert match is not None, f"Mu{k} coordinate not found"
            x_parsed = float(match.group(1))
            y_parsed = float(match.group(2))
            ex, ey = bary_to_cart(centroids[k])
            assert x_parsed == pytest.approx(ex, abs=0.001)
            assert y_parsed == pytest.approx(ey, abs=0.001)

    def test_data_point_count_matches_input(self):
        """Number of rendered data point nodes matches input count."""
        data_pts = [(float(i), float(i), i % 3) for i in range(50)]
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=data_pts,
        )
        node_count = tex.count("\\node[circle")
        assert node_count == 50

    def test_multi_panel_xshifts_monotonically_increase(self):
        """Panel xshifts are in strictly increasing order."""
        panels = [
            render_simplex_panel("Argmax", np.array([1.0, 1.0, 1.0]))
            for _ in range(5)
        ]
        tex = render_multi_panel(panels, panel_spacing_cm=7.0)
        shifts = [
            float(m.group(1)) for m in re.finditer(r"xshift=([\d.]+)cm", tex)
        ]
        assert len(shifts) == 5
        assert shifts == sorted(shifts)
        assert shifts[-1] > shifts[0]

    def test_vertex_coordinates_in_output_match_constants(self):
        """Vertex coordinates in output match module constants."""
        tex = render_affine_simplex(np.array([1.0, 1.0, 1.0]), "Argmax")
        for name, expected in [
            ("Top", _TOP),
            ("Right", _RIGHT),
            ("Left", _LEFT),
        ]:
            match = re.search(
                rf"\\coordinate \({name}\) at \(([^,]+), ([^)]+)\)", tex
            )
            assert match is not None
            assert float(match.group(1)) == pytest.approx(
                expected[0], abs=0.001
            )
            assert float(match.group(2)) == pytest.approx(
                expected[1], abs=0.001
            )


# Edge cases


class TestEdgeCases:
    """Edge cases and degenerate parameter values."""

    def test_very_skewed_thresholds(self):
        """Extreme threshold ratios (100:1:1) still produce valid tiling."""
        thresholds = np.array([100.0, 1.0, 1.0])
        intersections = affine_boundary_intersections(
            thresholds, "ThresholdRatio"
        )
        triple = _compute_triple_point(thresholds, "ThresholdRatio")
        assert triple is not None

        total_area = sum(
            abs(_shoelace(_affine_region_polygon(k, triple, intersections)))
            for k in range(3)
        )
        assert total_area == pytest.approx(_simplex_area(), rel=0.02)

    def test_equal_params_are_symmetric(self):
        """Equal thresholds produce symmetric regions (equal areas)."""
        thresholds = np.array([1.0, 1.0, 1.0])
        intersections = affine_boundary_intersections(thresholds, "Argmax")
        triple = _compute_triple_point(thresholds, "Argmax")

        areas = [
            abs(_shoelace(_affine_region_polygon(k, triple, intersections)))
            for k in range(3)
        ]
        assert areas[0] == pytest.approx(areas[1], rel=0.01)
        assert areas[1] == pytest.approx(areas[2], rel=0.01)

    def test_uniform_biases_same_as_zero_biases(self):
        """MultiDimThreshold with b=[c,c,c] is equivalent to b=[0,0,0]."""
        result_zero = affine_boundary_intersections(
            np.array([0.0, 0.0, 0.0]), "MultiDimThreshold"
        )
        result_uniform = affine_boundary_intersections(
            np.array([0.3, 0.3, 0.3]), "MultiDimThreshold"
        )
        for key in result_zero:
            for pt_z, pt_u in zip(
                sorted(result_zero[key]), sorted(result_uniform[key])
            ):
                assert pt_z[0] == pytest.approx(pt_u[0], abs=1e-9)
                assert pt_z[1] == pytest.approx(pt_u[1], abs=1e-9)

    def test_proportional_thresholds_same_boundaries(self):
        """ThresholdRatio: t and alpha*t produce the same boundaries."""
        t = np.array([2.0, 3.0, 1.0])
        result1 = affine_boundary_intersections(t, "ThresholdRatio")
        result2 = affine_boundary_intersections(5.0 * t, "ThresholdRatio")
        for key in result1:
            for pt1, pt2 in zip(sorted(result1[key]), sorted(result2[key])):
                assert pt1[0] == pytest.approx(pt2[0], abs=1e-9)
                assert pt1[1] == pytest.approx(pt2[1], abs=1e-9)

    def test_render_panel_with_empty_data_points(self):
        """Empty data_points list should not produce data point nodes."""
        tex = render_simplex_panel(
            "Argmax",
            np.array([1.0, 1.0, 1.0]),
            data_points=[],
        )
        assert "\\node[circle" not in tex


# Cost-sensitive boundary tests

# Reusable cost matrices
SYMMETRIC_COST = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
ASYMMETRIC_COST = np.array([[0, 1, 5], [1, 0, 3], [1, 1, 0]], dtype=float)


class TestCostIntersectionOnEdge:
    """Tests for _cost_intersection_on_edge."""

    def test_symmetric_boundary_01_on_edge2(self):
        """Symmetric 0/1 loss boundary (0,1): a = C[0]-C[1] = [-1,1,0].
        On edge 2 (p2=0): p0 + p1 = 1, so -p0 + p1 = 0 => p0=p1=0.5."""
        a = SYMMETRIC_COST[0] - SYMMETRIC_COST[1]  # [-1, 1, 0]
        bary = _cost_intersection_on_edge(a, k=2)
        assert bary is not None
        assert bary[0] == pytest.approx(0.5, abs=1e-9)
        assert bary[1] == pytest.approx(0.5, abs=1e-9)
        assert bary[2] == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_boundary_01_on_edge0(self):
        """On edge 0 (p0=0): p1 + p2 = 1, a = [-1,1,0] => p1*1 + p2*0 = 0.
        p1 = 0 is a vertex, so should return None."""
        a = SYMMETRIC_COST[0] - SYMMETRIC_COST[1]
        bary = _cost_intersection_on_edge(a, k=0)
        assert bary is None  # lands on vertex

    def test_parallel_boundary_returns_none(self):
        """When a[m] == a[n], the boundary is parallel to the edge."""
        a = np.array([1.0, 1.0, 0.0])  # a[0] == a[1]
        bary = _cost_intersection_on_edge(a, k=2)
        assert bary is None

    def test_asymmetric_boundary_shifts(self):
        """Asymmetric cost should produce a shifted intersection."""
        a = ASYMMETRIC_COST[0] - ASYMMETRIC_COST[2]  # [-1, 0, 5]
        bary = _cost_intersection_on_edge(a, k=1)
        # On edge 1 (p1=0): p0 + p2 = 1, a[0]*p0 + a[2]*p2 = 0
        # -p0 + 5*p2 = 0, p0 = 5*p2, p0 + p2 = 1 => p2 = 1/6, p0 = 5/6
        assert bary is not None
        assert bary[0] == pytest.approx(5 / 6, abs=1e-9)
        assert bary[2] == pytest.approx(1 / 6, abs=1e-9)


class TestCostBoundaryIntersections:
    """Tests for cost_boundary_intersections."""

    def test_symmetric_cost_returns_all_pairs(self):
        """Symmetric 0/1 loss should produce intersections for all 3 pairs."""
        result = cost_boundary_intersections(SYMMETRIC_COST)
        assert len(result) == 3
        for pair in [(0, 1), (0, 2), (1, 2)]:
            assert pair in result

    def test_symmetric_cost_intersection_count(self):
        """Each boundary in symmetric 0/1 loss crosses exactly 2 edges
        (the medians of the simplex)."""
        result = cost_boundary_intersections(SYMMETRIC_COST)
        for pair, pts in result.items():
            assert len(pts) == 1, (
                f"Boundary {pair}: expected 1, got {len(pts)}"
            )

    def test_asymmetric_cost_has_intersections(self):
        """Asymmetric cost should still produce intersections."""
        result = cost_boundary_intersections(ASYMMETRIC_COST)
        total = sum(len(pts) for pts in result.values())
        assert total >= 3  # at least one per boundary pair

    def test_symmetric_intersections_match_affine_medians(self):
        """Under symmetric cost, boundaries should pass through edge midpoints
        (same as affine with zero biases)."""
        result = cost_boundary_intersections(SYMMETRIC_COST)
        midpoints = [
            ((_TOP[0] + _RIGHT[0]) / 2, (_TOP[1] + _RIGHT[1]) / 2),
            ((_TOP[0] + _LEFT[0]) / 2, (_TOP[1] + _LEFT[1]) / 2),
            ((_RIGHT[0] + _LEFT[0]) / 2, (_RIGHT[1] + _LEFT[1]) / 2),
        ]
        all_pts = [pt for pts in result.values() for pt in pts]
        for mid in midpoints:
            found = any(
                abs(pt[0] - mid[0]) < 1e-6 and abs(pt[1] - mid[1]) < 1e-6
                for pt in all_pts
            )
            assert found, f"Midpoint {mid} not found in intersections"


class TestComputeCostTriplePoint:
    """Tests for _compute_cost_triple_point."""

    def test_symmetric_cost_at_centroid(self):
        """Symmetric 0/1 loss triple point is at the centroid (1/3,1/3,1/3)."""
        triple = _compute_cost_triple_point(SYMMETRIC_COST)
        assert triple is not None
        expected_x = (_TOP[0] + _RIGHT[0] + _LEFT[0]) / 3
        expected_y = (_TOP[1] + _RIGHT[1] + _LEFT[1]) / 3
        assert triple[0] == pytest.approx(expected_x, abs=1e-6)
        assert triple[1] == pytest.approx(expected_y, abs=1e-6)

    def test_asymmetric_cost_inside_simplex(self):
        """Asymmetric cost triple point should lie inside the simplex."""
        triple = _compute_cost_triple_point(ASYMMETRIC_COST)
        assert triple is not None
        bary = cart_to_bary(triple[0], triple[1])
        for v in bary:
            assert v > -1e-6, f"Triple point outside simplex: bary={bary}"

    def test_asymmetric_cost_not_at_centroid(self):
        """Asymmetric cost shifts the triple point away from centroid."""
        triple = _compute_cost_triple_point(ASYMMETRIC_COST)
        centroid = _compute_cost_triple_point(SYMMETRIC_COST)
        assert triple is not None and centroid is not None
        dist = math.sqrt(
            (triple[0] - centroid[0]) ** 2 + (triple[1] - centroid[1]) ** 2
        )
        assert dist > 0.1, "Triple point should differ from centroid"

    def test_degenerate_cost_returns_none(self):
        """A cost matrix with identical rows makes boundaries undefined."""
        degen = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
        triple = _compute_cost_triple_point(degen)
        assert triple is None


class TestCostBoundaryEndpoints:
    """Tests for cost_boundary_endpoints."""

    def test_symmetric_boundary_01(self):
        """Symmetric 0/1 loss boundary (0,1) should have 2 endpoints."""
        pts = cost_boundary_endpoints(SYMMETRIC_COST, 0, 1)
        assert len(pts) == 2

    def test_endpoints_on_simplex_edges(self):
        """All returned endpoints should lie on or inside the simplex."""
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            pts = cost_boundary_endpoints(SYMMETRIC_COST, i, j)
            for pt in pts:
                bary = cart_to_bary(pt[0], pt[1])
                for v in bary:
                    assert v >= -0.05, (
                        f"Endpoint {pt} outside simplex for ({i},{j})"
                    )

    def test_asymmetric_boundary_02_has_endpoints(self):
        """Asymmetric cost boundary (0,2) should still produce endpoints."""
        pts = cost_boundary_endpoints(ASYMMETRIC_COST, 0, 2)
        assert len(pts) >= 2


class TestRenderCostSimplex:
    """Tests for render_cost_simplex."""

    def test_symmetric_renders_without_error(self):
        """Symmetric 0/1 loss renders a panel fragment with coordinates."""
        tex = render_cost_simplex(SYMMETRIC_COST)
        assert "\\coordinate" in tex
        assert "\\fill" in tex

    def test_region_fills_present(self):
        """Output should contain region fill commands."""
        tex = render_cost_simplex(SYMMETRIC_COST)
        for color in ["green!15", "blue!15", "red!15"]:
            assert color in tex

    def test_asymmetric_with_normal_annotations(self):
        """Asymmetric cost with normal_annotations renders the annotation."""
        tex = render_cost_simplex(
            ASYMMETRIC_COST,
            normal_annotations=[
                {
                    "boundary": (0, 2),
                    "position": 0.5,
                    "length": 1.6,
                    "label": r"$\mathbf{n}_{02}$",
                    "label_anchor": "above right",
                }
            ],
        )
        assert "\\coordinate" in tex
        assert "mathbf{n}" in tex

    def test_show_medians(self):
        """Enabling show_medians adds dashed lines."""
        tex = render_cost_simplex(SYMMETRIC_COST, show_medians=True)
        assert "dashed" in tex

    def test_no_medians_by_default(self):
        """Medians are off by default."""
        tex = render_cost_simplex(SYMMETRIC_COST)
        assert "dashed" not in tex

    def test_hide_boundaries(self):
        """show_boundaries=False should suppress boundary lines."""
        tex_with = render_cost_simplex(SYMMETRIC_COST, show_boundaries=True)
        tex_without = render_cost_simplex(
            SYMMETRIC_COST, show_boundaries=False
        )
        # The version with boundaries should have more draw commands
        assert tex_with.count("\\draw") > tex_without.count("\\draw")

    def test_custom_region_label_positions(self):
        """Custom region_label_positions should appear in output."""
        positions = [(0.0, 3.0), (2.0, -1.0), (-2.0, -1.0)]
        tex = render_cost_simplex(
            SYMMETRIC_COST,
            region_label_positions=positions,
        )
        assert "\\mathcal{P}_0" in tex

    def test_multi_panel_with_cost(self):
        """render_cost_simplex output works in render_multi_panel."""
        panel = render_cost_simplex(SYMMETRIC_COST)
        tex = render_multi_panel([panel], scale=0.5)
        assert "\\documentclass" in tex


class TestOverlayCostMatrix:
    """Tests for overlay_cost_matrix in render_cost_simplex.

    Beyond the rendering toggles, these tests pin down the geometric
    claims exercised by the O'Brien figures in main.tex: row-wise shifts
    ``A = C + alpha * 1^T`` translate boundaries parallel to C's (Prop.
    parallel_invariance), while column scaling ``A = C @ diag(beta)``
    rotates them (Prop. column_rotation).
    """

    ALPHA = np.array([-0.4, 0.2, 0.2])
    BETA = np.array([1.5, 0.7, 1.0])

    @staticmethod
    def _row_shifted(C, alpha):
        return C + np.asarray(alpha, dtype=float)[:, None]

    @staticmethod
    def _col_scaled(C, beta):
        return C * np.asarray(beta, dtype=float)[None, :]

    @staticmethod
    def _boundary_direction(C, i, j):
        """Unit direction vector of boundary (i,j) inside the simplex.

        Uses the segment from the triple point to the boundary's edge
        intersection (always well-defined when the triple point is
        inside and the pair is admissible).  Returns ``None`` only if
        the triple point is outside the simplex or the pair has no
        edge intersection.
        """
        triple = _compute_cost_triple_point(C)
        if triple is None:
            return None
        pts = cost_boundary_intersections(C).get(tuple(sorted((i, j))), [])
        if not pts:
            return None
        dx = pts[0][0] - triple[0]
        dy = pts[0][1] - triple[1]
        norm = math.hypot(dx, dy)
        return (dx / norm, dy / norm)

    # --- rendering toggles ------------------------------------------------

    def test_overlay_emits_dashed_segments(self):
        """overlay_cost_matrix adds dashed segments for the baseline."""
        A = self._row_shifted(ASYMMETRIC_COST, self.ALPHA)
        tex = render_cost_simplex(A, overlay_cost_matrix=ASYMMETRIC_COST)
        assert "dashed" in tex
        assert "OvInt" in tex

    def test_overlay_none_is_noop(self):
        """overlay_cost_matrix=None leaves the output unchanged."""
        A = self._row_shifted(ASYMMETRIC_COST, self.ALPHA)
        tex_with = render_cost_simplex(A, overlay_cost_matrix=None)
        tex_plain = render_cost_simplex(A)
        assert tex_with == tex_plain

    def test_overlay_triple_marker_toggle(self):
        """overlay_triple_marker gates the gray dot at the overlay triple."""
        A = self._row_shifted(ASYMMETRIC_COST, self.ALPHA)
        tex_off = render_cost_simplex(
            A,
            overlay_cost_matrix=ASYMMETRIC_COST,
            overlay_triple_marker=False,
        )
        tex_on = render_cost_simplex(
            A,
            overlay_cost_matrix=ASYMMETRIC_COST,
            overlay_triple_marker=True,
        )
        assert "gray!60" in tex_on and "fill=gray!60" in tex_on
        assert "fill=gray!60" not in tex_off

    # --- geometric invariants --------------------------------------------

    @pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
    def test_row_shift_boundaries_parallel_to_baseline(self, pair):
        """A = C + alpha*1^T keeps boundaries parallel to C's (Prop. 6.1)."""
        i, j = pair
        A = self._row_shifted(ASYMMETRIC_COST, self.ALPHA)
        dir_C = self._boundary_direction(ASYMMETRIC_COST, i, j)
        dir_A = self._boundary_direction(A, i, j)
        assert dir_C is not None and dir_A is not None
        cross = dir_C[0] * dir_A[1] - dir_C[1] * dir_A[0]
        assert abs(cross) < 1e-9, (
            f"Boundary ({i},{j}): row-shift should be parallel to baseline; "
            f"got cross={cross:.2e}"
        )

    def test_column_scale_rotates_at_least_one_boundary(self):
        """A = C @ diag(beta) produces at least one non-parallel boundary
        (Prop. 6.2).  The exact set of rotated pairs depends on C and beta;
        here we only assert the overall geometric difference."""
        A = self._col_scaled(ASYMMETRIC_COST, self.BETA)
        rotated = False
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            dir_C = self._boundary_direction(ASYMMETRIC_COST, i, j)
            dir_A = self._boundary_direction(A, i, j)
            if dir_C is None or dir_A is None:
                continue
            cross = dir_C[0] * dir_A[1] - dir_C[1] * dir_A[0]
            if abs(cross) > 1e-6:
                rotated = True
                break
        assert rotated, "column scaling should rotate at least one boundary"


# Data point rendering details


class TestDataPointRenderingDetails:
    """Tests for data point opacity, clipping, and edge ordering."""

    DATA_PTS: typing.ClassVar = [
        (0.0, 0.0, 0),
        (1.0, 1.0, 1),
        (2.0, 2.0, 2),
    ]

    def test_data_points_have_fill_opacity(self):
        """Data points should render with fill opacity."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=self.DATA_PTS,
        )
        assert "fill opacity=0.6" in tex

    def test_data_points_clipped_to_triangle_affine(self):
        """Data points in affine panels are wrapped in a clip scope."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=self.DATA_PTS,
        )
        # A begin{scope} + clip should wrap the data points
        data_pos = tex.index("Data points")
        scope_pos = tex.rfind("\\begin{scope}", 0, data_pos + 200)
        clip_pos = tex.index(
            "\\clip (Top) -- (Right) -- (Left) -- cycle;",
            scope_pos,
        )
        end_scope = tex.index("\\end{scope}", clip_pos)
        # Data point nodes should be between clip and end scope
        node_pos = tex.index("\\node[circle", clip_pos)
        assert clip_pos < node_pos < end_scope

    def test_data_points_clipped_to_triangle_voronoi(self):
        """Data points in Voronoi panels are wrapped in a clip scope."""
        centroids = np.array(
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        )
        tex = render_voronoi_simplex(centroids, data_points=self.DATA_PTS)
        data_pos = tex.index("Data points")
        # Find the scope+clip near the data points section
        scope_pos = tex.rfind("\\begin{scope}", 0, data_pos + 200)
        remaining = tex[scope_pos:]
        assert "\\clip (Top) -- (Right) -- (Left) -- cycle;" in remaining
        clip_pos = tex.index(
            "\\clip (Top) -- (Right) -- (Left) -- cycle;",
            scope_pos,
        )
        end_scope = tex.index("\\end{scope}", clip_pos)
        node_pos = tex.index("\\node[circle", clip_pos)
        assert clip_pos < node_pos < end_scope

    def test_triangle_frame_after_data_points_affine(self):
        """Triangle frame drawn after data points (on top)."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=self.DATA_PTS,
        )
        data_pos = tex.index("Data points")
        frame_pos = tex.index(
            "\\draw[thick] (Top) -- (Right) -- (Left) -- cycle;",
            data_pos,
        )
        assert frame_pos > data_pos

    def test_triangle_frame_after_data_points_voronoi(self):
        """Triangle frame drawn after data points in Voronoi."""
        centroids = np.array(
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        )
        tex = render_voronoi_simplex(centroids, data_points=self.DATA_PTS)
        data_pos = tex.index("Data points")
        frame_pos = tex.index(
            "\\draw[thick] (Top) -- (Right) -- (Left) -- cycle;",
            data_pos,
        )
        assert frame_pos > data_pos

    def test_data_points_no_white_edge(self):
        """Data points should not have white edge stroke."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            data_points=self.DATA_PTS,
        )
        assert "draw=white" not in tex
        assert "draw=none" in tex


# Title positioning


class TestTitlePositioning:
    """Tests for panel title anchor and position."""

    def test_title_anchor_south_east(self):
        """Panel titles should use south east anchor."""
        tex = render_affine_simplex(
            np.array([1.0, 1.0, 1.0]),
            "Argmax",
            title="(a)",
        )
        assert "anchor=south east" in tex
        assert "(a)" in tex

    def test_voronoi_title_anchor_south_east(self):
        """Voronoi panel titles should use south east anchor."""
        centroids = np.array(
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        )
        tex = render_voronoi_simplex(centroids, title="(d)")
        assert "anchor=south east" in tex
        assert "(d)" in tex
