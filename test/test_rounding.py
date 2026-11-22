# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
import numpy as np

# Internal dependencies
from svgpathtools import (Path, Line, Arc, QuadraticBezier, polyline, polygon,
                          round_corners, CornerRoundingError)


def assert_smooth_joins(test_case, path, closed):
    """Every joint (incl. the seam for closed paths) is connected and
    tangent-continuous."""
    n = len(path)
    test_case.assertGreater(n, 0)
    pairs = [(path[i], path[(i + 1) % n]) for i in range(n)] if closed \
        else [(path[i], path[i + 1]) for i in range(n - 1)]
    for first, second in pairs:
        test_case.assertAlmostEqual(first.end, second.start)
        test_case.assertTrue(
            np.isclose(first.unit_tangent(1), second.unit_tangent(0)),
            msg='not tangent: %r -> %r' % (first, second))


def arcs_of(path):
    return [seg for seg in path if isinstance(seg, Arc)]


class TestRoundCorners(unittest.TestCase):

    def test_open_right_angle(self):
        path = polyline(0 + 0j, 2 + 0j, 2 + 2j)
        original = Path(*path)
        rounded = round_corners(path, [0, 0.5, 0])

        self.assertEqual([type(s) for s in rounded], [Line, Arc, Line])
        assert_smooth_joins(self, rounded, closed=False)

        # The arc is a true circular arc of the requested radius.
        arc = arcs_of(rounded)[0]
        self.assertAlmostEqual(arc.radius.real, 0.5)
        self.assertAlmostEqual(arc.radius.imag, 0.5)
        self.assertAlmostEqual(abs(arc.center - arc.start), 0.5)
        self.assertAlmostEqual(abs(arc.center - arc.end), 0.5)
        # Tangent points sit 0.5 away from the corner on each edge.
        self.assertAlmostEqual(arc.start, 1.5 + 0j)
        self.assertAlmostEqual(arc.end, 2 + 0.5j)

        # Open path endpoints are unchanged.
        self.assertEqual(rounded.start, path.start)
        self.assertEqual(rounded.end, path.end)

        # The input path was not modified.
        self.assertEqual(path, original)

        # The result is real geometry: it has a well-defined length,
        # slightly shorter than the sharp original.
        self.assertLess(rounded.length(), path.length())
        self.assertAlmostEqual(rounded.length(),
                               4 - 2 * 0.5 + 0.5 * np.pi / 2, places=6)

    def test_direction_preserved(self):
        # Left turn then right turn; midpoints of the arcs must lie on
        # the correct side of each corner.
        path = polyline(0 + 0j, 4 + 0j, 4 + 4j, 0 + 4j)
        rounded = round_corners(path, 1.0)
        arcs = arcs_of(rounded)
        self.assertEqual(len(arcs), 2)
        # First corner (left turn at 4+0j): arc bulges up-left of corner.
        self.assertGreater(arcs[0].point(0.5).imag, 0)
        self.assertLess(arcs[0].point(0.5).real, 4)
        # Second corner (right turn at 4+4j): arc bulges down-left.
        self.assertLess(arcs[1].point(0.5).imag, 4)
        self.assertLess(arcs[1].point(0.5).real, 4)
        assert_smooth_joins(self, rounded, closed=False)

    def test_concave_corner(self):
        # Arrow-like concave polygon; vertex 2+1j is reflex.
        path = polygon(0 + 0j, 4 + 0j, 4 + 4j, 2 + 1j, 0 + 4j)
        rounded = round_corners(path, 0.3)
        self.assertEqual(len(arcs_of(rounded)), 5)
        assert_smooth_joins(self, rounded, closed=True)
        for arc in arcs_of(rounded):
            self.assertAlmostEqual(arc.radius.real, 0.3)
            self.assertAlmostEqual(abs(arc.center - arc.start), 0.3)
            self.assertAlmostEqual(abs(arc.center - arc.end), 0.3)
        # The concave fillet bulges away from the reflex vertex (2+1j),
        # i.e. its midpoint stays above the corner.
        reflex_arc = min(arcs_of(rounded),
                         key=lambda a: abs(a.point(0.5) - (2 + 1j)))
        self.assertGreater(reflex_arc.point(0.5).imag, 1)

    def test_closed_seam_is_rounded(self):
        # Square; the seam vertex at 0+0j must be rounded like the rest.
        path = polygon(0 + 0j, 4 + 0j, 4 + 4j, 0 + 4j)
        original = Path(*path)
        rounded = round_corners(path, 1.0)

        self.assertEqual(len(arcs_of(rounded)), 4)
        self.assertTrue(rounded.isclosed())
        assert_smooth_joins(self, rounded, closed=True)
        for arc in arcs_of(rounded):
            self.assertAlmostEqual(arc.radius.real, 1.0)
        # No sharp corner remains at the seam: the closest approach to
        # 0+0j is the seam arc's midpoint, at r*(sqrt(2) - 1) for a
        # right angle rounded with r = 1.
        min_dist = min(abs(rounded.point(t) - 0j)
                       for t in np.linspace(0, 1, 2001))
        self.assertAlmostEqual(min_dist, np.sqrt(2) - 1, places=3)
        self.assertEqual(path, original)

    def test_zero_radius_keeps_corner(self):
        path = polyline(0 + 0j, 2 + 0j, 2 + 2j, 0 + 2j)
        rounded = round_corners(path, [0, 0.5, 0, 0])
        self.assertEqual(len(arcs_of(rounded)), 1)
        # The sharp corner at 0+2j survives as a line-to-line joint.
        joints = [seg.end for seg in rounded]
        self.assertTrue(any(abs(p - (0 + 2j)) < 1e-9 for p in joints))

    def test_auto_scale_uniform_shrink(self):
        # Square with side 2; radius 2 needs tangent length 2 at each end
        # of every edge -> common factor 2 / (2 + 2) = 0.5.
        path = polygon(0 + 0j, 2 + 0j, 2 + 2j, 0 + 2j)
        rounded, scale = round_corners(path, 2.0, auto_scale=True,
                                       return_scale=True)
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(len(arcs_of(rounded)), 4)
        for arc in arcs_of(rounded):
            self.assertAlmostEqual(arc.radius.real, 1.0)
        assert_smooth_joins(self, rounded, closed=True)
        # Each edge is fully consumed: neighbouring arcs meet exactly.
        self.assertTrue(all(isinstance(seg, Arc) for seg in rounded))

    def test_auto_scale_not_needed_keeps_radii(self):
        path = polygon(0 + 0j, 4 + 0j, 4 + 4j, 0 + 4j)
        rounded, scale = round_corners(path, 1.0, auto_scale=True,
                                       return_scale=True)
        self.assertAlmostEqual(scale, 1.0)
        for arc in arcs_of(rounded):
            self.assertAlmostEqual(arc.radius.real, 1.0)

    def test_oversized_radius_raises_without_auto_scale(self):
        path = polygon(0 + 0j, 2 + 0j, 2 + 2j, 0 + 2j)
        with self.assertRaises(CornerRoundingError) as ctx:
            round_corners(path, 2.0)
        self.assertIn('auto_scale=True', str(ctx.exception))

    def test_duplicate_points_are_stable(self):
        path = polyline(0 + 0j, 2 + 0j, 2 + 0j, 2 + 0j, 2 + 2j)
        rounded = round_corners(path, [0, 0.5, 0.5, 0.5, 0])
        self.assertEqual([type(s) for s in rounded], [Line, Arc, Line])
        arc = arcs_of(rounded)[0]
        self.assertAlmostEqual(arc.radius.real, 0.5)
        self.assertAlmostEqual(arc.start, 1.5 + 0j)
        self.assertAlmostEqual(arc.end, 2 + 0.5j)

        # Duplicates at the closed seam collapse too.
        ring = Path(Line(0 + 0j, 4 + 0j), Line(4 + 0j, 4 + 4j),
                    Line(4 + 4j, 0 + 4j), Line(0 + 4j, 0 + 0j),
                    Line(0 + 0j, 0 + 0j))
        rounded_ring = round_corners(ring, [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(len(arcs_of(rounded_ring)), 4)
        assert_smooth_joins(self, rounded_ring, closed=True)

    def test_collinear_vertex_radius_ignored(self):
        path = polyline(0 + 0j, 2 + 0j, 4 + 0j, 4 + 2j)
        rounded = round_corners(path, [0, 1.0, 0.5, 0])
        # Only the genuine corner at 4+0j gets an arc; the straight
        # vertex at 2+0j is passed through without error.
        self.assertEqual(len(arcs_of(rounded)), 1)
        self.assertEqual(rounded.start, 0 + 0j)
        self.assertEqual(rounded.end, 4 + 2j)
        assert_smooth_joins(self, rounded, closed=False)

    def test_uturn_raises(self):
        path = polyline(0 + 0j, 2 + 0j, 1 + 0j)
        with self.assertRaises(CornerRoundingError) as ctx:
            round_corners(path, [0, 0.5, 0])
        self.assertIn('U-turn', str(ctx.exception))

        # A zero radius at the U-turn keeps the sharp reversal instead.
        rounded = round_corners(path, [0, 0.0, 0])
        self.assertEqual(len(arcs_of(rounded)), 0)
        self.assertEqual(rounded.start, 0 + 0j)
        self.assertEqual(rounded.end, 1 + 0j)

    def test_scalar_and_endpoint_radii(self):
        # Scalar radius applies to every (interior) vertex.
        path = polyline(0 + 0j, 2 + 0j, 2 + 2j)
        rounded = round_corners(path, 0.5)
        self.assertEqual(len(arcs_of(rounded)), 1)
        # Non-zero endpoint radii are ignored for open paths.
        rounded = round_corners(path, [9.0, 0.5, 9.0])
        self.assertEqual(rounded.start, 0 + 0j)
        self.assertEqual(rounded.end, 2 + 2j)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            round_corners(Path(Line(0j, 1j), QuadraticBezier(1j, 2j, 3j)),
                          0.5)
        with self.assertRaises(ValueError):
            round_corners(polyline(0j, 1 + 0j, 1 + 1j), [0.5, 0.5])
        with self.assertRaises(ValueError):
            round_corners(polyline(0j, 1 + 0j, 1 + 1j), -0.5)

    def test_svg_exportable(self):
        # The rounded path converts to a genuine SVG path string with
        # real arc commands (no display styling involved).
        path = polyline(0 + 0j, 2 + 0j, 2 + 2j)
        d = round_corners(path, 0.5).d()
        self.assertIn('A', d)
        self.assertIn('L', d)


if __name__ == '__main__':
    unittest.main()
