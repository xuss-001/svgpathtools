# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from math import pi

# Internal dependencies
from svgpathtools import Path, Line, Arc, QuadraticBezier
from svgpathtools.fillet import rounded_path, FilletError


def assert_tangent(testcase, seg_a, seg_b):
    """seg_b should start where seg_a ends, heading in the same direction."""
    testcase.assertAlmostEqual(seg_a.end, seg_b.start)
    ta = seg_a.unit_tangent(1)
    tb = seg_b.unit_tangent(0)
    testcase.assertAlmostEqual(abs(ta - tb), 0)


class TestRoundedPath(unittest.TestCase):

    def test_single_corner_open(self):
        # 90-degree left turn, radius 2
        path = Path(Line(0, 10), Line(10, 10 + 10j))
        out = rounded_path(path, 2)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], Line)
        self.assertIsInstance(out[1], Arc)
        self.assertIsInstance(out[2], Line)

        # tangency with both neighboring lines
        assert_tangent(self, out[0], out[1])
        assert_tangent(self, out[1], out[2])

        # true circular arc of the requested radius
        self.assertAlmostEqual(out[1].radius.real, 2)
        self.assertAlmostEqual(out[1].radius.imag, 2)
        self.assertAlmostEqual(abs(out[1].center - out[1].start), 2)
        self.assertAlmostEqual(abs(out[1].center - out[1].end), 2)

        # tangent points sit at distance r from the vertex for a 90-deg turn
        self.assertAlmostEqual(out[0].end, 8)
        self.assertAlmostEqual(out[2].start, 10 + 2j)

        # open path endpoints unchanged
        self.assertEqual(out.start, path.start)
        self.assertEqual(out.end, path.end)

        # length: 10 + 10 - 2 - 2 + quarter circle of radius 2
        self.assertAlmostEqual(out.length(), 20 - 4 + pi)

    def test_input_path_not_modified(self):
        segs = [Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 10j)]
        path = Path(*segs)
        snapshot = [Line(s.start, s.end) for s in path]
        rounded_path(path, 2)
        self.assertEqual(list(path), snapshot)

    def test_closed_path_seam(self):
        # right triangle, closed; the seam at 0 is also a corner
        path = Path(Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 0))
        out = rounded_path(path, 1)
        arcs = [s for s in out if isinstance(s, Arc)]
        lines = [s for s in out if isinstance(s, Line)]
        self.assertEqual(len(arcs), 3)
        self.assertEqual(len(lines), 3)
        self.assertTrue(out.iscontinuous())
        self.assertTrue(out.isclosed())
        for i in range(len(out)):
            assert_tangent(self, out[i], out[(i + 1) % len(out)])
        for arc in arcs:
            self.assertAlmostEqual(arc.radius.real, 1)

    def test_concave_corner(self):
        # L-shaped (notched) closed path; vertex 2+2j is concave
        pts = [0, 4, 4 + 4j, 2 + 4j, 2 + 2j, 2j]
        path = Path(*[Line(pts[i], pts[(i + 1) % 6]) for i in range(6)])
        out = rounded_path(path, 0.5)
        arcs = [s for s in out if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 6)
        self.assertTrue(out.iscontinuous())
        self.assertTrue(out.isclosed())
        for i in range(len(out)):
            assert_tangent(self, out[i], out[(i + 1) % len(out)])
        for arc in arcs:
            self.assertAlmostEqual(arc.radius.real, 0.5)
        # the concave corner arcs the other way (sweep flag flipped)
        sweeps = [arc.sweep for arc in arcs]
        self.assertIn(True, sweeps)
        self.assertIn(False, sweeps)

    def test_zero_radius_keeps_corner(self):
        path = Path(Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 10j))
        out = rounded_path(path, [2, 0])
        self.assertEqual(len(out), 4)  # line, arc, line, line
        self.assertIsInstance(out[1], Arc)
        self.assertIsInstance(out[3], Line)
        # the zero-radius corner is untouched: lines meet at the vertex
        self.assertEqual(out[2].end, 10 + 10j)
        self.assertEqual(out[3].start, 10 + 10j)

    def test_per_vertex_radius_list_open(self):
        # one value per vertex including endpoints (endpoints ignored)
        path = Path(Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 10j))
        out = rounded_path(path, [99, 2, 0, 99])
        arcs = [s for s in out if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 1)
        self.assertAlmostEqual(arcs[0].radius.real, 2)

    def test_uniform_scaling(self):
        # 10 x 1 rectangle, radius 2 everywhere: the short edges (length 1)
        # must accommodate tangent lengths 2 + 2, so the common scale is 1/4
        path = Path(Line(0, 10), Line(10, 10 + 1j),
                    Line(10 + 1j, 1j), Line(1j, 0))
        out = rounded_path(path, 2)
        arcs = [s for s in out if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 4)
        for arc in arcs:
            self.assertAlmostEqual(arc.radius.real, 0.5)  # 2 * 1/4
        # short edges are fully consumed by the two scaled fillets
        short_lines = [s for s in out if isinstance(s, Line)
                       and abs(s.end - s.start) < 1e-8]
        self.assertEqual(len(short_lines), 2)
        # long edges keep 10 - 0.5 - 0.5 = 9 of line
        long_lines = [s for s in out if isinstance(s, Line)
                      and abs(s.end - s.start) > 1e-8]
        for line in long_lines:
            self.assertAlmostEqual(abs(line.end - line.start), 9)
        self.assertTrue(out.iscontinuous())
        self.assertTrue(out.isclosed())

    def test_no_scaling_raises(self):
        path = Path(Line(0, 10), Line(10, 10 + 1j),
                    Line(10 + 1j, 1j), Line(1j, 0))
        with self.assertRaises(FilletError):
            rounded_path(path, 2, scale_to_fit=False)
        # fits fine without scaling when the radius is small enough
        out = rounded_path(path, 0.5, scale_to_fit=False)
        self.assertEqual(len([s for s in out if isinstance(s, Arc)]), 4)

    def test_collinear_point_is_stable(self):
        # vertex at 1 is collinear (straight through); no arc needed there
        path = Path(Line(0, 1), Line(1, 5), Line(5, 5 + 3j))
        out = rounded_path(path, 1)
        arcs = [s for s in out if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 1)
        self.assertTrue(out.iscontinuous())
        self.assertEqual(out.start, 0)
        self.assertEqual(out.end, 5 + 3j)

    def test_duplicate_points_are_stable(self):
        path = Path(Line(0, 0), Line(0, 10), Line(10, 10),
                    Line(10, 10 + 10j))
        out = rounded_path(path, 2)
        arcs = [s for s in out if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 1)
        self.assertAlmostEqual(arcs[0].radius.real, 2)
        self.assertTrue(out.iscontinuous())
        self.assertEqual(out.start, 0)
        self.assertEqual(out.end, 10 + 10j)

    def test_uturn_raises(self):
        path = Path(Line(0, 10), Line(10, 2))
        with self.assertRaises(FilletError):
            rounded_path(path, 1)
        # radius 0 keeps the sharp reversal as-is
        out = rounded_path(path, 0)
        self.assertEqual(list(out), list(path))

    def test_non_line_path_raises(self):
        path = Path(QuadraticBezier(0, 1, 2), Line(2, 3))
        with self.assertRaises(TypeError):
            rounded_path(path, 1)

    def test_bad_radius_count_raises(self):
        path = Path(Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 10j))
        with self.assertRaises(ValueError):
            rounded_path(path, [1, 2, 3])
        with self.assertRaises(ValueError):
            rounded_path(path, -1)

    def test_direction_preserved(self):
        # rounded path must travel the same way as the original polyline
        path = Path(Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 5 + 10j))
        out = rounded_path(path, 2)
        self.assertGreater(out[0].unit_tangent(0).real, 0)
        self.assertAlmostEqual(out[0].unit_tangent(0).imag, 0)
        self.assertLess(out[-1].unit_tangent(1).real, 0)
        self.assertAlmostEqual(out[-1].unit_tangent(1).imag, 0)


if __name__ == '__main__':
    unittest.main()
