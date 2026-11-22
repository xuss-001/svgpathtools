# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from math import pi

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path, dashed_path,
)

TOL = 1e-6


def expected_intervals(length, dash, gap, offset=0.0):
    """Arc-length intervals covered by a simple two-entry dash pattern."""
    period = dash + gap
    off = offset % period
    intervals = []
    s = -off
    while s < length:
        a = max(s, 0.0)
        b = min(s + dash, length)
        if b > a:
            intervals.append((a, b))
        s += period
    return intervals


class TestDashedPath(unittest.TestCase):

    def check_segment_dashing(self, seg, dash, gap, offset=0.0):
        """Checks the pieces of a dashed single-segment path against the
        arc-length positions dictated by the pattern."""
        length = seg.length()
        pieces = dashed_path(Path(seg), [dash, gap], offset=offset)
        intervals = expected_intervals(length, dash, gap, offset)

        flat = [s for piece in pieces for s in piece]
        self.assertEqual(len(flat), len(intervals))
        for piece_seg, (a, b) in zip(flat, intervals):
            # original segment type is preserved
            self.assertEqual(type(piece_seg), type(seg))
            # visible ("on") length matches the pattern
            self.assertAlmostEqual(piece_seg.length(), b - a, delta=TOL)
            # pieces sit at the right arc-length positions and follow the
            # original direction of travel
            self.assertAlmostEqual(piece_seg.start,
                                   seg.point(seg.ilength(a)), delta=TOL)
            self.assertAlmostEqual(piece_seg.end,
                                   seg.point(seg.ilength(b)), delta=TOL)
        # each piece is a continuous Path
        for piece in pieces:
            self.assertTrue(piece.iscontinuous())
        return pieces

    def test_line(self):
        self.check_segment_dashing(Line(0, 10 + 0j), 2.0, 1.0)

    def test_line_exact_positions(self):
        pieces = dashed_path(Path(Line(0, 10 + 0j)), [2, 1])
        self.assertEqual(len(pieces), 4)
        for piece, (x0, x1) in zip(pieces, [(0, 2), (3, 5), (6, 8), (9, 10)]):
            self.assertEqual(len(piece), 1)
            self.assertAlmostEqual(piece[0].start, x0 + 0j)
            self.assertAlmostEqual(piece[0].end, x1 + 0j)

    def test_direction_preserved(self):
        pieces = dashed_path(Path(Line(10 + 0j, 0)), [4, 4])
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            self.assertGreater(piece.start.real, piece.end.real)
        self.assertAlmostEqual(pieces[0].start, 10 + 0j)
        self.assertAlmostEqual(pieces[0].end, 6 + 0j)

    def test_quadratic_bezier(self):
        self.check_segment_dashing(QuadraticBezier(0, 1 + 1j, 2 + 0j),
                                   0.3, 0.2)

    def test_cubic_bezier(self):
        self.check_segment_dashing(CubicBezier(0, 0, 1 + 1j, 1 + 0j),
                                   0.25, 0.15)

    def test_arc(self):
        # quarter circle of radius 1, length pi/2
        arc = Arc(1 + 0j, 1 + 1j, 0, False, True, 0 + 1j)
        self.assertAlmostEqual(arc.length(), pi / 2, delta=TOL)
        self.check_segment_dashing(arc, 0.5, 0.25)

    def test_multi_segment_curve(self):
        # a dash pattern crossing segment boundaries stays one Path
        path = Path(Line(0, 1 + 0j), CubicBezier(1 + 0j, 2 + 0j, 3 + 1j,
                                                 4 + 0j))
        pieces = dashed_path(path, [10, 1])
        self.assertEqual(len(pieces), 1)
        self.assertTrue(pieces[0].iscontinuous())
        self.assertEqual(type(pieces[0][0]), Line)
        self.assertEqual(type(pieces[0][1]), CubicBezier)
        self.assertAlmostEqual(pieces[0].length(), path.length(), delta=TOL)

    def test_odd_dash_array_repeats(self):
        line = Path(Line(0, 24 + 0j))
        odd = dashed_path(line, [2, 1, 3])
        even = dashed_path(line, [2, 1, 3, 2, 1, 3])
        self.assertEqual(len(odd), len(even))
        for p_odd, p_even in zip(odd, even):
            self.assertEqual(len(p_odd), len(p_even))
            for s_odd, s_even in zip(p_odd, p_even):
                self.assertAlmostEqual(s_odd.start, s_even.start)
                self.assertAlmostEqual(s_odd.end, s_even.end)

    def test_offset(self):
        line = Path(Line(0, 20 + 0j))
        pieces = dashed_path(line, [3, 2], offset=4)
        # offset 4 lands 1 unit before the end of the first gap
        self.assertAlmostEqual(pieces[0].start, 1 + 0j)
        self.assertAlmostEqual(pieces[0].end, 4 + 0j)

    def test_offset_equivalences(self):
        line = Path(Line(0, 50 + 0j))
        base = dashed_path(line, [3, 2], offset=4)
        for offset in (-1, -6, 9, 104):
            pieces = dashed_path(line, [3, 2], offset=offset)
            self.assertEqual(len(base), len(pieces))
            for p_base, p_other in zip(base, pieces):
                self.assertAlmostEqual(p_base.start, p_other.start)
                self.assertAlmostEqual(p_base.end, p_other.end)

    def test_zero_entries(self):
        line = Path(Line(0, 12 + 0j))
        # zero-length gaps join neighbouring dashes into one visible path
        pieces = dashed_path(line, [5, 0])
        self.assertEqual(len(pieces), 1)
        self.assertAlmostEqual(pieces[0].length(), 12, delta=TOL)
        self.assertTrue(pieces[0].iscontinuous())
        # zero-length dashes draw nothing
        self.assertEqual(dashed_path(line, [0, 5]), [])

    def test_invalid_dashes(self):
        line = Path(Line(0, 1 + 0j))
        for bad in ([], [0, 0], [0], [2, -1], [-0.5]):
            self.assertRaises(ValueError, dashed_path, line, bad)

    def test_discontinuous_subpaths_restart(self):
        path = Path(Line(0, 10 + 0j), Line(100 + 0j, 110 + 0j))
        pieces = dashed_path(path, [4, 6])
        self.assertEqual(len(pieces), 2)
        self.assertAlmostEqual(pieces[0].start, 0 + 0j)
        self.assertAlmostEqual(pieces[0].end, 4 + 0j)
        # the second subpath restarts the pattern at its own start
        self.assertAlmostEqual(pieces[1].start, 100 + 0j)
        self.assertAlmostEqual(pieces[1].end, 104 + 0j)

    def test_closed_seam_joined(self):
        square = Path(Line(0, 10 + 0j), Line(10 + 0j, 10 + 10j),
                      Line(10 + 10j, 10j), Line(10j, 0j))
        self.assertTrue(square.isclosed())
        # perimeter 40, pattern period 10, offset 2: the dash starting at
        # arc length 38 wraps across the seam and ends at arc length 4
        pieces = dashed_path(square, [6, 4], offset=2)
        self.assertEqual(len(pieces), 4)
        for piece in pieces:
            self.assertAlmostEqual(piece.length(), 6, delta=TOL)
            self.assertTrue(piece.iscontinuous())
        seam_crossers = [p for p in pieces
                         if any(p[k].end == 0j for k in range(len(p) - 1))]
        self.assertEqual(len(seam_crossers), 1)
        seam = seam_crossers[0]
        self.assertAlmostEqual(seam[0].start, 2j, delta=TOL)
        self.assertAlmostEqual(seam[0].end, 0j, delta=TOL)
        self.assertAlmostEqual(seam[1].start, 0j, delta=TOL)
        self.assertAlmostEqual(seam[1].end, 4 + 0j, delta=TOL)

    def test_closed_no_seam_join(self):
        square = Path(Line(0, 10 + 0j), Line(10 + 0j, 10 + 10j),
                      Line(10 + 10j, 10j), Line(10j, 0j))
        pieces = dashed_path(square, [6, 4])
        self.assertEqual(len(pieces), 4)
        for piece in pieces:
            self.assertEqual(len(piece), 1)
            self.assertAlmostEqual(piece.length(), 6, delta=TOL)

    def test_inputs_not_modified(self):
        path = Path(Line(0, 5 + 0j),
                    CubicBezier(5 + 0j, 6 + 1j, 7 + 1j, 8 + 0j))
        dashes = [2, 1, 3]
        d_before = path.d()
        dashes_before = list(dashes)
        dashed_path(path, dashes, offset=1.5)
        self.assertEqual(path.d(), d_before)
        self.assertEqual(dashes, dashes_before)

    def test_path_method_and_segment_input(self):
        line = Line(0, 10 + 0j)
        from_method = Path(line).dashed([4, 6])
        from_function = dashed_path(line, [4, 6])
        self.assertEqual(len(from_method), 1)
        self.assertEqual(len(from_function), 1)
        self.assertAlmostEqual(from_method[0].start, from_function[0].start)
        self.assertAlmostEqual(from_method[0].end, from_function[0].end)


if __name__ == '__main__':
    unittest.main()
