# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from copy import deepcopy
from math import pi

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path, polygon, dashed_path
)

TOL = 1e-4


class TestDashedPath(unittest.TestCase):

    def test_line_dashes(self):
        path = Path(Line(0j, 100 + 0j))
        runs = dashed_path(path, [10, 5])
        self.assertEqual(len(runs), 7)
        for i, run in enumerate(runs):
            self.assertIsInstance(run, Path)
            self.assertTrue(run.iscontinuous())
            self.assertAlmostEqual(run.length(), 10, delta=TOL)
            self.assertTrue(all(isinstance(seg, Line) for seg in run))
            # position and direction preserved
            self.assertAlmostEqual(run.start, (15 * i) + 0j, delta=TOL)
            self.assertAlmostEqual(run.end, (15 * i + 10) + 0j, delta=TOL)

    def test_odd_dash_array_is_repeated(self):
        path = Path(Line(0j, 30 + 0j))
        runs = dashed_path(path, [10])
        self.assertEqual(len(runs), 2)
        self.assertAlmostEqual(runs[0].start, 0j, delta=TOL)
        self.assertAlmostEqual(runs[0].end, 10 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[1].start, 20 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[1].end, 30 + 0j, delta=TOL)

    def test_offset(self):
        path = Path(Line(0j, 30 + 0j))
        # positive offset shifts the pattern start backwards
        runs = dashed_path(path, [10, 5], offset=5)
        self.assertEqual([round(run.length(), 6) for run in runs],
                         [5, 10, 5])
        self.assertAlmostEqual(runs[0].start, 0j, delta=TOL)
        # negative offset
        runs = dashed_path(path, [10, 5], offset=-5)
        self.assertEqual(len(runs), 2)
        self.assertAlmostEqual(runs[0].start, 5 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[0].end, 15 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[1].start, 20 + 0j, delta=TOL)
        # offsets differing by a whole period are equivalent
        runs_a = dashed_path(path, [10, 5], offset=7)
        runs_b = dashed_path(path, [10, 5], offset=7 + 3 * 15)
        runs_c = dashed_path(path, [10, 5], offset=7 - 2 * 15)
        self.assertEqual(len(runs_a), len(runs_b))
        self.assertEqual(len(runs_a), len(runs_c))
        for a, b, c in zip(runs_a, runs_b, runs_c):
            self.assertAlmostEqual(a.start, b.start, delta=TOL)
            self.assertAlmostEqual(a.start, c.start, delta=TOL)
            self.assertAlmostEqual(a.end, b.end, delta=TOL)
            self.assertAlmostEqual(a.end, c.end, delta=TOL)

    def test_segment_types_and_direction_preserved(self):
        cubic = CubicBezier(0j, 50 + 80j, 100 - 80j, 150 + 0j)
        quad = QuadraticBezier(0j, 75 + 100j, 150 + 0j)
        arc = Arc(0j, radius=100 + 50j, rotation=30, large_arc=1,
                  sweep=1, end=150 + 0j)
        for seg in (cubic, quad, arc):
            path = Path(seg)
            runs = dashed_path(path, [20, 10])
            self.assertGreater(len(runs), 1)
            for run in runs:
                self.assertTrue(run.iscontinuous())
                self.assertTrue(
                    all(isinstance(s, type(seg)) for s in run))
            # first run starts at the path start, direction preserved
            self.assertAlmostEqual(runs[0].start, path.start, delta=TOL)
            # dash and gap lengths along the curve
            for run in runs[:-1]:
                self.assertAlmostEqual(run.length(), 20, delta=1e-2)
            visible = sum(run.length() for run in runs)
            # the gaps between consecutive runs are exactly 10 long; a
            # trailing partial gap may remain at the end of the path
            trailing = path.length() - visible - 10 * (len(runs) - 1)
            self.assertGreaterEqual(trailing, -1e-6)
            self.assertLess(trailing, 10)

    def test_closed_seam_joined(self):
        # square of side 10, perimeter 40
        path = polygon(0j, 10 + 0j, 10 + 10j, 10j)
        self.assertTrue(path.isclosed())
        runs = dashed_path(path, [8, 4])
        # intervals (0,8), (12,20), (24,32), (36,40); the dash crossing
        # the seam joins the last and first intervals into one run
        self.assertEqual(len(runs), 3)
        joined = runs[0]
        self.assertTrue(joined.iscontinuous())
        self.assertAlmostEqual(joined.length(), 12, delta=TOL)
        self.assertAlmostEqual(joined.start, 4j, delta=TOL)
        self.assertAlmostEqual(joined.end, 8 + 0j, delta=TOL)
        total = sum(run.length() for run in runs)
        self.assertAlmostEqual(total, 28, delta=TOL)
        # when the seam falls inside a gap, no joining happens
        runs = dashed_path(path, [6, 4])
        self.assertEqual(len(runs), 4)
        for run in runs:
            self.assertAlmostEqual(run.length(), 6, delta=TOL)

    def test_closed_curve_seam_joined(self):
        circle = Path(
            Arc(1 + 0j, radius=1 + 1j, rotation=0, large_arc=1, sweep=1,
                end=-1 + 0j),
            Arc(-1 + 0j, radius=1 + 1j, rotation=0, large_arc=1, sweep=1,
                end=1 + 0j))
        self.assertTrue(circle.isclosed())
        runs = dashed_path(circle, [4, 1])
        # circumference 2*pi < 5, so the single dash crosses the seam
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertTrue(run.iscontinuous())
        self.assertTrue(all(isinstance(seg, Arc) for seg in run))
        self.assertAlmostEqual(run.length(), 4 + (2 * pi - 5), delta=1e-2)

    def test_subpaths_restart_pattern(self):
        path = Path(Line(0j, 10 + 0j), Line(20 + 0j, 30 + 0j))
        runs = dashed_path(path, [4, 2])
        self.assertEqual(len(runs), 4)
        self.assertAlmostEqual(runs[0].start, 0j, delta=TOL)
        self.assertAlmostEqual(runs[0].end, 4 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[1].start, 6 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[2].start, 20 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[2].end, 24 + 0j, delta=TOL)
        self.assertAlmostEqual(runs[3].start, 26 + 0j, delta=TOL)

    def test_zero_entries(self):
        path = Path(Line(0j, 12 + 0j))
        # zero-length gap: solid stroke
        runs = dashed_path(path, [5, 0])
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].length(), 12, delta=TOL)
        # zero-length dash: nothing visible
        runs = dashed_path(path, [0, 5])
        self.assertEqual(runs, [])

    def test_invalid_dash_arrays(self):
        path = Path(Line(0j, 10 + 0j))
        for bad in ([], [0], [0, 0, 0], [-1, 5], [5, -1], [0, -1], 5):
            with self.assertRaises(ValueError):
                dashed_path(path, bad)
            with self.assertRaises(ValueError):
                path.dashed(bad)

    def test_input_unmodified(self):
        path = Path(
            Line(0j, 10 + 0j),
            CubicBezier(10 + 0j, 20 + 10j, 30 - 10j, 40 + 0j),
            Arc(40 + 0j, radius=5 + 5j, rotation=0, large_arc=0, sweep=1,
                end=50 + 0j))
        path_copy = deepcopy(path)
        dash_array = [3, 2, 1, 4]
        dash_array_copy = list(dash_array)
        runs = dashed_path(path, dash_array, offset=1.5)
        self.assertGreater(len(runs), 0)
        self.assertEqual(path, path_copy)
        self.assertEqual(dash_array, dash_array_copy)

    def test_single_segment_and_method(self):
        seg = Line(0j, 10 + 0j)
        runs = dashed_path(seg, [4, 2])
        self.assertEqual(len(runs), 2)
        path = Path(seg)
        self.assertEqual([r.start for r in path.dashed([4, 2])],
                         [r.start for r in runs])
        # empty path
        self.assertEqual(dashed_path(Path(), [1, 1]), [])


if __name__ == '__main__':
    unittest.main()
