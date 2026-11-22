"""Tests for svgpathtools/morph.py (closed-outline morphing)."""

import unittest
from math import cos, pi, sin

import numpy as np

from svgpathtools import (Arc, CubicBezier, Line, Path, morph_closed_paths,
                          parse_path, polygon)
from svgpathtools.morph import (_best_cyclic_shift, _sample_closed_path,
                                _signed_area)
from svgpathtools.path import closest_point_in_path


def regular_polygon_path(num_sides, radius, center=0j, start_angle=0.0):
    """A closed regular polygon Path, counter-clockwise."""
    points = [center + radius * complex(cos(start_angle + 2 * pi * i / num_sides),
                                        sin(start_angle + 2 * pi * i / num_sides))
              for i in range(num_sides)]
    return polygon(*points)


def frame_vertices(frame):
    return np.array([seg.start for seg in frame], dtype=complex)


def max_distance_to_path(points, path):
    return max(closest_point_in_path(z, path)[0] for z in points)


def square_lines(start=0j, side=100.0):
    """Axis-aligned square of 4 Line segments, counter-clockwise (y up)."""
    corners = [start, start + side, start + side + side * 1j, start + side * 1j]
    return polygon(*corners)


def square_beziers_split():
    """The same square as square_lines(), but made of 8 cubic Bezier
    segments (two per edge, control points on the edges) and starting at
    the midpoint of the bottom edge."""
    side = 100.0
    corners = [0j, side + 0j, side + side * 1j, side * 1j]
    mids = [(corners[i] + corners[(i + 1) % 4]) / 2 for i in range(4)]
    # sequence of boundary points along the square, starting at mids[0]
    boundary = [mids[0], corners[1], mids[1], corners[2],
                mids[2], corners[3], mids[3], corners[0]]
    segments = []
    for i in range(8):
        a = boundary[i]
        b = boundary[(i + 1) % 8]
        segments.append(CubicBezier(a, a + (b - a) / 3,
                                    a + 2 * (b - a) / 3, b))
    return Path(*segments)


def circle_arcs(radius=50.0, center=0j, sweep=True, start=None):
    """A circle made of two Arc segments."""
    if start is None:
        start = center + radius
    opposite = 2 * center - start
    half1 = Arc(start, radius + radius * 1j, 0, False, sweep, opposite)
    half2 = Arc(opposite, radius + radius * 1j, 0, False, sweep, start)
    return Path(half1, half2)


class TestSamplingAndAlignment(unittest.TestCase):
    """Direct checks of the documented alignment rules."""

    def test_samples_are_uniformly_distributed_by_arclength(self):
        path = regular_polygon_path(5, 40.0)
        num_samples = 150
        samples = _sample_closed_path(path, num_samples)
        spacings = np.abs(np.roll(samples, -1) - samples)
        expected = path.length() / num_samples
        np.testing.assert_allclose(spacings, expected, rtol=1e-6, atol=1e-9)

    def test_signed_area_detects_orientation(self):
        ccw = _sample_closed_path(regular_polygon_path(6, 10.0), 60)
        self.assertGreater(_signed_area(ccw), 0)
        self.assertLess(_signed_area(ccw[::-1]), 0)

    def test_best_cyclic_shift_recovers_known_offset(self):
        rng = np.random.default_rng(42)
        reference = rng.random(200) + 1j * rng.random(200)
        for shift in (0, 1, 77, 199):
            target = np.roll(reference, shift)
            recovered = _best_cyclic_shift(reference, target)
            np.testing.assert_array_equal(np.roll(target, recovered),
                                          reference)


class TestMorphDifferentSegmentation(unittest.TestCase):
    """Morphing between the same shape stored with different segment
    counts, segment types and start points must neither rotate nor
    collapse: every intermediate frame is (to sampling precision) the
    original square."""

    def check_frame_is_square(self, frame, num_samples):
        square = square_lines()
        vertices = frame_vertices(frame)
        # every vertex lies on the square boundary
        self.assertLess(max_distance_to_path(vertices, square), 1e-6)
        # the frame does not collapse: it encloses the full area
        self.assertAlmostEqual(_signed_area(vertices), 100.0 ** 2, places=4)
        # corresponding points stay uniformly distributed along the outline
        edge_lengths = np.abs(np.roll(vertices, -1) - vertices)
        np.testing.assert_allclose(edge_lengths,
                                   square.length() / num_samples,
                                   rtol=1e-6, atol=1e-9)

    def test_line_square_vs_split_bezier_square(self):
        num_samples = 200
        square_a = square_lines()
        square_b = square_beziers_split()
        self.assertNotEqual(len(square_a), len(square_b))
        for t in (0.25, 0.5, 0.75):
            frame = morph_closed_paths(square_a, square_b, t,
                                       num_samples=num_samples)
            self.assertEqual(len(frame), num_samples)
            self.assertTrue(all(isinstance(seg, Line) for seg in frame))
            self.assertTrue(frame.iscontinuous() and frame.isclosed())
            self.check_frame_is_square(frame, num_samples)

    def test_start_point_offset_does_not_rotate_or_collapse(self):
        """The same morph performed with a different start point on the
        first outline must produce the same intermediate geometry."""
        num_samples = 160
        triangle = regular_polygon_path(3, 60.0, center=50 + 50j)
        square_a = square_lines()
        # same square, but the start point is moved one segment ahead
        square_b = Path(*list(square_a[1:]) + [square_a[0]])
        self.assertNotEqual(square_a.start, square_b.start)
        for t in (0.3, 0.5, 0.8):
            frame_a = morph_closed_paths(square_a, triangle, t,
                                         num_samples=num_samples)
            frame_b = morph_closed_paths(square_b, triangle, t,
                                         num_samples=num_samples)
            # identical enclosed area (no collapse, no rotation)
            self.assertAlmostEqual(_signed_area(frame_vertices(frame_a)),
                                   _signed_area(frame_vertices(frame_b)),
                                   places=6)
            # identical geometry: every vertex of each frame lies on the
            # other frame's outline
            self.assertLess(
                max_distance_to_path(frame_vertices(frame_a), frame_b), 1e-6)
            self.assertLess(
                max_distance_to_path(frame_vertices(frame_b), frame_a), 1e-6)


class TestMorphOppositeDirectionArcs(unittest.TestCase):
    """Two circles drawn in opposite directions (and with different start
    points) must morph through circles, not collapse through the interior."""

    def test_circle_frames_stay_circular(self):
        num_samples = 400
        circle_cw = circle_arcs(50.0, sweep=True)
        # opposite direction and a start point half a circle away
        reversed_circle = circle_cw.reversed()
        circle_ccw = Path(reversed_circle[1], reversed_circle[0])
        self.assertNotEqual(circle_cw.start, circle_ccw.start)
        self.assertGreater(
            _signed_area(_sample_closed_path(circle_cw, num_samples)), 0)
        self.assertLess(
            _signed_area(_sample_closed_path(circle_ccw, num_samples)), 0)

        # chord sag of a 400-gon approximating a radius-50 circle
        tol = 50.0 * (1 - np.cos(np.pi / num_samples)) + 1e-6
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            frame = morph_closed_paths(circle_cw, circle_ccw, t,
                                       num_samples=num_samples)
            vertices = frame_vertices(frame)
            radii = np.abs(vertices)
            self.assertLess(np.max(np.abs(radii - 50.0)), tol)
            self.assertAlmostEqual(_signed_area(vertices),
                                   np.pi * 50.0 ** 2, delta=1.0)


class TestMorphUnequalSides(unittest.TestCase):
    """Morphing a triangle into a pentagon (different segment counts and
    side lengths)."""

    def setUp(self):
        self.triangle = regular_polygon_path(3, 60.0, start_angle=pi / 2)
        self.pentagon = regular_polygon_path(5, 60.0, start_angle=pi / 2)

    def test_intermediate_frames_follow_alignment_rules(self):
        num_samples = 180
        area_tri = abs(_signed_area(_sample_closed_path(self.triangle,
                                                        num_samples)))
        area_pent = abs(_signed_area(_sample_closed_path(self.pentagon,
                                                         num_samples)))
        self.assertLess(area_tri, area_pent)
        previous_area = None
        for t in (0.2, 0.4, 0.6, 0.8):
            frame = morph_closed_paths(self.triangle, self.pentagon, t,
                                       num_samples=num_samples)
            # a single closed, continuous polyline of num_samples Lines
            self.assertEqual(len(frame), num_samples)
            self.assertTrue(all(isinstance(seg, Line) for seg in frame))
            self.assertTrue(frame.iscontinuous() and frame.isclosed())
            # orientation is preserved (no collapse through the interior)
            area = _signed_area(frame_vertices(frame))
            self.assertGreater(area, 0)
            # the enclosed area stays between the endpoint areas and grows
            # monotonically for these convex, aligned shapes
            self.assertGreater(area, area_tri - 1e-6)
            self.assertLess(area, area_pent + 1e-6)
            if previous_area is not None:
                self.assertGreater(area, previous_area)
            previous_area = area
            # the frame serializes back to a valid, closed SVG path
            reparsed = parse_path(frame.d())
            self.assertTrue(reparsed.iscontinuous() and reparsed.isclosed())
            self.assertIn('Z', frame.d(use_closed_attrib=True).upper())

    def test_mixed_segment_types(self):
        """Lines, Beziers and arcs may be mixed within one outline."""
        num_samples = 120
        mixed = Path(
            Line(0j, 100 + 0j),
            CubicBezier(100 + 0j, 120 + 20j, 120 + 80j, 100 + 100j),
            Arc(100 + 100j, 50 + 50j, 0, False, True, 0 + 100j),
            Line(0 + 100j, 0j))
        frame = morph_closed_paths(mixed, self.pentagon, 0.5,
                                   num_samples=num_samples)
        self.assertEqual(len(frame), num_samples)
        self.assertTrue(frame.iscontinuous() and frame.isclosed())
        self.assertGreater(_signed_area(frame_vertices(frame)), 0)


class TestMorphEndpoints(unittest.TestCase):
    """The ends of the morph reproduce the original outlines exactly
    (and therefore within any sampling precision)."""

    def check_endpoints(self, path1, path2, num_samples):
        frame0 = morph_closed_paths(path1, path2, 0, num_samples=num_samples)
        frame1 = morph_closed_paths(path1, path2, 1, num_samples=num_samples)
        self.assertEqual(frame0, path1)
        self.assertEqual(frame1, path2)
        # deep copies: mutating the results must not touch the inputs
        self.assertIsNot(frame0, path1)
        self.assertIsNot(frame1, path2)
        # geometric fidelity: every sampled point of the originals lies
        # exactly on the returned endpoint frames
        for original, frame in ((path1, frame0), (path2, frame1)):
            original_samples = _sample_closed_path(original, num_samples)
            frame_samples = _sample_closed_path(frame, num_samples)
            np.testing.assert_allclose(frame_samples, original_samples,
                                       rtol=0, atol=1e-9)

    def test_endpoints_different_segmentation(self):
        self.check_endpoints(square_lines(), square_beziers_split(), 200)

    def test_endpoints_opposite_direction_arcs(self):
        circle_cw = circle_arcs(50.0, sweep=True)
        circle_ccw = circle_cw.reversed()
        self.check_endpoints(circle_cw, circle_ccw, 400)

    def test_endpoints_unequal_sides(self):
        self.check_endpoints(regular_polygon_path(3, 60.0),
                             regular_polygon_path(5, 60.0), 180)


class TestMorphErrors(unittest.TestCase):

    def setUp(self):
        self.square = square_lines()
        self.triangle = regular_polygon_path(3, 60.0, center=50 + 50j)

    def test_progress_out_of_range(self):
        for t in (-0.1, 1.1, -1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                morph_closed_paths(self.square, self.triangle, t)

    def test_progress_not_a_number(self):
        for t in ('0.5', None, [0.5]):
            with self.assertRaises(TypeError):
                morph_closed_paths(self.square, self.triangle, t)

    def test_invalid_num_samples(self):
        with self.assertRaises(ValueError):
            morph_closed_paths(self.square, self.triangle, 0.5, num_samples=2)
        with self.assertRaises(TypeError):
            morph_closed_paths(self.square, self.triangle, 0.5,
                               num_samples=100.5)

    def test_unsupported_inputs(self):
        # not a Path
        with self.assertRaises(TypeError):
            morph_closed_paths(Line(0j, 1j), self.triangle, 0.5)
        # empty path
        with self.assertRaises(ValueError):
            morph_closed_paths(Path(), self.triangle, 0.5)
        # not closed
        open_path = Path(Line(0j, 1j), Line(1j, 2j))
        with self.assertRaises(ValueError):
            morph_closed_paths(open_path, self.triangle, 0.5)
        # not continuous
        broken = Path(Line(0j, 1j), Line(2j, 0j))
        with self.assertRaises(ValueError):
            morph_closed_paths(broken, self.triangle, 0.5)


if __name__ == '__main__':
    unittest.main()
