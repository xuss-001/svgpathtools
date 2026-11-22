import unittest

import numpy as np

from svgpathtools import (
    Arc, CubicBezier, FlatteningLimitError, Line, Path, QuadraticBezier,
    flatten_path, flatten_segment
)


def _affine_matrix(transform):
    if transform is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape == (3, 3):
        return matrix[:2, :]
    if matrix.shape == (6,):
        return np.array([[matrix[0], matrix[2], matrix[4]],
                         [matrix[1], matrix[3], matrix[5]]])
    return matrix


def _transform_point(point, matrix):
    return complex(
        matrix[0, 0]*point.real + matrix[0, 1]*point.imag + matrix[0, 2],
        matrix[1, 0]*point.real + matrix[1, 1]*point.imag + matrix[1, 2])


def _point_segment_distance(point, start, end):
    direction = end - start
    length_squared = abs(direction)**2
    if length_squared == 0:
        return abs(point - start)
    offset = point - start
    t = (offset.real*direction.real +
         offset.imag*direction.imag) / length_squared
    t = min(1.0, max(0.0, t))
    return abs(point - (start + t*direction))


def _actual_max_error(path, polylines, transform=None, samples=2001):
    """Independent dense-sample audit of curve-to-polyline distance."""
    matrix = _affine_matrix(transform)
    subpaths = path.continuous_subpaths()
    worst = 0.0
    for subpath, vertices in zip(subpaths, polylines):
        for segment in subpath:
            for t in np.linspace(0.0, 1.0, samples):
                point = _transform_point(segment.point(float(t)), matrix)
                distance = min(
                    _point_segment_distance(point, vertices[index],
                                            vertices[index + 1])
                    for index in range(len(vertices) - 1))
                worst = max(worst, distance)
    return worst


class FlattenPathTest(unittest.TestCase):

    def assert_flat(self, path, tolerance, transform=None, vertices=None):
        polylines = flatten_path(path, tolerance, transform)
        error = _actual_max_error(path, polylines, transform)
        self.assertLessEqual(error, tolerance*(1 + 1e-10))
        if vertices is not None:
            self.assertEqual(len(polylines[0]), vertices)
        return polylines, error

    def test_mixed_bezier_segments_and_affine_vertices(self):
        path = Path(
            Line(0, 10),
            QuadraticBezier(10, 20 + 15j, 30),
            CubicBezier(30, 40 + 20j, 50 - 20j, 60),
        )
        transform = np.array([[2.0, -0.5, 7.0],
                              [0.25, 3.0, -4.0],
                              [0.0, 0.0, 1.0]])
        polylines, error = self.assert_flat(path, 1e-3, transform)

        self.assertEqual(len(polylines), 1)
        matrix = _affine_matrix(transform)
        self.assertEqual(polylines[0][0], _transform_point(0, matrix))
        self.assertEqual(polylines[0][-1], _transform_point(60, matrix))
        self.assertLess(error, 1e-3)

        six_value = (2.0, 0.25, -0.5, 3.0, 7.0, -4.0)
        self.assertEqual(
            flatten_path(path, 1e-3, six_value),
            flatten_path(path, 1e-3, transform))

    def test_arc_with_shear_and_strong_stretch(self):
        arc = Arc(-100 + 20j, 120 + 35j, 37, large_arc=1, sweep=1,
                  end=100 - 40j)
        path = Path(arc)
        transform = np.array([[5.0, 2.0, 10.0],
                              [-3.0, 0.2, 7.0],
                              [0.0, 0.0, 1.0]])
        _, error = self.assert_flat(path, 1e-2, transform)
        self.assertLess(error, 0.004)

    def test_strong_anisotropic_scaling_uses_transformed_coordinates(self):
        path = Path(CubicBezier(0, 100 + 100j, -100 + 100j, 10j))
        transform = np.diag([1000.0, 0.01, 1.0])

        source_tolerance = flatten_path(path, 1.0)
        transformed = flatten_path(path, 0.5, transform)
        self.assertLessEqual(
            _actual_max_error(path, transformed, transform), 0.5)
        self.assertNotEqual(len(source_tolerance[0]), len(transformed[0]))

    def test_fold_and_cusp_are_certified(self):
        fold = Path(CubicBezier(0, 100, -100, 0))
        _, fold_error = self.assert_flat(fold, 1e-2)
        self.assertTrue(flatten_path(fold, 1e-2)[0][0] ==
                        flatten_path(fold, 1e-2)[0][-1])

        cusp = Path(CubicBezier(0, 1 + 1j, -1 + 1j, 0))
        _, cusp_error = self.assert_flat(cusp, 1e-4)
        self.assertLess(cusp_error, 1e-4)
        self.assertLess(fold_error, 1e-2)

    def test_disconnected_and_closed_subpaths_are_preserved(self):
        closed = [
            Line(0, 10),
            Line(10, 10 + 10j),
            Line(10 + 10j, 0),
        ]
        path = Path(*closed, Line(20, 30), Line(30, 40))
        polylines = flatten_path(path, 0.1)

        self.assertEqual(len(polylines), 2)
        self.assertEqual(polylines[0][0], 0)
        self.assertEqual(polylines[0][-1], 0)
        self.assertEqual(polylines[1][0], 20)
        self.assertEqual(polylines[1][-1], 40)
        self.assertEqual(polylines[0], [0, 10, 10 + 10j, 0])

    def test_zero_length_segment_is_stable_and_keeps_endpoint(self):
        point = 5 + 5j
        path = Path(Line(point, point), Line(point, 6 + 5j))
        polylines = flatten_path(path, 0.1)

        self.assertEqual(polylines, [[point, point, 6 + 5j]])

    def test_segment_api(self):
        segment = QuadraticBezier(0, 1 + 2j, 2)
        points = flatten_segment(segment, tolerance=1e-3)
        self.assertEqual(points[0], 0)
        self.assertEqual(points[-1], 2)
        self.assertGreater(len(points), 2)

    def test_path_methods_and_tolerance_aliases(self):
        path = Path(Line(0, 1), QuadraticBezier(1, 2 + 1j, 3))
        expected = flatten_path(path, tolerance=1e-2)
        self.assertEqual(path.flatten(1e-2), expected)
        self.assertEqual(path.to_polyline(1e-2), expected)
        self.assertEqual(path.to_polylines(1e-2), expected)
        self.assertEqual(flatten_path(path, error=1e-2), expected)
        self.assertEqual(flatten_path(path, tol=1e-2), expected)
        self.assertEqual(flatten_path(path, max_error=1e-2), expected)

    def test_invalid_tolerances_raise(self):
        path = Path(Line(0, 1))
        for tolerance in (0, -1, float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                flatten_path(path, tolerance)
        for tolerance in ('bad', 1 + 2j):
            with self.assertRaises(TypeError):
                flatten_path(path, tolerance)

    def test_resource_limits_fail_explicitly(self):
        path = Path(CubicBezier(0, 100, -100, 0))
        with self.assertRaises(FlatteningLimitError) as context:
            flatten_path(path, 1e-12, max_depth=1)
        self.assertIn('max_depth', str(context.exception))
        self.assertIn('tolerance', str(context.exception))

        with self.assertRaises(FlatteningLimitError) as context:
            flatten_path(path, 1e-12, max_segments=1)
        self.assertIn('max_segments', str(context.exception))


if __name__ == '__main__':
    unittest.main()
