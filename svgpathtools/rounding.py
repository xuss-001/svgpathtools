"""Geometric corner rounding (filleting) for polylines.

This module provides :func:`round_corners`, which replaces the vertices
of a ``Path`` made of ``Line`` segments by real circular ``Arc``
segments that are tangent to the two adjacent edges.  The result is
genuine geometry (not an SVG display style such as ``stroke-linejoin``),
so the rounded path can be exported to SVG and used for arc-length
measurements.

Conventions and edge cases
--------------------------
* Both open and closed paths are supported.  For closed paths the seam
  vertex (where the last segment meets the first) is rounded like any
  other vertex.  Open path endpoints are never moved.
* A radius of zero keeps the original (sharp) corner.
* The turn angle at a vertex is always taken in ``(0, pi)``, so convex
  and concave (reflex) corners are rounded uniformly.
* Consecutive duplicate points are collapsed before rounding (a run of
  duplicates keeps the largest radius specified for the run).
* Vertices where the path continues straight (same-direction collinear
  edges) cannot be rounded; any radius specified there is ignored.
* Vertices where the path doubles back on itself (a 180 degree U-turn)
  cannot be rounded either; specifying a non-zero radius there raises
  :class:`CornerRoundingError`.
"""

from math import atan2, tan

import numpy as np

from .path import Path, Line, Arc


class CornerRoundingError(ValueError):
    """Raised when the requested corner rounding cannot be realized."""


def _vertex_turn(u_in, u_out, cos_tol=1e-12):
    """Classify the turn from unit vector ``u_in`` to unit vector ``u_out``.

    Returns ``(kind, cross, dot)`` where ``kind`` is one of ``'straight'``,
    ``'uturn'`` or ``'turn'``, ``cross`` is the z-component of
    ``u_in x u_out`` (positive for a left/CCW turn) and ``dot`` is the
    scalar product ``u_in . u_out``.
    """
    rot = u_in.conjugate() * u_out
    cross, dot = rot.imag, rot.real
    if dot >= 1.0 - cos_tol:
        return 'straight', cross, dot
    if dot <= -(1.0 - cos_tol):
        return 'uturn', cross, dot
    return 'turn', cross, dot


def _collapse_duplicates(points, radii, closed, tol):
    """Merge consecutive (and, for closed rings, wrap-around) duplicates."""
    pts, rads = [points[0]], [radii[0]]
    for point, radius in zip(points[1:], radii[1:]):
        if abs(point - pts[-1]) <= tol:
            rads[-1] = max(rads[-1], radius)
        else:
            pts.append(point)
            rads.append(radius)
    if closed:
        while len(pts) > 1 and abs(pts[0] - pts[-1]) <= tol:
            pts.pop()
            rads[0] = max(rads[0], rads.pop())
    return pts, rads


def round_corners(path, radii, auto_scale=False, closed=None, tol=1e-9,
                  return_scale=False):
    """Return a copy of ``path`` with its corners rounded by circular arcs.

    Parameters
    ----------
    path : Path
        A continuous path consisting only of ``Line`` segments.  It is
        not modified.
    radii : float or sequence of float
        Corner radius per vertex.  A single number applies the same
        radius to every vertex.  Otherwise one value per vertex is
        expected: ``len(path) + 1`` values for an open path (the two
        endpoint values are ignored, as endpoints are never rounded) or
        ``len(path)`` values for a closed path.  A radius of zero keeps
        the original corner.
    auto_scale : bool
        If True, all non-zero radii are uniformly multiplied by the
        largest factor in ``(0, 1]`` such that the tangent lengths of
        neighbouring corners fit on every edge.  If False (default), a
        :class:`CornerRoundingError` is raised when the requested radii
        do not fit simultaneously.
    closed : bool or None
        Whether to treat ``path`` as a closed ring.  None (default)
        auto-detects by comparing the path's start and end points.
    tol : float
        Absolute tolerance used to detect duplicate points and
        degenerate (zero-length) edges.
    return_scale : bool
        If True, return a ``(rounded_path, scale)`` tuple where
        ``scale`` is the uniform factor actually applied to the non-zero
        radii (1.0 unless ``auto_scale`` shrunk them).

    Returns
    -------
    Path or (Path, float)
        A new ``Path`` made of ``Line`` and ``Arc`` segments.  Every arc
        is tangent to its two neighbouring edges and the travel
        direction of the input is preserved.

    Raises
    ------
    TypeError
        If ``path`` contains a segment that is not a ``Line``.
    CornerRoundingError
        If a non-zero radius is requested at a U-turn vertex, or if the
        radii do not fit on the available edge lengths and
        ``auto_scale`` is False.
    """
    if not isinstance(path, Path):
        raise TypeError('path must be a svgpathtools Path, got %r'
                        % type(path).__name__)
    for seg in path:
        if not isinstance(seg, Line):
            raise TypeError('round_corners only accepts paths made of Line '
                            'segments, got %r' % type(seg).__name__)
    if len(path) == 0:
        return (Path(), 1.0) if return_scale else Path()

    if closed is None:
        closed = path[0].start == path[-1].end
    else:
        closed = bool(closed)

    points = [seg.start for seg in path]
    if not closed:
        points.append(path[-1].end)

    if np.isscalar(radii):
        radii = [float(radii)] * len(points)
    else:
        radii = [float(r) for r in radii]
    if len(radii) != len(points):
        raise ValueError('expected %d radii (one per vertex), got %d'
                         % (len(points), len(radii)))
    for radius in radii:
        if radius < 0:
            raise ValueError('corner radii must be non-negative, got %r'
                             % radius)

    points, radii = _collapse_duplicates(points, radii, closed, tol)
    n = len(points)

    if closed and n < 3:
        raise CornerRoundingError(
            'a closed path needs at least 3 distinct points to round, '
            'got %d' % n)
    if n < 3:
        # A single straight edge (or nothing) has no corners to round.
        result = Path(*[Line(points[i], points[i + 1])
                        for i in range(n - 1)])
        return (result, 1.0) if return_scale else result

    n_edges = n if closed else n - 1
    edge_vec = [points[(i + 1) % n] - points[i] for i in range(n_edges)]
    edge_len = [abs(v) for v in edge_vec]
    edge_dir = [v / length for v, length in zip(edge_vec, edge_len)]

    # Tangent length consumed by the fillet at each vertex (0 = sharp).
    tangent = [0.0] * n
    turn_cross = [0.0] * n
    first_vertex = 0 if closed else 1
    last_vertex = n if closed else n - 1
    for i in range(first_vertex, last_vertex):
        if radii[i] == 0:
            continue
        u_in = edge_dir[(i - 1) % n_edges]
        u_out = edge_dir[i % n_edges]
        kind, cross, dot = _vertex_turn(u_in, u_out)
        if kind == 'straight':
            # Same-direction collinear edges: nothing to round, the
            # requested radius is silently ignored.
            continue
        if kind == 'uturn':
            raise CornerRoundingError(
                'cannot round vertex %d at %r: the path doubles back on '
                'itself (180 degree U-turn), so no finite circle is '
                'tangent to both edges at this radius' % (i, points[i]))
        theta = atan2(abs(cross), dot)  # deflection angle in (0, pi)
        tangent[i] = radii[i] * tan(theta / 2.0)
        turn_cross[i] = cross

    # Fit the fillets on the available edge lengths.
    scale = 1.0
    for i in range(n_edges):
        needed = tangent[i] + tangent[(i + 1) % n]
        if needed == 0:
            continue
        if auto_scale:
            scale = min(scale, edge_len[i] / needed)
        elif needed > edge_len[i] + tol:
            raise CornerRoundingError(
                'corner radii do not fit on edge %d (from %r to %r): '
                'the two adjacent corners need %g of tangent length but '
                'the edge is only %g long; pass auto_scale=True to '
                'uniformly shrink all non-zero radii'
                % (i, points[i], points[(i + 1) % n], needed, edge_len[i]))
    if scale != 1.0:
        tangent = [t * scale for t in tangent]
        radii = [r * scale for r in radii]

    # Build the rounded path: alternating lines and tangent arcs.
    segments = []
    for i in range(n_edges):
        j = (i + 1) % n
        line_start = points[i] + tangent[i] * edge_dir[i]
        line_end = points[j] - tangent[j] * edge_dir[i]
        if abs(line_end - line_start) > tol:
            segments.append(Line(line_start, line_end))
        if tangent[j] > 0 and (closed or j < n - 1):
            radius = radii[j]
            segments.append(Arc(line_end, radius=radius + 1j * radius,
                                rotation=0, large_arc=False,
                                sweep=turn_cross[j] > 0,
                                end=points[j] + tangent[j] * edge_dir[j]))

    result = Path(*segments)
    return (result, scale) if return_scale else result
