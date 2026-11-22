"""This submodule contains functions for geometrically rounding (filleting)
the corners of paths made of straight Line segments.  Unlike display-time
SVG styling, the rounded corners are realized as real Arc segments that are
tangent to the neighboring Line segments, so the result can be exported to
SVG and measured (e.g. with ``Path.length()``) directly."""

# External Dependencies
from __future__ import division, absolute_import, print_function
from math import atan2, tan

# Internal Dependencies
from .path import Path, Line, Arc


class FilletError(ValueError):
    """Raised when a requested corner rounding cannot be realized."""


def _unit(vector):
    return vector / abs(vector)


def _as_radius_list(radius, n_corners, n_vertices, closed):
    """Normalize the ``radius`` argument to a list of n_corners floats."""
    try:
        r = float(radius)
    except TypeError:
        radii = [float(r) for r in radius]
        if len(radii) == n_corners:
            return radii
        if not closed and len(radii) == n_vertices:
            # one value per vertex; the two endpoint values are unused
            return radii[1:-1]
        raise ValueError(
            "radius must be a scalar or have one value per corner (got %d "
            "values for %d corners)." % (len(radii), n_corners))
    if r < 0:
        raise ValueError("radius must be non-negative, got %s." % r)
    return [r] * n_corners


def rounded_path(path, radius, scale_to_fit=True, tol=1e-9):
    """Round the corners of a path of Line segments with circular arcs.

    Parameters
    ----------
    path : Path
        A continuous path composed exclusively of ``Line`` segments.  It may
        be open or closed; for a closed path the seam between the last and
        first segment is treated as a corner like any other.
    radius : float or sequence of floats
        The fillet radius for each corner.  A single scalar applies the same
        radius to every corner.  A sequence may give one value per corner
        (for an open path: one per interior vertex; for a closed path: one
        per vertex), or, for an open path, one value per vertex including
        the two endpoints (whose values are ignored).  A radius of zero
        keeps the original sharp corner.
    scale_to_fit : bool
        If True (default), whenever neighboring fillets would together
        consume more than the length of the segment between them, all
        non-zero radii are shrunk by a single common factor -- the largest
        factor with which every fillet fits.  If False, a ``FilletError``
        is raised instead when the requested radii cannot all be
        accommodated.
    tol : float
        Tolerance used to detect zero-length segments, straight
        (collinear) joints and 180-degree reversals.

    Returns
    -------
    Path
        A new ``Path`` made of ``Line`` and ``Arc`` segments.  Every arc is
        tangent to the two line segments it connects and preserves the
        direction of travel.  The input path is not modified, and the
        endpoints of an open path are left unchanged.

    Notes
    -----
    Zero-length segments (consecutive duplicate points) are dropped before
    rounding, and joints where the path continues straight (collinear
    segments) need no fillet and are kept as they are.  A 180-degree
    reversal (the path doubling back on itself) cannot be rounded by an
    arc tangent to both sides, so requesting a non-zero radius there
    raises a ``FilletError``.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a Path, got %r." % type(path))
    for seg in path:
        if not isinstance(seg, Line):
            raise TypeError(
                "rounded_path only supports paths made of Line segments; "
                "found %r." % type(seg))
    if len(path) == 0:
        raise ValueError("cannot round an empty path.")

    # Drop zero-length segments (consecutive duplicate points).
    segs = [Line(seg.start, seg.end) for seg in path
            if abs(seg.end - seg.start) > tol]
    if len(segs) == 0:
        raise ValueError("path has no segments of positive length.")

    for i in range(len(segs) - 1):
        if abs(segs[i].end - segs[i + 1].start) > tol:
            raise ValueError(
                "path is not continuous between segments %d and %d."
                % (i, i + 1))
    closed = abs(segs[-1].end - segs[0].start) <= tol

    n = len(segs)
    dirs = [_unit(seg.end - seg.start) for seg in segs]
    lengths = [abs(seg.end - seg.start) for seg in segs]

    if closed:
        n_corners = n
        # corner j sits at segs[j].start, between segs[j-1] and segs[j]
        corner_vertex = [segs[j].start for j in range(n)]
        incoming = [(j - 1) % n for j in range(n)]
        outgoing = list(range(n))
    else:
        n_corners = n - 1
        # corner j sits at segs[j].end == segs[j+1].start
        corner_vertex = [segs[j].end for j in range(n - 1)]
        incoming = list(range(n - 1))
        outgoing = list(range(1, n))

    radii = _as_radius_list(radius, n_corners, n + 1, closed)
    for r in radii:
        if r < 0:
            raise ValueError("radius must be non-negative, got %s." % r)

    # Per corner: turning angle, tangent length along each side, sweep.
    tangent_len = [0.0] * n_corners  # distance from vertex to tangent point
    sweep = [True] * n_corners
    filleted = [False] * n_corners
    for j in range(n_corners):
        r = radii[j]
        if r == 0:
            continue
        u = dirs[incoming[j]]
        w = dirs[outgoing[j]]
        cross = (u.conjugate() * w).imag
        dot = u.real * w.real + u.imag * w.imag
        if abs(cross) <= tol:
            if dot > 0:
                # collinear, same direction: no corner to round
                continue
            raise FilletError(
                "cannot round corner %d at %s: the path reverses direction "
                "there (180-degree turn), so no arc tangent to both sides "
                "exists.  Use radius 0 to keep the sharp corner."
                % (j, corner_vertex[j]))
        phi = atan2(abs(cross), dot)  # turning (deflection) angle in (0, pi)
        tangent_len[j] = r * tan(phi / 2.0)
        sweep[j] = cross > 0
        filleted[j] = True

    # Compute how much each segment must give up at its start and end.
    start_take = [0.0] * n
    end_take = [0.0] * n
    for j in range(n_corners):
        if not filleted[j]:
            continue
        end_take[incoming[j]] += tangent_len[j]
        start_take[outgoing[j]] += tangent_len[j]

    # Largest common scale factor with which every fillet fits.
    scale = 1.0
    for j in range(n):
        need = start_take[j] + end_take[j]
        if need > 0 and lengths[j] / need < scale:
            scale = lengths[j] / need
    if 1.0 - scale <= 1e-9:
        # exact fit (up to floating point round-off)
        scale = 1.0
    if scale < 1.0 and not scale_to_fit:
        bad = [j for j in range(n)
               if start_take[j] + end_take[j]
               > lengths[j] + max(tol, 1e-9 * lengths[j])]
        raise FilletError(
            "the requested radii do not fit: segment(s) %s are too short "
            "for the fillets at their ends (a common scale factor of %.6g "
            "would be required).  Pass scale_to_fit=True to shrink all "
            "non-zero radii uniformly instead."
            % (bad, scale))
    if scale < 1.0:
        tangent_len = [d * scale for d in tangent_len]
        radii = [r * scale for r in radii]
        start_take = [d * scale for d in start_take]
        end_take = [d * scale for d in end_take]

    # Build the trimmed lines.
    lines = []
    for j in range(n):
        a = segs[j].start + dirs[j] * start_take[j]
        b = segs[j].end - dirs[j] * end_take[j]
        lines.append(Line(a, b))

    # Build the arcs and interleave them with the lines.
    def corner_arc(j):
        if not filleted[j]:
            return None
        r = radii[j]
        t1 = corner_vertex[j] - dirs[incoming[j]] * tangent_len[j]
        t2 = corner_vertex[j] + dirs[outgoing[j]] * tangent_len[j]
        return Arc(t1, r + 1j * r, 0, False, sweep[j], t2,
                   autoscale_radius=False)

    out = []
    if closed:
        for j in range(n):
            out.append(lines[j])
            arc = corner_arc((j + 1) % n)
            if arc is not None:
                out.append(arc)
    else:
        out.append(lines[0])
        for j in range(n_corners):
            arc = corner_arc(j)
            if arc is not None:
                out.append(arc)
            out.append(lines[j + 1])
    return Path(*out)
