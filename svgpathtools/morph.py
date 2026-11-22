"""Morphing (interpolation) between closed outlines.

This module provides :func:`morph_closed_paths`, which interpolates between
two single, continuous, non-self-intersecting closed ``Path`` objects and
returns an intermediate outline that can be serialized back to SVG.

Alignment rules
---------------
Because the two source outlines may have different segment counts, different
start points, and opposite drawing directions, naive segment-by-segment
interpolation would twist and collapse.  The following rules are applied to
establish point correspondences before interpolating:

1. **Uniform arc-length sampling.**  Each outline is sampled at
   ``num_samples`` points equally spaced by arc length (measured along the
   outline's own drawing direction, starting from its start point).  This
   makes the correspondence independent of how each outline is segmented
   (lines, Beziers, and elliptical arcs may be mixed freely).

2. **Orientation normalization.**  The signed area of each sampled polygon
   is computed.  If the two outlines wind in opposite directions, the
   sample sequence of the second outline is reversed so both sequences
   traverse their shapes in the same sense.  This prevents the morph from
   collapsing through the interior.

3. **Start-point (phase) alignment.**  A cyclic shift of the second sample
   sequence is chosen to minimize the total squared distance between
   corresponding samples (a least-squares alignment, computed exactly via
   FFT cross-correlation).  This makes the result invariant to where each
   outline's start point happens to lie: the same shape with a different
   start point morphs identically, without spurious rotation.

4. **Interpolation.**  Corresponding samples are interpolated linearly and
   the result is returned as a closed ``Path`` of ``Line`` segments, which
   approximates both source outlines to within the sampling precision.

5. **Endpoint fidelity.**  At ``t == 0`` and ``t == 1`` the original paths
   are returned (as deep copies), so the exact source geometry -- including
   all curve and arc segments -- is preserved at the ends of the morph.
"""

from copy import deepcopy

import numpy as np

from .path import Path, Line, QuadraticBezier, CubicBezier, Arc

__all__ = ['morph_closed_paths']

_SUPPORTED_SEGMENT_TYPES = (Line, QuadraticBezier, CubicBezier, Arc)


def _validate_path(path, name):
    """Raises a clear error if ``path`` is not a supported closed outline."""
    if not isinstance(path, Path):
        raise TypeError(
            "{} must be a Path, got {}.".format(name, type(path).__name__))
    if len(path) == 0:
        raise ValueError("{} must contain at least one segment.".format(name))
    for seg in path:
        if not isinstance(seg, _SUPPORTED_SEGMENT_TYPES):
            raise ValueError(
                "{} contains an unsupported segment type: {}.  Only Line, "
                "QuadraticBezier, CubicBezier and Arc segments are "
                "supported.".format(name, type(seg).__name__))
    if not path.iscontinuous():
        raise ValueError(
            "{} must be a single continuous path (every segment must start "
            "where the previous one ended).".format(name))
    if not path.isclosed():
        raise ValueError("{} must be closed (its end point must coincide "
                         "with its start point).".format(name))


def _sample_closed_path(path, num_samples):
    """Returns ``num_samples`` points on ``path`` equally spaced by arc
    length, as a complex numpy array.  The first sample is the path's start
    point and the samples follow the path's drawing direction."""
    # Path.point(pos) parameterizes pos in [0, 1] by fraction of total arc
    # length, so uniform pos values give arc-length-uniform samples.
    ts = np.linspace(0.0, 1.0, num_samples, endpoint=False)
    return np.array([path.point(t) for t in ts], dtype=complex)


def _signed_area(points):
    """Signed area of the polygon through ``points`` (complex array).

    Positive for counter-clockwise winding, negative for clockwise."""
    x = points.real
    y = points.imag
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _best_cyclic_shift(reference, target):
    """Returns the cyclic shift ``k`` minimizing
    ``sum_i |reference[i] - target[(i - k) % n]|**2``, i.e. the shift to
    apply with ``np.roll(target, k)`` to align ``target`` to ``reference``.

    Minimizing that sum is equivalent to maximizing the cross-correlation
    ``sum_i Re(reference[i] * conj(target[(i - k) % n]))``, which is
    computed exactly for all shifts in O(n log n) via FFT."""
    n = len(reference)
    # c[k] = sum_i reference[i] * conj(target[(i - k) % n]), which is
    # exactly the score for aligning target with np.roll(target, k).
    correlation = np.fft.ifft(np.fft.fft(reference)
                              * np.conj(np.fft.fft(target)))
    return int(np.argmax(correlation.real))


def morph_closed_paths(path1, path2, t, num_samples=100):
    """Interpolates between two closed outlines.

    Parameters
    ----------
    path1, path2 : Path
        Single, continuous, non-self-intersecting closed paths.  Segments
        may be any mixture of ``Line``, ``QuadraticBezier``, ``CubicBezier``
        and ``Arc``.  The two paths may have different segment counts,
        different start points and opposite drawing directions; see the
        module docstring for the alignment rules used to establish point
        correspondences.
    t : float
        Interpolation progress, 0 <= t <= 1.  ``t == 0`` returns (a deep
        copy of) ``path1`` and ``t == 1`` returns ``path2``, preserving the
        exact source geometry at the endpoints.
    num_samples : int
        Sampling precision: the number of arc-length-uniform sample points
        used per outline (and hence the number of ``Line`` segments in
        intermediate frames).  Must be at least 3.

    Returns
    -------
    Path
        The interpolated closed outline.  For ``0 < t < 1`` this is a
        closed polygon of ``num_samples`` ``Line`` segments that can be
        serialized with ``Path.d()`` or ``wsvg()`` like any other path.

    Raises
    ------
    TypeError
        If ``t`` is not a real number, ``num_samples`` is not an integer,
        or either outline is not a ``Path``.
    ValueError
        If ``t`` lies outside [0, 1], ``num_samples`` < 3, or an outline is
        empty, discontinuous, not closed, contains unsupported segment
        types, or is degenerate (zero signed area).
    """
    if isinstance(t, bool) or not isinstance(t, (int, float, np.floating,
                                                 np.integer)):
        raise TypeError("t must be a real number, got {}."
                        .format(type(t).__name__))
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must satisfy 0 <= t <= 1, got {}.".format(t))
    if isinstance(num_samples, bool) or not isinstance(num_samples,
                                                       (int, np.integer)):
        raise TypeError("num_samples must be an integer, got {}."
                        .format(type(num_samples).__name__))
    if num_samples < 3:
        raise ValueError("num_samples must be at least 3, got {}."
                         .format(num_samples))
    _validate_path(path1, 'path1')
    _validate_path(path2, 'path2')

    if t == 0:
        return deepcopy(path1)
    if t == 1:
        return deepcopy(path2)

    samples1 = _sample_closed_path(path1, num_samples)
    samples2 = _sample_closed_path(path2, num_samples)

    area1 = _signed_area(samples1)
    area2 = _signed_area(samples2)
    if area1 == 0 or area2 == 0:
        raise ValueError("Degenerate outline with zero signed area; the "
                         "paths must enclose a non-zero area.")
    # Orientation normalization: make both sequences wind the same way.
    if np.sign(area1) != np.sign(area2):
        samples2 = samples2[::-1]

    # Start-point alignment: cyclically shift samples2 to minimize the
    # total squared distance between corresponding samples.
    shift = _best_cyclic_shift(samples1, samples2)
    samples2 = np.roll(samples2, shift)

    points = (1.0 - t) * samples1 + t * samples2
    segments = [Line(points[i], points[(i + 1) % num_samples])
                for i in range(num_samples)]
    return Path(*segments)
