"""Tools for expanding dashed strokes into visible geometry.

Laser cutters, plotters and similar devices only understand solid
geometry, so an SVG ``stroke-dasharray`` style must be converted into
the set of visible subpaths it represents.  This module provides
``dashed_path``, which cuts a path along its arc length according to an
alternating sequence of dash (drawn) and gap (skipped) lengths.
"""

# External dependencies
from __future__ import division, absolute_import, print_function

# Internal dependencies
from .path import Path, is_path_segment


def _validate_dashes(dashes):
    """Returns `dashes` as a list of floats with an even number of entries.

    Per the SVG specification, an odd number of entries is repeated to
    yield an even-length pattern.  Individual entries may be zero, but
    negative entries and all-zero patterns raise a ValueError.
    """
    if isinstance(dashes, (int, float)):
        dashes = [dashes]
    dashes = [float(d) for d in dashes]
    if not dashes:
        raise ValueError("dashes must contain at least one entry.")
    if any(d < 0 for d in dashes):
        raise ValueError("dash lengths must be non-negative, got %s." %
                         (dashes,))
    if len(dashes) % 2 == 1:
        dashes = dashes * 2
    if sum(dashes) <= 0:
        raise ValueError("at least one dash length must be positive.")
    return dashes


def _emit_pieces(segments, lengths, starts, s0, s1, pieces):
    """Appends the portions of `segments` between arc lengths s0 and s1
    (measured from the start of the subpath) to `pieces` as
    (segment, s_start, s_end) tuples."""
    for seg, seg_len, seg_start in zip(segments, lengths, starts):
        if seg_len <= 0:
            continue
        a = max(s0, seg_start)
        b = min(s1, seg_start + seg_len)
        if b <= a:
            continue
        a_local = a - seg_start
        b_local = b - seg_start
        t0 = 0.0 if a_local <= 0.0 else seg.ilength(a_local)
        t1 = 1.0 if b_local >= seg_len else seg.ilength(b_local)
        if t1 <= t0:
            continue
        pieces.append((seg.cropped(t0, t1), a, b))


def _dash_subpath(subpath, dashes, offset):
    """Returns the visible pieces of a single continuous subpath as a list
    of Paths.  `offset` must already be reduced modulo the pattern's
    period."""
    segments = list(subpath)
    lengths = [seg.length() for seg in segments]
    starts = []
    total = 0.0
    for seg_len in lengths:
        starts.append(total)
        total += seg_len
    if total <= 0:
        return []

    # Locate the pattern entry containing the start of the subpath.
    idx = 0
    off = offset
    while off >= dashes[idx]:
        off -= dashes[idx]
        idx += 1
    on = (idx % 2 == 0)
    remaining = dashes[idx] - off

    # Walk the subpath, emitting the "on" portions.  Every iteration either
    # advances the arc length position `s` or advances `idx`; since the
    # pattern contains a positive entry, zero-length entries cannot loop.
    pieces = []
    s = 0.0
    while s < total:
        step = min(remaining, total - s)
        if on and step > 0:
            _emit_pieces(segments, lengths, starts, s, s + step, pieces)
        s += step
        remaining -= step
        if remaining == 0:
            idx = (idx + 1) % len(dashes)
            remaining = dashes[idx]
            on = not on

    # Group contiguous pieces into Paths.  Consecutive pieces touch exactly
    # (same float) whenever no positive-length gap separates them.
    groups = []
    for piece, s0, s1 in pieces:
        if groups and groups[-1][2] == s0:
            groups[-1][0].append(piece)
            groups[-1][2] = s1
        else:
            groups.append([[piece], s0, s1])

    # On a closed subpath, ink that is continuous across the start/end seam
    # is a single visible stroke; join the two pieces into one Path.
    if (len(groups) >= 2 and subpath.start == subpath.end
            and groups[0][1] == 0.0 and groups[-1][2] == total):
        first_segs, _, _ = groups.pop(0)
        groups[-1][0].extend(first_segs)

    return [Path(*segs) for segs, _, _ in groups]


def dashed_path(path, dashes, offset=0.0):
    """Expands a dashed stroke into the visible geometry it represents.

    Args:
        path (Path or path segment): the path to dash.  Each discontinuous
            subpath restarts the dash pattern, as in SVG.
        dashes (sequence of floats): alternating dash (drawn) and gap
            (skipped) lengths, like SVG's ``stroke-dasharray``.  An odd
            number of entries is repeated once, per the SVG specification.
            Individual entries may be zero, but entries may not be negative
            and at least one entry must be positive.
        offset (float): distance into the dash pattern at which the first
            subpath starts, like SVG's ``stroke-dashoffset``.  May be
            negative or larger than the pattern's period.

    Returns:
        list of Path: the visible ("on") portions of `path`.  Each returned
        Path is continuous and its segments keep their original type (Line,
        QuadraticBezier, CubicBezier, Arc) and direction.  If a subpath is
        closed and a single dash crosses the start/end seam, the pieces on
        either side of the seam are joined into one Path.  `path` and
        `dashes` are not modified.

    Raises:
        ValueError: if `dashes` is empty, contains a negative entry, or
            contains only zeros.
    """
    dashes = _validate_dashes(dashes)
    period = sum(dashes)
    offset = float(offset) % period

    if is_path_segment(path):
        path = Path(path)

    visible = []
    for subpath in path.continuous_subpaths():
        visible.extend(_dash_subpath(subpath, dashes, offset))
    return visible
