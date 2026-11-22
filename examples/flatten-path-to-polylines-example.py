"""Flatten an SVG Path to tolerance-bounded polylines.

Run from the repository root with:

    python examples/flatten-path-to-polylines-example.py
"""

from svgpathtools import flatten_path, parse_path


path = parse_path(
    'M 0 0 L 10 0 Q 20 10 30 0 C 40 -10 50 10 60 0 '
    'M 80 0 A 20 10 25 1 1 100 0'
)

# x' = 1000*x + 5, y' = 0.01*y - 5
transform = (1000, 0, 0, 0.01, 5, -5)
polylines = flatten_path(path, tolerance=1e-3, transform=transform)

for index, vertices in enumerate(polylines):
    print("subpath {}: {} vertices, closed={}".format(
        index, len(vertices), vertices[0] == vertices[-1]))
    print("  start:", vertices[0])
    print("  end:  ", vertices[-1])
