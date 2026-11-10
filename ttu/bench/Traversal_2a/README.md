# Traversal_2a: does a query return without a hit?

Needs a checked stage 1 `setup3` binary built with `--clock`. Scenes `par:N:s` (stacked) and `par:N:o`
(overlapping): N triangles parallel to a +z ray at x = y = 0, 1e-3 off it, so the ray visits every box
and hits nothing.

    python run.py BIN_S1 WORK
    /usr/bin/python3 run.py --report WORK
