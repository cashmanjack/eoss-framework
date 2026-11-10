# BVH_5b: local query starts for short rays

Needs checked stage 0 and stage 1 `setup3` binaries built with `--clock --smfilter`. Analysis needs numpy.

    /usr/bin/python3 bvh5b.py prep BIN_S0 WORK soup:100000 0.02      # rays, start paths, CPU reference
    python bvh5b.py run BIN_S1 WORK
    /usr/bin/python3 bvh5b.py report WORK

A ray starts at the deepest node reached by descending while exactly one child box touches its segment
(and that child is internal), so no other subtree can hold a hit. GAS dumps are NVIDIA data: keep them out of git.
