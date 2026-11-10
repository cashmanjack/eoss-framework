# BVH_5: start traversal at an internal node

Needs a stage 1 `setup3` binary built with `--clock` that passed `check_seq.py` (see
`ttu/harness/setup3/README.md`). No new TTU sequence: only the 0x180 input word changes.

    python run.py /dev/shm/bvh5/setup3_s1c /dev/shm/bvh5/out
    /usr/bin/python3 analyze.py OUT.json OUT.gas OUT.tris [--root-ref]
    /usr/bin/python3 latency.py OUT.json ...

`gas.py` walks an OptiX SM89 GAS dump and decodes node boxes (format hypothesis in its docstring).
Analysis needs numpy (system python3). GAS dumps are NVIDIA data: keep them out of git.
