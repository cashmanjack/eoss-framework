# Memory_4: rays grouped by subtree vs randomly interleaved

Needs checked stage 0 and stage 1 `setup3` binaries built with `--clock` (see `ttu/harness/setup3/README.md`).
Analysis needs numpy (system python3). WORK is a directory in /dev/shm.

    /usr/bin/python3 prep.py BIN_S0 WORK/s100000 soup:100000 8192      # also soup:3000000 -> WORK/s3000000
    # exact keys from the hardware: root run recording leaf blocks, then leafkeys.py, a check run with
    # per ray starts, fixkeys.py, regroup; bigset.py does all of it in one driver for its ray set
    python run.py BIN_S0 BIN_S1 WORK timing|ncu|b-timing|b-ncu
    python bigset.py BIN_S0 BIN_S1 WORK                                 # 65,536 rays, working set > L2
    /usr/bin/python3 summary.py WORK [OUT.json]

`analyze.py`: per finished ray costs of one timing run; `ncu_analyze.py`: RT core traffic per finished ray;
`startfiles.py`: per warp and per ray start nodes. GAS dumps are NVIDIA data: keep them out of git.
