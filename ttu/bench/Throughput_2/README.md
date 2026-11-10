# Throughput_2: active lanes per warp, idle lanes masked vs riding along

Needs a checked stage 1 `setup3` binary built with `--clock --sync --smfilter` (the SM filter doubles as
the lane mask: `--active L --idle mask` gives idle lanes an SM id no SM has, so they exit before TTUOPEN).

    python run.py BIN_S1 WORK
    /usr/bin/python3 analyze.py WORK

Idle modes: mask (exit before the query), tmax0, tneg (tmax -1), empty (tmin 1 > tmax 0).
