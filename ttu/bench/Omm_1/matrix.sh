#!/bin/bash
# Omm_1 matrix: 3 outer repeats in rotating order, then ncu. Raw in notes/omm1 (private).
export TTU_GPU=1; W=$HOME/eoss/notes/omm1; R=$HOME/eoss/ttu/bench/Omm_1/run.py
PFX=${PFX:-omm}; BIG=${BIG:-2097152}; NREP=${NREP:-3}
for rep in $(seq $NREP); do for N in 100000 1000000; do
  for cfg in A0 A D0 D B:2 B:4 B:6 B:8 C:2 C:4 C:6 C:8; do v=${cfg%%:*}; L=4; [ "$v" != "${cfg}" ] && L=${cfg##*:}
    /usr/bin/python3 $R $W ${PFX}_N${N}_${cfg/:/}_r$rep variant=$v level=$L N=$N big=$BIG reps=20 $([ $rep = 1 ] && echo dumpres=@RES) || exit 1
  done; done; done
[ -n "$NONCU" ] && exit 0
for N in 100000 1000000; do for cfg in A D B:4 B:6 C:6; do v=${cfg%%:*}; L=4; [ "$v" != "${cfg}" ] && L=${cfg##*:}
  source ~/cuda128.sh; /usr/bin/python3 $R $W ncu_N${N}_${cfg/:/} ncu variant=$v level=$L N=$N big=2097152 reps=1 count=0 || exit 1; done; done
