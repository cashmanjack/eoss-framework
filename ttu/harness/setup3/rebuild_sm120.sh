#!/bin/bash
# pc14 (RTX 5090) SM120 harness rebuild, PREPARE ONLY: builds the stage 0 and stage 1 Latency_1 binaries (clk, twice) into /dev/shm/t9sm120 and runs check_seq on each.
# Needs /dev/shm/t9sm120/setup3 and setup3_dev.ptx (nvcc -arch=sm_120 ... -lcuda -lnvidia-ml, see README.md) and words.txt (the computed store words, notes/protocol.md SM120).
# Stage 1 binaries hold NVIDIA code: /dev/shm only, never git, NEVER RUN without Cesar's OK and Jack's approval (run.py refuses them on pc14 anyway: PC14_STAGE0 is stage 0 only, and the driver is not validated for TTU).
cd /scratch/tgrogers-pc14-disk01/cashman3/eoss/ttu/harness/setup3
. /scratch/tgrogers-pc14-disk01/cashman3/cuda128.sh
. /scratch/tgrogers-pc14-disk01/cashman3/eoss/.venv/bin/activate
D=/dev/shm/t9sm120; DUMP=../../../notes/setup4_sm120/dump_sm120_fn-0x20000_len0x20000.bin
for spec in "clk_s0:0:--clock --regs 96 --max-out 93" "clk_s1:1:--clock --regs 96 --max-out 93" "twice_s0:0:--clock --twice --compact-out" "twice_s1:1:--clock --twice --compact-out"; do
  IFS=: read n st fl <<< "$spec"; python3 build_patch.py --sm 120 --stage $st --bin $D/setup3 --out $D/$n $fl 2>&1 | tail -1 | cut -c1-110; done
for n in clk_s0 clk_s1 twice_s0 twice_s1; do echo "== $n"; python3 check_seq.py $D/$n $DUMP 2>&1 | tail -${TAIL:-4}; done
