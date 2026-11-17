#!/bin/bash
# Rebuild /dev/shm/t9 (host, ptx, words) and /dev/shm/t9u (checked t1r builds, soup dumps, replay tables, ray files) for the Unit_3 / lane group / knee runs.
# Run from the repo root. Stage 0 launches (no TTU instruction) make the GAS dumps; needs an idle GPU (TTU_GPU). Words: the none case, P0 true path.
set -e; source ~/cuda128.sh; V=~/eoss/.venv/bin/python; S=ttu/harness/setup3
mkdir -p /dev/shm/t9 /dev/shm/t9u
(cd $S && nvcc -O2 -arch=sm_89 -I$HOME/optix_sdk/include -o /dev/shm/t9/setup3 setup3.cu -lcuda -lnvidia-ml 2>/dev/null && nvcc -O2 -arch=sm_89 -ptx -I$HOME/optix_sdk/include -o /dev/shm/t9/setup3_dev.ptx setup3_dev.cu 2>/dev/null)
echo "ff030207 5a400408 00ff0000 00000000 ff030407 5a700408 00ff0000 00000000" > /dev/shm/t9/words.txt; cp /dev/shm/t9/words.txt /dev/shm/t9/setup3_dev.ptx /dev/shm/t9u/
for st in 0 1; do $V $S/build_patch.py --stage $st --bin /dev/shm/t9/setup3 --out /dev/shm/t9u/t1r_s$st --clock --sync --remap --compact-out --regs 64 --max-out 61 | tail -1; $V $S/check_seq.py /dev/shm/t9u/t1r_s$st | tail -1; done
for n in 1000 10000 30000 100000 300000 1000000; do timeout -k 2 120 /usr/bin/python3 $S/run.py /dev/shm/t9u/t1r_s0 hit zero /dev/shm/t9u/s0_$n.json --scene soup:$n --warps 1 --repeat 1 --dump-gas /dev/shm/t9u/gas_$n.bin --dump-tris /dev/shm/t9u/tris_$n.bin | grep "^exit"; done
/usr/bin/python3 ttu/bench/Unit_3/prepare_groups.py /dev/shm/t9u | tail -1 | cut -c1-80
/usr/bin/python3 ttu/bench/Unit_3/prepare_class.py /dev/shm/t9u
