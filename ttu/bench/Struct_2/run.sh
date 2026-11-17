#!/bin/bash
# Struct_2 runs (OptiX only). Raw notes/struct2/<tag>_<TS>.json (mode x). usage: run.sh WHAT   WHAT = build | inst | part | ncu
export LD_LIBRARY_PATH=$HOME/opt/nvoptix-580.178.04:$LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1; source ~/cuda128.sh
O=$HOME/eoss/notes/struct2; mkdir -p $O /dev/shm/s2b/nt; B=/dev/shm/s2b; ts() { date +%Y%m%dT%H%M%S; }
idle() { [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] || { echo busy; exit 3; }; }
run() { tag=$1; shift; idle; timeout -k 2 300 $B/s2 $B/s2_dev.ptx $O/${tag}_$(ts).json "$@" > /dev/null || { echo FAIL $tag; exit 1; }; }
case $1 in
 build) for p in 1 2 3; do for N in 100000 1000000; do for f in none trace build update compact; do run build_N${N}_${f}_p$p mode=build N=$N flags=$f reps=15; done; done; done ;;
 inst) for p in 1 2 3; do for M in 0 1 16 256 4096 65536; do run inst_M${M}_p$p mode=inst M=$M total=1000000; done; done ;;
 part) for p in 1 2 3; do for A in 32 16 8 1; do run part_A${A}_s0_p$p mode=part active=$A spread=0; [ $A != 32 ] && run part_A${A}_s1_p$p mode=part active=$A spread=1; done; done ;;
 ncu) for cfg in "inst M=0" "inst M=1" "inst M=16" "inst M=256" "inst M=4096" "inst M=65536"; do set -- $cfg; idle; tag=ncu_$1_$2; ts=$(ts); TMPDIR=$B/nt timeout 300 ncu --target-processes all --cache-control none --clock-control none -k regex:optixLaunch -s 1 -c 1 --metrics smsp__inst_executed.sum,smsp__thread_inst_executed.sum,l1tex__t_sectors.sum,gpu__time_duration.sum --csv --page raw --log-file $O/${tag}_$ts.csv $B/s2 $B/s2_dev.ptx $O/${tag}_$ts.json mode=$1 $2 reps=1 >/dev/null || echo FAIL $tag; done ;;
esac
