#!/bin/bash
# model v1 on the RTX 5090 (pc14): OptiX only (no TTU code of ours). Predictions: model/predictions_5090_batch5.md part B (commit a6923fd, before any run).
# Adapted from ttu/bench/Phase10f/run_item3.sh (paths only). usage: run_workloads_pc14.sh REPS   raw to notes/pc14_workloads (noclobber, timestamped).
# Builds: /scratch/tgrogers-pc14-disk01/cashman3/thirdparty/div1_sm120 (sm_120 rebuilds from notes/rebuild recipes, outside git). Every run: pc14_gpucheck.sh first, timeout -k 2 60, stop at the first failure.
set -C; P=/scratch/tgrogers-pc14-disk01/cashman3; source $P/cuda128.sh; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 RTNN_DRAM_GB=11.5958
E=$P/eoss; O=$E/notes/pc14_workloads; mkdir -p $O; ts=$(date +%Y%m%dT%H%M%S); D=$P/thirdparty/div1_sm120; DT=$D/data
run() { # name dir command...
  local n=$1 d=$2; shift 2; bash $E/notes/pc14_gpucheck.sh >/dev/null || { echo "GPU busy, stop"; exit 3; }; local b=$O/${n}_${ts}_$i
  nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm --format=csv,noheader -lms 50 > $b.clk & local sp=$!
  (cd $d && timeout -k 2 60 "$@" > $b.txt 2>&1); local rc=$?; kill $sp; wait $sp 2>/dev/null
  echo "$n rep $i rc $rc $(grep -E 'search compute|LAUNCH|KERNEL|RND|Total' $b.txt | head -8 | tr '\n' ' ' | cut -c1-300)"; [ $rc -eq 0 ] || exit 4
}
for i in $(seq $1); do
  run rtnn_r05 $D/rtnn/src/build/bin ./optixNSearch -f $DT/u100000.csv -r 0.05 -k 50 -p 0 -s 0 -m 0 -d 0
  run rtnn_r025 $D/rtnn/src/build/bin ./optixNSearch -f $DT/u100000.csv -r 0.025 -k 50 -p 0 -s 0 -m 0 -d 0
  run knn_orig $D/knn_orig ./knn_host $DT/u200000.csv $DT/u200000.csv 16 0.004
  run knn_reg $D/knn_reg ./knn_host $DT/u200000.csv $DT/u200000.csv 16 0.004
  run ark_100kq $D/ark/build_fixed env DEVOUT=1 ./sample01-knn $DT/ark2e5.txt 100000 100000 0.005
  run ark_10kq $D/ark/build_fixed env DEVOUT=1 ./sample01-knn $DT/ark_a1e5.txt 100000 10000 0.005
  run tk400k $D/tk/ark/b ./sample03-trueknn $DT/u400000.csv 400000 3 0.004 16 $O/tk_${ts}_$i.tk.txt
done
