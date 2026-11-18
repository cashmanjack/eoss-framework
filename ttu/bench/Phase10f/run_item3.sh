#!/bin/bash
# Phase 10f item 3 (predictions model/predictions_phase10f.md, committed first): model v1 on the 4090. OptiX only (no TTU code), GPU 1, idle check before every run, SM clock sampled (nvidia-smi, 50 ms) during every run.
# usage: run_item3.sh REPS    raw to notes/p10f/raw (noclobber, timestamped). Needs /dev/shm/knnr/{orig,reg}/{knn_host,knn.ptx}, u2e5.csv, ark2e5.txt (see notes log) and ~/thirdparty/div1 builds.
set -C; source ~/cuda128.sh; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$HOME/opt/nvoptix-580.178.04:$LD_LIBRARY_PATH RTNN_DRAM_GB=11.5958
O=$HOME/eoss/notes/p10f/raw; mkdir -p $O; ts=$(date +%Y%m%dT%H%M%S); D=$HOME/thirdparty/div1; K=/dev/shm/knnr
idle() { [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU 1 busy, stop"; exit 3; }; }
run() { # name dir command...
  local n=$1 d=$2; shift 2; idle; local b=$O/${n}_${ts}_$i
  nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm --format=csv,noheader -lms 50 > $b.clk & local sp=$!
  (cd $d && timeout -k 2 300 "$@" > $b.txt 2>&1); local rc=$?; kill $sp; wait $sp 2>/dev/null
  echo "$n rep $i rc $rc $(grep -E 'search compute|LAUNCH|KERNEL|RND|Total' $b.txt | head -8 | tr '\n' ' ' | cut -c1-300)"; [ $rc -eq 0 ] || exit 4
}
for i in $(seq $1); do
  run rtnn_r05 $D/rtnn/src/build/bin ./optixNSearch -f $D/data/u100000.csv -r 0.05 -k 50 -p 0 -s 0 -m 0 -d 0
  run rtnn_r025 $D/rtnn/src/build/bin ./optixNSearch -f $D/data/u100000.csv -r 0.025 -k 50 -p 0 -s 0 -m 0 -d 0
  run knn_orig $K/orig ./knn_host $K/u2e5.csv $K/u2e5.csv 16 0.004
  run knn_reg $K/reg ./knn_host $K/u2e5.csv $K/u2e5.csv 16 0.004
  run ark_100kq $D/ark/build_fixed env DEVOUT=1 ./sample01-knn $K/ark2e5.txt 100000 100000 0.005
  run ark_10kq $D/ark/build_fixed env DEVOUT=1 ./sample01-knn $D/data/ark_a1e5.txt 100000 10000 0.005
  run tk400k $D/tk/ark/b ./sample03-trueknn $D/data/u400000.csv 400000 3 0.004 16 $O/tk_${ts}_$i.tk.txt
done
