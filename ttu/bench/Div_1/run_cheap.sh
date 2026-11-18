#!/bin/bash
# Phase 10c: RTNN with the lane probe in four modes, interleaved, REPS rounds. OptiX only (no TTU code). usage: run_cheap.sh REPS   (raw to notes/div1c/raw, noclobber)
set -C; export LD_LIBRARY_PATH=$HOME/opt/nvoptix-580.178.04:$LD_LIBRARY_PATH; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 RTNN_DRAM_GB=11.5958; source ~/cuda128.sh
O=$HOME/eoss/notes/div1c/raw; mkdir -p $O; ts=$(date +%Y%m%dT%H%M%S); D=$HOME/thirdparty/div1
idle() { [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU 1 busy, stop"; exit 3; }; }
for i in $(seq $1); do for r in 0.05 0.025; do for mode in plain full sample7 notime; do idle
  b=$O/rtnn_${r}_${mode}_${ts}_$i; unset LHOUT LHSAMPLE LHNOTIME
  case $mode in full) export LHOUT=$b;; sample7) export LHOUT=$b LHSAMPLE=7;; notime) export LHOUT=$b LHNOTIME=1;; esac
  (cd $D/rtnn/src/build/bin && timeout -k 2 300 ./optixNSearch -f $D/data/u100000.csv -r $r -k 50 -p 0 -s 0 -m 0 -d 0 > $b.txt 2>&1); echo "$r $mode $i rc $? $(grep KERNEL $b.txt | head -3 | tr '\n' ' ')"
done; done; done
