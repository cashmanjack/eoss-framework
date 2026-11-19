#!/bin/bash
# RTNN real against stub build (callback calibration, model v1 variant S) on the RTX 5090 (pc14). OptiX only. Predictions: model/predictions_5090_batch7.md part 3 (commit 2362bd4).
# usage: run_rtnn_calib_pc14.sh REPS    raw to notes/pc14_workloads (noclobber, timestamped); gpucheck before every run, timeout -k 2 60, stop at the first failure.
set -C; P=/scratch/tgrogers-pc14-disk01/cashman3; source $P/cuda128.sh; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 RTNN_DRAM_GB=11.5958
E=$P/eoss; O=$E/notes/pc14_workloads; ts=$(date +%Y%m%dT%H%M%S); D=$P/thirdparty/div1_sm120
for i in $(seq $1); do for r in 0.05 0.025; do for k in real stub; do
  d=$D/rtnn/src/build/bin; [ $k = stub ] && d=$D/rtnn_stub/src/build/bin
  bash $E/notes/pc14_gpucheck.sh >/dev/null || { echo "GPU busy, stop"; exit 3; }; b=$O/calib_${k}_r${r}_${ts}_$i
  (cd $d && timeout -k 2 60 ./optixNSearch -f $D/data/u100000.csv -r $r -k 50 -p 0 -s 0 -m 0 -d 0 > $b.txt 2>&1); rc=$?
  echo "$k r $r rep $i rc $rc $(grep -E 'KERNEL|search compute' $b.txt | tr '\n' ' ')"; [ $rc -eq 0 ] || exit 4
done; done; done
