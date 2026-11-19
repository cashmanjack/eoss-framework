#!/bin/bash
# ncu on the OptiX workload builds (pc14, no TTU code of ours). Predictions: model/predictions_5090_batch6.md. usage: ncu_workloads_pc14.sh NAME METRICS_COMMA_LIST -- command...   (run in the program's directory)
# gpucheck first, timeout -k 2 240, raw csv to notes/pc14_workloads/ncu_NAME_<ts>.csv (noclobber).
set -C; P=/scratch/tgrogers-pc14-disk01/cashman3; source $P/cuda128.sh; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 RTNN_DRAM_GB=11.5958 TMPDIR=$P/tmp
n=$1; m=$2; shift 3; E=$P/eoss; bash $E/notes/pc14_gpucheck.sh >/dev/null || { echo "GPU busy, stop"; exit 3; }
out=$E/notes/pc14_workloads/ncu_${n}_$(date +%Y%m%dT%H%M%S).csv
timeout -k 2 240 ncu --csv --clock-control none --cache-control none --metrics $m "$@" > $out 2> $out.err; rc=$?; echo "$n rc $rc $out"; exit $rc
