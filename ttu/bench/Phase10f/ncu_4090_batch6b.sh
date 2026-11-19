#!/bin/bash
# ncu on the OptiX only builds (no TTU code of ours), RTX 4090 tethys GPU 1. Predictions: model/predictions_4090_batch6.md (commit 25789e1, before any run).
# usage: ncu_4090_batch6.sh NAME -- command...   (run in the program's directory). Raw csv to notes/ncu4090/ncu_NAME_<ts>.csv (noclobber); SM clock sampled beside it.
set -C; source ~/cuda128.sh; export LD_LIBRARY_PATH=$HOME/opt/nvoptix-580.178.04:$LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TTU_GPU=1 RTNN_DRAM_GB=11.5958 TMPDIR=$HOME/ncutmp
M=gpu__time_duration.sum,lts__t_bytes.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_requests_srcunit_tex_op_read.sum,lts__t_requests_srcunit_tex_op_write.sum,l1tex__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_lookup_hit.sum,lts__t_sectors_srcunit_tex_lookup_miss.sum
n=$1; shift 2; ts=$(date +%Y%m%dT%H%M%S); out=$HOME/eoss/notes/ncu4090/ncu2_${n}_$ts.csv
[ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] && [ "$(nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -lt 10 ] || { echo "GPU 1 busy, stop"; exit 3; }
nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm --format=csv,noheader -lms 50 > ${out%.csv}.clk & sp=$!
timeout -k 2 240 ncu --csv --clock-control none --cache-control none --metrics $M "$@" > $out 2> $out.err; rc=$?; kill $sp; wait $sp 2>/dev/null
echo "$n rc $rc $out"; exit $rc
