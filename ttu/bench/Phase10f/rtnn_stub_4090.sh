#!/bin/bash
# RTNN full against stub on the 4090 (session 7 part B, predictions model/predictions_4090_batch7.md). OptiX only. Plain runs (search compute, KERNEL event) and ncu (optixLaunch kernel duration), interleaved, 3 repetitions. Raw to notes/rtnn_stub4090 (noclobber, timestamped).
set -C; source ~/cuda128.sh; export LD_LIBRARY_PATH=$HOME/opt/nvoptix-580.178.04:$LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TTU_GPU=1 RTNN_DRAM_GB=11.5958 TMPDIR=$HOME/ncutmp
D=$HOME/thirdparty; O=$HOME/eoss/notes/rtnn_stub4090; mkdir -p $O; ts=$(date +%Y%m%dT%H%M%S); S=$HOME/eoss/ttu/bench/Phase10f/ncu_4090_batch6.sh
idle() { [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU 1 busy, stop"; exit 3; }; }
for i in 1 2 3; do for r in 0.05 0.025; do for v in full stub; do
  [ $v = full ] && d=$D/div1/rtnn/src/build/bin || d=$D/div1_stub/rtnn/src/build/bin
  idle; (cd $d && timeout -k 2 120 ./optixNSearch -f $D/div1/data/u100000.csv -r $r -k 50 -p 0 -s 0 -m 0 -d 0 > $O/plain_${v}_${r}_${ts}_$i.txt 2>&1); echo "plain $v $r $i rc $? $(grep -E 'search compute|KERNEL' $O/plain_${v}_${r}_${ts}_$i.txt | tr '\n' ' ')"
  for w in $(seq 30); do [ "$(nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -lt 10 ] && break; sleep 1; done
  (cd $d && $S rtnn_${v}_$r -- --kernel-name regex:optixLaunch ./optixNSearch -f $D/div1/data/u100000.csv -r $r -k 50 -p 0 -s 0 -m 0 -d 0) 
done; done; done
