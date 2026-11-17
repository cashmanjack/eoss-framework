#!/bin/bash
# Div_1 runs (tethys GPU 1, OptiX only, no TTU code). Lane histograms: LHOUT=<prefix> (files created with mode x, never overwritten).
# usage: run_div.sh WORKLOAD REPS MODE   WORKLOAD mochi|knn|rtnn_r05|rtnn_r025|ark_r0112|ark_r005|tk100k|tk400k   MODE probe|plain
export LD_LIBRARY_PATH=$HOME/opt/nvoptix-580.178.04:$LD_LIBRARY_PATH; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 RTNN_DRAM_GB=11.5958; source ~/cuda128.sh
O=$HOME/eoss/notes/div1/raw; ts=$(date +%Y%m%dT%H%M%S); W=$1; REPS=$2; MODE=$3; D=/dev/shm/dv
idle() { [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU 1 busy, stop"; exit 3; }; }
for i in $(seq $REPS); do idle; b=$O/${W}_${MODE}_${ts}_$i; if [ $MODE = probe ]; then export LHOUT=$b; else unset LHOUT; fi
 case $W in
  mochi) cd $D/mochi/build && timeout -k 2 300 ./sample23-3Dparticlesim $HOME/Mochi-DCD-on-RT/datasets/100k_particles.txt 1 > $b.txt 2>&1 ;;
  knn) cd $D/knn && timeout -k 2 300 ./knn_host u1e5.csv u1e5.csv 8 0.02 > $b.txt 2>&1 ;;
  rtnn_r05) cd $D/rtnn/src/build/bin && timeout -k 2 300 ./optixNSearch -f $D/data/u100000.csv -r 0.05 -k 50 -p 0 -s 0 -m 0 -d 0 > $b.txt 2>&1 ;;
  rtnn_r025) cd $D/rtnn/src/build/bin && timeout -k 2 300 ./optixNSearch -f $D/data/u100000.csv -r 0.025 -k 50 -p 0 -s 0 -m 0 -d 0 > $b.txt 2>&1 ;;
  ark_r0112) cd $D/ark/build_fixed && DEVOUT=1 timeout -k 2 300 ./sample01-knn $D/data/ark_a1e5.txt 100000 10000 0.0112 > $b.txt 2>&1 ;;
  ark_r005) cd $D/ark/build_fixed && DEVOUT=1 timeout -k 2 300 ./sample01-knn $D/data/ark_a1e5.txt 100000 10000 0.005 > $b.txt 2>&1 ;;
  tk100k) cd $D/tk/ark/b && timeout -k 2 300 ./sample03-trueknn $D/data/u100000.csv 100000 3 0.004 16 $b.tk.txt > $b.txt 2>&1 ;;
  tk400k) cd $D/tk/ark/b && timeout -k 2 300 ./sample03-trueknn $D/data/u400000.csv 400000 3 0.004 16 $b.tk.txt > $b.txt 2>&1 ;;
 esac; echo "$W $MODE $i rc $?"
done
