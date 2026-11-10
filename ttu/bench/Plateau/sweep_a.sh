#!/bin/bash
# Plateau (a): L2 latency vs number of SMs chasing at once, for 1, 32, 256, 1024 chasers per SM
for C in 1 32 256 1024; do
  case $C in 1|32) N=2000;; 256) N=500;; 1024) N=200;; esac
  for G in 1 2 4 8 16 32 48 64 71 100 142; do /dev/shm/chase_load first:$G $C $N 6; done
done
