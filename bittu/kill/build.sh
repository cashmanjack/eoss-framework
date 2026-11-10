#!/bin/bash
# Build the BITTU kill test tool against NVBit (default ~/opt/nvbit-1.8.1) into $1 (default /dev/shm/bittu).
set -e
NVBIT=${NVBIT:-$HOME/opt/nvbit-1.8.1}; OUT=${1:-/dev/shm/bittu}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
nvcc -ccbin=g++ -I$NVBIT/core -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_89 -Xcompiler -fPIC -c $HERE/inject.cu -o $OUT/inject.o
nvcc -ccbin=g++ -I$NVBIT/core -Xcompiler -fPIC -dc -c $HERE/kill.cu -o $OUT/kill.o -arch=sm_89 -O2
nvcc -ccbin=g++ -arch=sm_89 -O2 $OUT/kill.o $OUT/inject.o -L$NVBIT/core -lnvbit -lcuda -shared -o $OUT/bittu_kill.so
echo "built $OUT/bittu_kill.so"
