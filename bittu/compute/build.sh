#!/bin/bash
# Build the BITTU compute mode tool against NVBit (default ~/opt/nvbit-1.8.1) into $1 (default /dev/shm/bittu). CUDA tools must be on PATH (source ~/cuda128.sh).
set -e
NVBIT=${NVBIT:-$HOME/opt/nvbit-1.8.1}; OUT=${1:-/dev/shm/bittu}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
[ -f $HERE/ttu_tables.h ] || /home/min/a/cashman3/eoss/.venv/bin/python $HERE/gen_tables.py > $HERE/ttu_tables.h
nvcc $BITTU_EXTRA -ccbin=g++ -I$NVBIT/core -I$HERE -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_89 -Xcompiler -fPIC -c $HERE/inject_bittu.cu -o $OUT/inject_bittu.o
nvcc $BITTU_EXTRA -ccbin=g++ -I$NVBIT/core -I$HERE -Xcompiler -fPIC -dc -c $HERE/bittu.cu -o $OUT/bittu.o -arch=sm_89 -O2
nvcc -ccbin=g++ -arch=sm_89 -O2 $OUT/bittu.o $OUT/inject_bittu.o -L$NVBIT/core -lnvbit -lcuda -shared -o $OUT/bittu.so
echo "built $OUT/bittu.so"
