#!/bin/bash
# Builds liboptixdump.so (LD_PRELOAD loader hook) and libcuptidump.so
# (CUPTI module dump) next to this script. Needs CUDA_HOME (source ~/cuda128.sh).
set -e
cd "$(dirname "$0")"
: "${CUDA_HOME:?set CUDA_HOME}"
CUPTI=$CUDA_HOME/extras/CUPTI
gcc -O2 -Wall -fPIC -shared -o liboptixdump.so hook.c -ldl -lpthread
gcc -O2 -Wall -fPIC -shared -o libcuptidump.so cupti_dump.c \
    -I"$CUDA_HOME/include" -I"$CUPTI/include" -L"$CUPTI/lib64" -lcupti -lpthread
echo "built liboptixdump.so libcuptidump.so in $(pwd)"
