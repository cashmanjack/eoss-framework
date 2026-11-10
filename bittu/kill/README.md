# BITTU kill tests (NVBit)

NVBit 1.8.1 (needs CMake 3.29 for its own build; `build.sh` uses nvcc directly). CUDA tools must be on PATH
(NVBit calls cuobjdump and nvdisasm).

    ./build.sh /dev/shm/bittu
    BITTU_MODE=list|probe|regs BITTU_GROUP_OFF=0x320 BITTU_GROUP_LEN=19 BITTU_OUT=report.txt \
        python ../../ttu/harness/setup3/run.py BIN hit zero OUT.json --inject /dev/shm/bittu/bittu_kill.so

The tool instruments only when the instructions NVBit hides form one gap that matches the checked group
(offset and length from the check_seq layout), and puts probes only on the instructions around it.
