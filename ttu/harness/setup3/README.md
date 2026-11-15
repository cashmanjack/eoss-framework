# Setup_3 harness: one ray, one triangle, one TTU group

Build (CUDA 12.8, OptiX SDK 8.1), all outputs in /dev/shm/setup3:

    nvcc -O2 -arch=sm_89 -I$HOME/optix_sdk/include -o /dev/shm/setup3/setup3 setup3.cu -lcuda
    nvcc -O2 -arch=sm_89 -ptx -I$HOME/optix_sdk/include -o /dev/shm/setup3/setup3_dev.ptx setup3_dev.cu
    # words.txt: the 8 words of TTUST 0 and 0x80, generated from private notes
    python build_patch.py --stage 0|1 --dump <private OptiX dump>
    python check_seq.py /dev/shm/setup3/setup3_s<stage>
    python run.py /dev/shm/setup3/setup3_s<stage> hit|miss zero|random OUT.json [--no-ttu]
    python analyze.py OUT.json <stage>

Stage 0 has no TTU instruction (the group slot is NOPs). Stage 1 copies one fused
TTU group from a private dump at build time. Patched binaries contain NVIDIA code:
keep them in /dev/shm and delete them after a run; never commit them. run.py only
runs a binary whose sha256 passed check_seq.py, on an idle GPU 1, under a 30 s timeout.

## IAS (BVH_4, F91, F114)

    nvcc ... -o /dev/shm/t9i/setup3 setup3.cu -lcuda -lnvidia-ml          # host with --ias
    python ias_words.py IAS_DUMP.bin /dev/shm/t9i/words.txt               # store 0 and 0x80 words, computed from the dump's straight line code
    python build_patch.py --stage 0|1 --bin /dev/shm/t9i/setup3 --dump IAS_DUMP.bin --ias --out /dev/shm/t9i/ias_s<stage>
    python check_seq.py /dev/shm/t9i/ias_s<stage> IAS_DUMP.bin            # shape (A4, A12, A13: offsets, predicates, fuse 15) and, with --remap, A10
    python run.py /dev/shm/t9i/ias_s0 hit zero OUT.json --ias             # stage 0 only; analyze.py reads "ias" from the JSON
Stage 1 IAS builds are checked but not run (the .checked marker is deleted).
