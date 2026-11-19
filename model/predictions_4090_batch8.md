# RTX 4090 (tethys GPU 1), session 8 predictions, 2026-11-19. Written BEFORE the run. Plain CUDA only (no TTU, no OptiX), same rules as sessions 6 and 7.

Test: the 5090 session's ttu/bench/pc14/l2stream.cu (commit 49b3f58) compiled unchanged with nvcc -arch=sm_89 and run on the 4090 (all modes and footprints 2, 8, 24, 48, 72, 256 MB; the 4090 L2 is 72 MB so 72 MB is at the capacity edge, not inside it), plus one ncu pass (24 MB, modes 0 and 1) for lts pct and sector counts. GPU 1 idle checked first. Disclosure: I have seen the 5090's results (read 82.7, write 41.3 sectors per cycle at 24 MB, ratio 0.50) and my own l2rw numbers on this chip (coalesced 128 B per warp: write over read rate 0.66; per lane sector 0.40). The pc14 session predicted a write over read sector rate of 0.15 to 0.25 if the F419 6.8 fit were right; F423 already measured 2.5 cost (rate 0.40) for per lane sectors and 1.5 (0.66) coalesced, so the fit is expected to fail here as well.
Predictions (4090, 24 MB unless stated):
1. Write only over read only sector rate between 0.4 and 0.8: 75 percent; between 0.15 and 0.25: 3 percent; within 0.1 of the 5090's 0.50: 35 percent.
2. Read only between 3.5 and 6.0 TB/s requested: 65 percent (l2rw coalesced 3.92; the 4090 has fewer SMs and a smaller L2 than the 5090's 6.80).
3. Copy 1:1 between the write only and read only rates: 80 percent. Two reads one write above the read only rate (as on the 5090): 35 percent.
4. 8 to 48 MB within 15 percent of each other per mode: 65 percent. 72 MB within 15 percent of 48 MB: 45 percent. 2 MB below 8 MB by more than 15 percent for read only: 50 percent.
5. 256 MB: read only at most 0.5 of the 24 MB rate: 80 percent; write only at most 0.5: 65 percent.
6. ncu: lts pct of peak at least 85 for read only and write only both (as on the 5090 and in l2rw): 70 percent; sector counts equal bytes / 32 within 3 percent: 85 percent.
7. Rule for the report: the write over read sector rate here is compared with F423 (0.66 coalesced l2rw, 0.40 per lane) and with the 5090 (0.50); if within 0.1 of the l2rw coalesced value the two benchmarks agree on this chip. The 6.8 fit counts as refuted for the 4090 if the ratio is above 0.25.
