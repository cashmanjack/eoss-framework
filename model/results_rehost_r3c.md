# R3 with the count loop (2026-11-24, triton, GPU 1): C1 exact, C2 mismatch, stopped
Prediction first: model/results_rehost_r3_offline.md, section "Proposed fix" (de9e77d): C1 exact 0.85; sessions = answers (+1) 0.8; no fault or hang 0.95. Code d15a281 (inner count loop, BSSY/BSYNC around it, check rules L1 two loops and L2b; regression of the 24 old build configurations x 2 stages: 48 of 48 byte identical, diff notes/check_seq_count_20261124.diff). Raw (private): notes/rehost/r3c_*, GAS of every scene notes/rehost/gas_r3c_*.bin.
Stage 0 (no TTU), injected answers with counts 1, 2, 3 and 8 per lane (lane i uses count 1, 2, 3, 8 for i mod 4), r 0.15 to 0.8, 6 runs, 256 or 512 lanes: every neighbour count and id equals the host's float32 expression. (A first stage 0 run showed false mismatches: the host compared an unsorted expected list; fixed.)
## Stage 1, one warp, 32 queries each, no fault, no hang
| config | points, r, seed | result |
|---|---|---|
| C1 | 4,096, 0.08, 7 | EXACT: 0 of 32 differ from OptiX and from the CPU search; 566 candidates in 460 answers (14.4 per query), loop 59,900 cycles per lane, session window 22,200 |
| C2 | 4,096, 0.12, 8 | MISMATCH: 30 of 32 queries differ; neighbour counts equal OptiX's, but our lists contain 0xffffffff (2 or 4 times per lane) in place of the same number of real neighbours; OptiX candidates 48.8 per query, our answers 39.9; stopped |
| C3, C4 | not run (stop rule) | |
C2 reading from the saved GAS and the lists: the record table has exactly 4,096 records, none null (word 0 = primitive index for all, word 1 = 0), and the data after the table does not hold 0xffffffff at the places a run could overshoot to, so the stored id is not table data: it is an invalid or stale register value taken as a primitive id in the new count loop (the point it addresses is 16 bytes before the points array). It appears only with larger counts and many answers per query; stage 0 with counts up to 8 and one answer per lane cannot show it. A per session trace of the count loop build on C2 would show which answers and which loop iteration (not run; needs approval).
Scoring: C1 exact (0.85 held), sessions equal answers (held), no fault or hang (held). C2: no prediction was committed for the other scenes beyond "continue if C1 is exact".
## Protocol fact recorded (F452, protocol.md)
Candidate answer = a run of consecutive 8 byte records: count in R33 bits 5:0, first record at (R45:R44) x 8, record j at pointer + 8 j, word 0 = primitive index. R1 and R2 only ever saw count 1.
