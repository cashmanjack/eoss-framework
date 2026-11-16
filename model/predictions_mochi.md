# Mochi predictions (collision detection, 100,000 particles, 10 frames of 400 refit + launch substeps), written before any measured (recorder) run

2026-11-16, triton GPU 1 (RTX 6000 Ada, 142 SMs). Program: github MDurgaKeerthi/Mochi-DCD-on-RT, src/s23-3Dparticlesim, cloned to ~/Mochi-DCD-on-RT (outside the repo, never committed), built with RENDER 0, COUNT_COLLISIONS 0, BOX_WIDTH 1, 100,000 particles, CUDA 12.8, OptiX SDK 8.1. Unpatched build runs 10 frames: 2.2 s wall, totals printed by the program 1,327.8 ms refit (host timers), 284.9 ms launches, 53.4 ms update (4,000 substeps, so about 71 microseconds per launch by host timer, not a device span). Patched copy /dev/shm/mochi/src_copy (diff in notes/q24, private): after the last substep of each frame the refitted GAS of the particle group and the particles are dumped (run A1, no recorder, no timing); a recorder (second run) counts intersection calls per launch index and the in kernel span of %globaltimer (first raygen start to last end) for every launch.

What a query is: one ray per particle from its centre, direction (0,0,1), tmax 1e-16 (a point query) against one box per particle (centre -/+ 2 radius, custom primitive); the intersection program runs for every primitive of an entered leaf slot, computes a distance and, for an overlapping pair (processed once per pair), does 6 atomic adds. Particles are Morton sorted once, so launch index = row, order fixed over frames. The BVH is built once (frame 0 setup) and REFIT every substep, so its topology stays the initial one while the particles fall, hit the floor (from frame 5) and pile up.

Determinism check made first (two dump runs A1, A2 with the same input): positions are identical in frames 0 to 4, and differ from frame 5 on (float atomics, order dependent), so predictions for frames 5 to 9 from A1 describe a statistically similar state, not the exact positions of the measured run.

## Replay predictions (CPU, from the dumps of run A1; model/predict_mochi.py)

| frame | nodes / query | leaf slots / query | primitives in slots / query | hardware candidates / query | exact box hits (incl. self) / query | lane efficiency | RT part of the launch, ms (F127, 142 SMs at 2.5 GHz) |
|---|---|---|---|---|---|---|---|
| 0 | 6.25 | 1.00 | 1.08 | 1.08 | 2.00 | 0.602 | 0.0339 |
| 1 | 6.25 | 1.00 | 1.08 | 1.08 | 2.00 | 0.602 | 0.0339 |
| 2 | 6.25 | 1.00 | 1.08 | 1.08 | 2.00 | 0.602 | 0.0339 |
| 3 | 6.25 | 1.00 | 1.08 | 1.08 | 2.00 | 0.602 | 0.0339 |
| 4 | 6.25 | 1.00 | 1.08 | 1.08 | 2.00 | 0.602 | 0.0339 |
| 5 | 13.51 | 2.33 | 2.41 | 2.42 | 2.88 | 0.406 | 0.0357 |
| 6 | 49.89 | 2.96 | 3.09 | 3.11 | 2.91 | 0.412 | 0.0366 |
| 7 | 112.99 | 4.18 | 4.19 | 4.22 | 3.15 | 0.451 | 0.0382 |
| 8 | 186.76 | 5.91 | 5.36 | 5.40 | 3.28 | 0.483 | 0.0398 |
| 9 | 253.21 | 8.18 | 6.46 | 6.50 | 3.42 | 0.497 | 0.0413 |

(Table built with the replay of F125: hardware candidates = 1.006 x primitives in the entered leaf slots; exact box hits include the particle itself.) Frames 0 to 4 are one falling lattice, so identical rows.

## Time

Form F127 only (fixed 110 cycles plus 4.9 cycles per candidate and one, per SM), 142 SMs at 2.5 GHz assumed, no node term and no callback term (F127 had its miss term omitted): the RT part is 0.0339 ms (frames 0 to 4) rising to 0.0413 ms (frame 9). That is a LOWER BOUND by construction: it ignores that frames 6 to 9 fetch 8 to 40 times more nodes per query than frames 0 to 4 (50 to 253 against 6.25), and the SM side of the program (distance computation and 6 atomics per colliding pair, loads of two 32 byte particles per candidate). Callbacks took 25 to 83 percent of warp time in earlier programs (F134).

Committed time statements:
1. Measured span of a launch (median over the 400 substeps of a frame) is at least the RT part: at least 0.034 ms in frames 0 to 4.
2. Bracket for frames 0 to 4: 0.034 to 0.17 ms (RT part to 5 times it, the callback share of 80 percent seen for kNN and Arkade).
3. Trend: the candidate only form gives frame 9 over frame 0 of 1.22. Because the nodes per query rise 40 times, the measured ratio of the span of frame 9 to frame 0 is predicted to be ABOVE 1.5 (any value below 1.22 or near 1 means the node count does not matter and this statement fails).

## Count statements and the success rule (set now)
- Run B (recorder plus dump of its own state): the measured intersection calls per query (exact, per launch index, from the last launch of each frame) have a mean within 10 percent of the table for frames 0 to 4 (identical state) and the replay of run B's OWN dump matches the measured mean within 5 percent (F125 found 1.006 times the primitives in slots); for frames 5 to 9, run A1's table is compared with run B's measurement with a 25 percent tolerance (different trajectory), and run B's own replay with 5 percent.
- Per query exactness is not required; also reported: lane efficiency from the per query measured counts against the table (tolerance 0.05).
- Failure is reported as such. No tuning of the constants afterwards; a second attempt is labelled.

## What can fail
Candidates beyond the 1.006 factor (a refit tree may behave differently from a freshly built one, as the replay uses the dumped, refit boxes); the time lower bound (the recorder adds a globaltimer read and two atomics per ray); the clock assumption 2.5 GHz (the card allows 3.1 GHz).
