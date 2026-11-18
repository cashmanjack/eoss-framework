# Predictions Phase 10f item 3 (tethys, RTX 4090, 2026-11-18): model v1 on a second chip, synthetic inputs. Written BEFORE any run of these four programs on this GPU.

Model v1 as frozen (model/priorworks/FORM_combined_v1.md: time = cycles / (SM count x clock), the form and every constant from triton, nothing refitted). Chip change only: SM 142 -> 128 (4090), model clock constant 2.5e9 unchanged (the 4090 runs at about 2.5 to 2.73 GHz in our arms; the frozen form has one constant, so I keep it). Same program builds (~/thirdparty/div1, OptiX only, no TTU code, probes off where a switch exists). Same input structure as on triton, NOT the same point files (regenerated from the seeds in notes/rebuild/rebuild_div1.sh: u100000, u400000 unit cube 3D, Arkade 2D 110,000 points, kNN 200,000 uniform points). I have not looked at any 4090 timing of these four programs.

Rule: within a factor 2 (ratio measured/predicted 0.5 to 2); also the median absolute log error. Throughput scaled rows: predicted = triton measured x 142 / 128 = x 1.109; latency bound rounds (the last, with a handful of active queries) x 1.0.

| program | input | predicted (ms) | basis |
|---|---|---|---|
| RTNN search compute (-p 0 -s 0 -m 0 -k 50) | u100000, r 0.05 | 0.191 | CPU counts: cube candidates 93.8, sphere hits 50.5, qm 102.2, (110 + 4.9 qm) N / (128 x 2.5e9) |
| RTNN | u100000, r 0.025 | 0.058 | counts 13.0 / 7.35, qm 15.3 |
| kNN original (own program, any hit, heap in global memory, K 16, r0 0.004, doubling), rounds 1 to 4 per round | 200,000 uniform points = queries | 0.733 / 1.086 / 3.884 / 12.22; total rounds 1 to 4 17.9 | F138 triton 0.661 / 0.979 / 3.502 / 11.02 ms x 1.109 |
| kNN register heap (same, F138 fix), rounds 1 to 4 | same | 0.183 / 0.250 / 0.767 / 3.26; total 4.46 | F138 0.165 / 0.225 / 0.692 / 2.94 x 1.109 |
| kNN speedup of the register heap over the original, rounds 1 to 4 total | same | 4.02 | F138 measured 4.02 (a chip independent ratio in this form) |
| Arkade fixed (float distance, device buffer, r 0.005, 100,000 points, 10,000 queries as in the Div runs), kernel time | ark_a1e5 | 0.20 | F151 float device buffer 0.18 ms x 1.109 (note: F151 used 100,000 queries; the Div runs use 10,000, so this row is for the SAME launch only if I run 100,000 queries: I will run 100,000 and report 10,000 as an extra, unscored) |
| TrueKNN (port, k 16, r0 0.004, doubling, 5 rounds), total of the per round launch spans | u400000 | 12.95 | F160 measured total 11.74 ms; rounds 1 to 4 x 1.109 plus round 5 (0.60 ms, latency bound) x 1.0 |

Predictions with probabilities:
1. RTNN rows both inside a factor 2: 60 percent (F250: v1 without a callback calibration is a lower bound of the launch time on new data, so I expect measured above predicted, 1.0 to 2.5 times; the 0.025 row is small and has the fixed launch cost).
2. kNN original and register heap, all four rounds inside a factor 2: 60 percent each. Total of rounds 1 to 4 within 25 percent: 40 percent.
3. Register heap speedup between 3.0 and 5.5: 65 percent (it is about the ratio of two global memory chains; the 4090 has a smaller L2 (72 MB against 96) that still holds the 25.6 MB of heaps).
4. Arkade fixed inside a factor 2: 45 percent (a 0.2 ms launch; launch overhead and the fixed part decide).
5. TrueKNN total inside a factor 2: 65 percent; measured above predicted (the rebuild of this program was 1.4 to 3.3 times slower than the F160 build in rounds 1 to 3, F160): 65 percent.
6. All eight scored rows (RTNN 2, kNN original total, kNN reg total, speedup, Arkade, TrueKNN; the per round kNN rows counted separately): at least 6 of the 8 headline rows inside a factor 2: 50 percent. The cycle form transfers to a second chip within a factor 2 on most rows: 55 percent.

# Item 4: node state on the 4090, validated sequences (written before any TTU launch of these arms)
Same arms as the triton runs, our TTU kernel (the checked group), one TTU warp on SM 0, two sessions of the same ray, scenes par:N:s (N 1,024 = 68 KB, 2,048 = 132 KB BVH, size independent of the GPU), 3 repetitions of 30 launches, medians, stage 0 of every build first then one stage 1 launch per arm, 128 blocks where the arm needs the sibling block (TPC = SM 2k, 2k+1 holds on the 4090: probe_placement; the F145 placement rule is in gpucfg). Saved fraction s = 1 - (second session cycles) / (single session cycles).
Hold = the same qualitative outcome as on triton: 
- F150 capacity curve: s between 0.45 and 0.70 up to N 2,048 (132 KB) with no trend, s at most 0.25 from N 4,096 (263 KB) on, at most 0.08 from N 16,384 on, and the TTUCCTL arm (cctl) within 0.05 of 0 at every N. Probability that all hold: 60 percent (the L1 is 128 KB per SM on both chips, but the 4090 has a smaller L2 and the knee position depends on the BVH shape; the knee location can move by one step: I accept a knee between 132 and 330 KB).
- F159 fence decomposition (N 1,024 and 2,048): sibling polling without a fence s 0.45 to 0.70 (b1); cta fence on the sibling no loss (s above 0.45); gpu fence on the sibling s 0.15 to 0.40; gpu fence on the TTU's own SM s 0.15 to 0.40; gpu fences on both s at most 0.12. Probability all five hold: 55 percent.
- F161 TTUCCTL alone on the sibling SM: s at most 0.06 at both sizes (0.00 on triton). Probability 75 percent. F162 own TTUCCTL (arm i): s at most 0.06. Probability 75 percent.
- F174 own L1 stream (1 MB of ordinary loads on the TTU's SM between the sessions) against the same wait without loads: s(stream) at least 0.10 below s(wait) at both sizes (triton: 0.29 and 0.27 against 0.53 and 0.42). Probability 60 percent.
Overall: at least 4 of the 5 result groups hold on the 4090: 60 percent.
Run safety: only checked builds (check_seq ALL PASS), stage 0 first, smallest launch first, timeout 30 via run.py, GPU 1 idle, stop at the first fault, wrong exit, missing block or timeout.
