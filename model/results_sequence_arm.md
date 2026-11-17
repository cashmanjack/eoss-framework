# Clock watch, nodes check, OWL flags and the between launch sequence arm (F172; form and predictions: model/FORM_sequence_arm.md, committed before the sequence runs)

2026-11-16, triton GPU 1, OptiX only (raw private in notes/q31). The checks and their results are written in FORM_sequence_arm.md (clock: 2,700 MHz in 498 of 589 samples during unpatched Mochi, longest warp of frame 0 242,321 cycles in the 100.4 us span so the in kernel clock is at least 2.41 GHz; replica nodes per query 6.25, leaf slots 1.00, primitives 1.08, lane efficiency 0.602, identical to Mochi frame 0; OWL ORs no flags into the GAS or the instance group, GAS build flags are exactly ALLOW_UPDATE).

## Sequence arm (replica = Mochi program, instance level, update flag only, 100,000 queries; 200 warmup iterations, 400 timed launches, median)
| queries | sequence before each launch | span median us | span min / max | warp cycles p10 / median / p90 / max |
|---|---|---|---|---|
| 32 | none (back to back) | 3.07 | 3.07 / 4.10 | 8766 / 8766 / 8766 / 8766 |
| 32 | update kernel + sync | 3.07 | 3.07 / 4.10 | 8797 / 8797 / 8797 / 8797 |
| 32 | update kernel + bounds + GAS refit + instance group refit + syncs | 4.10 | 4.10 / 4.10 | 10430 / 10430 / 10430 / 10430 |
| 100000 | none (back to back) | 11.26 | 10.24 / 12.29 | 15720 / 19218 / 22168 / 27001 |
| 100000 | update kernel + sync | 11.26 | 10.24 / 12.29 | 15562 / 19170 / 22157 / 26949 |
| 100000 | update kernel + bounds + GAS refit + instance group refit + syncs | 11.26 | 10.24 / 12.29 | 15675 / 19152 / 21955 / 26858 |

Mochi frame 0 for comparison (run C, F168): span 100.4 us; warp cycles 17,147 / 39,658 / 189,321 / 242,321 (10th, median, 90th, max; mean 78,314).

Verdict by the rule set before the runs: the sequence is RULED OUT (SEQ 2 median 11.26 us, the same as back to back; SEQ 1 also 11.26 us, so the update kernel's cache state is not it either). Predictions: SEQ 0 and 1 held (10 to 13 and 10 to 16 us); SEQ 2 (20 to 100 us, centre 40) missed low. The clock is ruled out (above). The replica replicates the traversal work exactly (nodes, leaf slots, primitives per query, lane efficiency), and the build flags and structure.

What is left, and a lead in the data: the replica's warps are all alike (10th to 90th percentile 15,700 to 22,200 cycles), Mochi's are not (17,000 to 189,000), the excess is a tail of long warps that grows with the warp index (correlation 0.65 with the index; the first eight warps of the launch all take 30,8xx cycles within 20 cycles of each other, the next four 118,3xx), as if groups of warps wait for the same event. Candidates not run: the OWL pipeline and stack configuration (maxTraceDepth 2, OWL's continuation stack setting; the replica used 1 and OptiX defaults), the SBT record loads (Mochi reads the geometry data from the hit group record, the replica from the launch parameters), OWL's launch of the 2D launch (the same dimensions here), the real moving particles (the replica's tree has the same nodes and calls as the dumped frame 0 tree, so the traversal work is the same, but the memory layout of the GAS built by OWL's bounds program differs only if OWL lays out differently), and anything that differs between the OWL process and a plain OptiX one (memory placement, other allocations, context options). Cheapest next step: run Mochi's own pipeline (OWL) on the replica's inputs, or the replica with OWL's pipeline settings, predictions first.
