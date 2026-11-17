# Box quantization false positives: predictions (2026-11-17, tethys RTX 4090 GPU 1, written BEFORE any Struct_1 dump or counted run and before any replay)

Question: how often do the 8 bit child boxes (F39: per axis exponent, 8 bit lo and hi relative to the node origin; conservative rounding) make the hardware visit nodes or leaf slots that exact boxes would skip?

Definitions (same tree, same leaf slots, same rays):
- H: hardware intersection program calls per query (counted by the custom primitive intersection program, which never reports a hit).
- Q: CPU replay over the decoded quantized boxes (what the unit tests), calls = primitives in the entered leaf slots, plus nodes fetched and leaf slots entered.
- E1: the same replay with exact boxes: every child slot box replaced by the tight float union of the primitive boxes below it (leaf slot: its primitives; node slot: its children). Same topology, same slot assignment.
- E0: queries whose own primitive box contains the point (the exact candidate count, F116).
Quantization false positives = Q minus E1; grouping (a slot box covers more than one primitive, boxes of siblings overlap) = E1 minus E0.

Method: ttu/bench/Struct_1 dumps the GAS, the primitive boxes and the rays and records H per query (one launch, 262,144 queries). Scenes (custom AABB, 100,000 boxes): u uniform cube, half width 0.035 and 0.07; c 200 clusters (sigma 0.03), half width 0.01; t thin rods (half widths 0.3 R, R, R in a random axis, R 0.01, so long thin boxes); o uniform shifted by +1000 (half width 0.07). Rays: point queries (tmax 1e-16) on all scenes; segments (random direction, tmax 0.25) on u 0.07 and t. Leaf slot assignment by box matching (exact on a fresh tree, F167), checked by the sanity rule below. Triangles: the 100,000 soup of F187 (existing dump, notes/tmaxreplay, 262,144 rays, tmax unlimited): replay Q against E1 (mode 1 assignment), and with F187's weights (2.2 per node, 9.9 per leaf slot) against the measured 80.8 unit requests per ray; no new hardware run for triangles.

Predictions:
1. Sanity: the quantized box contains the exact box in every slot (100 percent; if more than 1 percent of slots violate, the assignment is wrong and E1 is invalid for that scene).
2. Recorder check: H against Q mean calls per query within 3 percent on every scene (F166 and F167: 1.4 to 2.9 percent), and H closer to Q than to E1 on every scene. Per query H equals Q exactly for at least 90 percent of queries on the point query sets.
3. Size, uniform scenes: Q over E1 calls per query 1.02 to 1.08 (quantization), E1 over E0 1.03 to 1.12 (grouping); total H over E0 1.05 to 1.20 (F116: 1.10 on a uniform cloud).
4. Thin rods have the largest quantization share: Q over E1 at least 1.10, and at least twice the excess of uniform 0.07 in relative terms.
5. Clusters: Q over E1 between 1.03 and 1.15 (child boxes small against the node extent, slack 1/256 of the node extent).
6. Offset +1000: H within 3 percent of the unshifted uniform scene (relative encoding), the box slack the same.
7. Nodes and leaf slots (replay only, no hardware count of these on custom primitives): extra node fetches Q over E1 +2 to +12 percent, extra leaf slot entries +3 to +15 percent on uniform scenes; fraction of queries with at least one extra leaf slot or node 30 to 80 percent on u 0.07, and below 50 percent on u 0.035.
8. Segments: the same ordering as the points (H closer to Q than E1, Q over E1 above 1), within the ranges of 3 within a factor 1.5.
9. Triangle soup: Q requests (weights) within 10 percent of 80.8, E1 requests 3 to 15 percent below Q and further from the measured value than Q.

Reading rules: scored per number, held if inside the range on every scene it names. If 2 fails (H not within 3 percent of Q) the replay is not the hardware's box test for that scene and no quantization share is claimed from it. The claim that quantization causes the F166 residual (4.6 to 27.5 percent of queries with more calls in hardware than the replay) is tested only if the Mochi dumps are used; they are not here (existing run B files only if still on disk, otherwise not tested). No refit.
