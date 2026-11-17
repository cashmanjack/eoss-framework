# Lone masked instance among visible ones: predictions (2026-11-16, tethys GPU 1, written BEFORE any run; OptiX only, no TTU code of ours)

Question: F181 showed 63 masked out instances in one contiguous block cost 4 sectors and 43 cycles (rejection without a fetch). Does a masked instance that sits BETWEEN visible siblings cost a fetch when the siblings are read?

Scenes (new modes of ttu/bench/Flags_1, ias:K:mode): K = 256 instances of one triangle GAS at z = 2k, one IAS. Rays miss the triangle but pass through every instance's box (origin near (3.5, 3.5), triangle edge x = 0.25 at y = 3.5), so every VISIBLE instance is entered down to the triangle test and no hit prunes the rest; no any hit program. Ray mask 1.
- V: all 256 visible (mask 1). R16: instances k = 8 mod 16 absent, 240 visible. L16: same instances present with mask 2 (lone masked, 16 of them, each between visible ones), 240 visible. E: odd k masked (mask 2), 128 visible interleaved with 128 masked. Q: only the even k present, 128 visible.
Measures (as Flags_1): one warp 32 rays, 100 launches, trace cycles cold (trace0) and warm (trace1 of nrep 3), L1 fill sectors of a 32 ray launch under ncu (fills32), RT unit L1 requests per ray at 262,144 rays.

Predictions:
1. Visible instance cost (V minus Q divided by 128 visible): fills32 5 to 13 sectors per visible instance (instance node line 4 sectors, GAS root, triangle leaf); V fills32 1,200 to 3,500.
2. H1 (mask rejected at the parent's child entry, no instance fetch; the F111 layout puts the mask in the entry): L16 minus R16 within 3 sectors in fills32 and within 3 percent in trace0 and trace1. I put 65 percent on H1.
3. H2 (the masked instance's own line is read when its siblings are, because it shares a node with them): L16 minus R16 = +16 to +64 sectors (one to four sectors per masked instance), trace +3 to +10 percent.
4. E against Q (128 visible either way, 128 masked interleaved): H1 means within 3 percent; H2 means E higher by up to 128 x 4 sectors.
5. R16 against V: lower by about 16/256 of V (6 percent) in sectors and time.
Reading rule: |L16 - R16| over (V - R16) below 0.25 = masked instance costs no fetch (H1); above 0.75 = costs like a visible one; between = partly (e.g. only the interior node entry).
