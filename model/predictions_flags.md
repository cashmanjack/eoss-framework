# Flags, masks and tmax: predictions, written 2026-11-16 BEFORE any run of these tests (Feature_1, Feature_2, Feature_5 of MASTER_PLAN Phase 9; OptiX launches only, no TTU sequence of ours)

Pre-registration. GPU: tethys GPU 1 (RTX 4090), ordinary OptiX launches of a new small host program (ttu/bench/Flags_1). Not edited after the run; results go in model/results_flags.md.

## Method (fixed now)
- Raygen with clock64 probes before and after optixTrace (as HG_1: TRACE = latest return minus earliest start over the 32 lanes of one warp), 100 launches of 32 rays, L2 warm and L1 cold per launch, median. The first call of a launch pays about 1,200 cycles of cold code and OptiX adds a constant of about 4,000 cycles per call (HG_1), so differences between configurations are the quantity, not the absolute numbers.
- A second raygen mode traces the same ray 3 times and reports the third (L1 and code warm).
- ncu (TMPDIR set to an own directory): on a launch of 262,144 jittered rays of the same scene, RT unit L1 sectors per ray = (l1tex__t_sectors.sum minus l1tex__t_sectors_pipe_lsu.sum) / rays (as F124), thread instructions per ray, kernel time.
- Reported next to every number: the hit and primitive the ray returned (checked against a CPU reference where the flag does not change the answer).
- Scenes: layers:L = L large triangles in parallel planes z = 0.1 k (k a random permutation, so storage order is not distance order), rays along +z from z = -1 near the axis; alternating facing for the cull test (even planes front, odd planes back). soup:100000 (the F81 soup, random rays) for tmax. IAS of K instances of one triangle GAS for masks.

## 1. Terminate on first hit (layers:64)
Flags NONE against OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT. F56 says the unit returns on every hit anyway and the flag only stops the resume loop on the SM.
- Prediction: TRACE falls by 5 to 30 percent (point 15: one or two resumes saved, each about 800 to 1,600 cycles at one warp); RT L1 sectors per ray fall by 20 to 60 percent; the returned primitive is the nearest plane in under half of the rays with the flag and in all rays without.
- A fall by less than 3 percent in sectors would say the resumes are already pruned by tmax after the first hit (nearest first) and the flag buys nothing for this scene.

## 2. Disable any hit (layers:16, an any hit program that ignores every hit)
NONE (the any hit program runs for every hit), OPTIX_RAY_FLAG_DISABLE_ANYHIT, ENFORCE_ANYHIT, and the geometry flag OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT at build time.
- Prediction: NONE costs one SM round trip per layer: 850 to 900 cycles each as F49 found for custom primitives (range 500 to 1,100), so TRACE with 16 hits is 12,000 to 22,000 above DISABLE_ANYHIT; ENFORCE equals NONE within 3 percent; the geometry flag equals the ray flag within 3 percent. L1 sectors per ray: DISABLE about half of NONE or less (no resumes).
- Per any hit call (difference divided by 16): 850 (500 to 1,100).

## 3. Cull flags (layers:16, even planes front facing, odd planes back facing, the plane nearest the ray is back facing)
NONE, CULL_BACK_FACING_TRIANGLES, CULL_FRONT_FACING_TRIANGLES, both culls (a miss).
- Prediction: F56 found back face culling works through the flag words, so the unit skips culled triangles without a return. CULL_BACK: the nearest plane is skipped, the next (front) is returned: TRACE within 10 percent of NONE (range -10 to +15); CULL_FRONT: the nearest plane is returned, as NONE: within 5 percent. Both culls: no hit, tmax never shrinks, every box on the ray is visited: TRACE 1.2 to 2.5 times NONE, L1 sectors per ray 1.5 to 4 times NONE.

## 4. Masked out instance, does it still cost a node fetch? (Feature_1)
IAS of K = 64 instances of one triangle GAS at z = 2 k (the ray passes through all 64 boxes). Instance masks 1 for k = 0 to 62, 2 for k = 63 (the farthest). Scene A: ray mask 2 (63 nearer instances masked out, the farthest visible). Scene B: ray mask 255 (the nearest instance is hit, the rest pruned by tmax). Scene C: an IAS of the farthest instance only, ray mask 2 (nothing masked). Scene D: 64 instances, all masks 2, ray mask 2 (nothing masked out, closest is the nearest).
F111: the mask is byte 6 of the IAS child entry, so the rejection can be made from the IAS node alone, without fetching the instance node or the GAS root of that instance.
- Hypothesis H1 (rejected from the child entry): A minus C = the IAS interior nodes needed to pass 64 instance boxes only (64 children at 3 per node: about 22 to 30 lines, 4 sectors each) and no instance node or GAS fetch: RT L1 sectors per ray A minus C = +60 to +130 (point 90).
- Hypothesis H2 (every masked instance still costs its instance node and GAS root line): A minus C = H1 plus 2 lines per masked instance = +250 to +400 sectors.
- I predict H1, with 65 percent. Sectors per ray A against D (same traversal, nothing rejected, but D's closest hit prunes after the first instance): D is the lower bound for the same tree.
- TRACE A minus C: +4,000 to +10,000 cycles (22 to 30 node fetches at a cold launch, 150 to 340 each); with the warm third trace +500 to +2,000.

## 5. tmax pruning on a range query scene (Feature_5)
soup:100000, 262,144 random rays (origin uniform in the cube [-1, 1]^3, random direction), tmax r in 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 1e16. tmin 0.
- Prediction (F70: boxes beyond tmax are never visited; F44: an empty interval costs most of a query for origins inside the scene; F81/F124: about 85 to 95 RT L1 sectors per ray at tmax unlimited on the 100,000 soup; the scene edge is 2, the triangle edge 0.065, the mean distance to the closest hit about 0.1 to 0.2): RT L1 sectors per ray 12 (6 to 25) at r = 0, 20 at 0.01, 30 at 0.02, 45 at 0.05, 65 at 0.1, 80 at 0.2, 90 (+-12) from 0.5 on and equal to unlimited at r >= 1 (within 3 percent). Hit fraction rises from 0 to the unlimited value (about 0.9 or more) by r about 0.3.
- Time per ray (kernel time of the launch, mean of 20): follows the sectors, 0.3 (0.15 to 0.5) of the unlimited value at r = 0.01, 0.75 at r = 0.2.
- The saving by r = 0.05 relative to unlimited: nodes and time both 40 to 65 percent lower.
- One warp, TRACE: r = 0.05 against unlimited, 32 random rays: 0.5 to 0.8 of the unlimited TRACE (the constant 4,000 cycle per call overhead of HG_1 stays, so the ratio is closer to 1 than the per ray sectors).

## What each outcome would mean
- Items 1 to 3 inside the ranges: the flags save SM side resume rounds, not RT unit work, except culling which saves returns; the model needs no flag term except the resume count.
- Item 4 H2: masks do not prune cheaply; instance heavy mapping with mask pruning pays a line per instance.
- Item 5 flat in r (within 10 percent from r = 0.05 on): tmax does not prune on this scene (all work is the root path), the radius saving of RTNN style mappings would come from other effects.
