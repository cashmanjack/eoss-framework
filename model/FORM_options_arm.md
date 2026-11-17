# Mochi last arm: OptiX setup differences between Mochi (OWL) and the replica (committed before any run)

Date 2026-11-16. Source reading only (no GPU run) for the table; the flips run on triton GPU 1, OptiX only, no TTU code.

## Setup as read from source (Mochi: owl/DeviceContext.cpp configurePipelineOptions, buildPipeline; src/s23-3Dparticlesim; replica: ttu/bench/Load_1/load3.cu, load3_dev.cu)
| item | Mochi through OWL | replica (load3) | same? |
| module optLevel | LEVEL_3 | default 0 = all optimizations (header: "run all") | same |
| module debugLevel | MINIMAL | default 0 ("Default currently is minimal", header) | same |
| maxRegisterCount, boundValues | default, none | default, none | same |
| exceptionFlags | NONE | NONE | same |
| traversableGraphFlags | SINGLE_LEVEL_INSTANCING (max instancing depth 1) | SINGLE_LEVEL_INSTANCING (ias arm) | same |
| numPayloadValues | 2 | 2 | same |
| usesMotionBlur | false | false | same |
| numAttributeValues | 2 | 0 | DIFFERENT (flip F1) |
| usesPrimitiveTypeFlags | CUSTOM or SPHERE (curves off) | CUSTOM | DIFFERENT (flip F2) |
| link maxTraceDepth | 2 | 1 | DIFFERENT (flip F3) |
| stack sizes | optixPipelineSetStackSize(2048, 2048, 2048, depth 2), fixed, not computed | never set: OptiX default (not readable through the API) | DIFFERENT (flip F4) |
| SBT | raygen and hitgroup records carry the program data (particles pointer, world handle, stiffness, restitution), read by the programs through the SBT data pointer; records 32 byte header + 16 byte data | header only records; the same data read from the launch parameters | DIFFERENT (flip F5) |
| PTX | nvcc -O3 -DNDEBUG -lineinfo -std=c++11, no -arch: .target sm_52 (JIT by OptiX) | .target sm_89 | DIFFERENT (flip F6) |
| launch | owlLaunch2D(N, 1): optixLaunch N x 1 x 1; launch params copied before each launch | optixLaunch N x 1 x 1 | same |
| trace call | flags DISABLE_CLOSESTHIT, mask 255, SBT offset 0 stride 1 (one ray type) | same | same |
| programs | Mochi raygen and intersection | the same, OWL removed (F171) | same except the SBT data and PTX target |
Not compared: any internal state OptiX derives (stack size default, register count of the compiled module). No stack or register readout exists through the API; ncu is admin only on triton.

## Arms (one flip at a time from the baseline replica, Mochi program, instance level, real dataset, Morton order, 100,000 queries, 400 timed launches after 200 warmups, as F171/F172)
F1 attribute values 2. F2 primitive flag SPHERE added. F3 maxTraceDepth 2. F4 stack sizes as OWL. F5 program data in the SBT records. F6 PTX with Mochi's nvcc flags (.target sm_52). Then one extra arm, ALL (F1 to F6 together), only to check an interaction; it is not part of the stop rule.
Baseline to beat: replica 11.3 us span (F171/F172), per warp raygen cycles narrow (15.7k to 22.2k); Mochi frame 0: 100 us, p10 17k, median 40k, p90 189k, max 242k.

## Prediction (before any run)
I expect none of F1 to F6 to produce it. Priors: F1, F2, F3 (they change compile options of unused features) each below 5 percent; F4 and F5 about 10 percent; F6 about 15 percent. Every arm span expected within 9.6 to 13.0 us (baseline plus or minus 15 percent) and warp p90 over median below 1.5.
Rule for "produces it": span at least 34 us (3 times the baseline) AND p90 over median at least 2.0 (Mochi 4.7, replica 1.2). If an arm meets both, stop there and report it as the cause candidate (not a proven cause until the flip back removes it). If none does (including ALL), stop and record the gap as unexplained; PAPER1_OUTLINE scopes the time model to plain OptiX launches, Mochi's count result kept.
A partial effect (span 1.3 to 3 times the baseline) is reported as such without a stop and without interpretation.
Run rules: smoke launch of 32 queries first, then 100,000; timeout -k 2 30; GPU 1 idle; every exit code checked; raw files timestamped, mode x.
