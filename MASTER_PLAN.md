# Introduction
The goal of this project is to characterize the NVIDIA RT core which is expressed as TTU "Tree Traversal Unit" instructions at the SASS level. The RTX 6000 Ada (SM89) will be the primary target hardware for this effort. Every microbenchmark and BITTU test will be repeated on an RTX 4070 Ti (SM89, a smaller chip of the same generation) and on a GPU from a different generation (TBD). This project will consist of a suite of microbenchmarks designed to characterize a performance model of the RT core and a binary instrumentation tool to measure and validate performance characteristics.

The RAGRT project showed us that mapping a problem onto the RT cores can make the performance slower despite expectations. Currently the only way to find this out is to build the entire mapping then measure it. Search dominating runtime, data being natively 3D, scenes being sufficiently large, rays having few hits, and data being irregular enough that grids lose are the expected conditions which would justify RT speedup but these conditions are largely qualitative and not enough.

This project aims to replace the existing approach of reasoning about mapping performance with a performance model of the RT core based on actual measurements of the hardware, with the tooling needed to evaluate RT core performance in real programs.

# EoSS
EoSS is a framework which runs arbitrary SASS instructions on real GPUs bypassing the NVIDIA assembler. EoSS uses the readable ISA description "Rosetta String" from NVIDIA's tooling and parses it into a per architecture JSON. This includes every instruction class along with its opcode, operand bit fields, modifiers, scheduling properties and the constraint expressions that make an encoding legal. From this EoSS encodes using a constraint aware generator which builds valid 128 bit encodings for any class which includes the control bits, stall count, yield, scoreboard read, write, and wait masks. A pure Python CUDA ELF parser finds the .text sections in a compiled binary, locates the kernel code by byte anchors and patches instructions in place. Then, an nvcc compiled kernel with thousands of patchable "slots" bracketed by clock reads has its instructions patched by EoSS. EoSS then runs the binary and reads back the per slot timings. Finally EoSS provides a CLI tool and a SASS editor for testing and experimentation.

## EoSS Insights
EoSS discovered eight TTU instruction classes: TTUOPEN, TTUMACROFUSE, TTUST, TTUGO, TTULD, TTULD.CLOSE, TTUCLOSE, and TTUCCTL. 

TTU instructions are hidden by cuobjdump and nvdisasm and unreachable to PTX as only OptiX's JIT compiler emits them. The EoSS paper runs them from a plain CUDA compute kernel. The standard TTU sequence always starts with TTUOPEN then TTUMACROFUSE N and then N fused instructions including TTUST, TTUGO, TTULD, and TTULD.CLOSE. 

If the fuse count is not equal to the correct number of fused instructions then the GPU hangs. Sessions are per warp and are tracked using SR_TTU_TICKET_INFO (SR133). A second TTUOPEN on an open session faults. Omitting TTULD.CLOSE permanently hangs the GPU. The driver closes open sessions during preemption.

The BVH must live in the low 4GB of the address space as some pointers are 32 bit. Child pointers are absolute addresses. (Not true for OptiX's own GAS on SM89: ours sat at 0x7fc0ede00a00 and traced fine. See notes/protocol.md; BVH_6 tests the limit.)

Measured on SM120, traversal latency is about 1,600 cycles at depth 1, then 2,100, 2,900, and 4,400 cycles at depths 2, 3, and 4, and nearly flat in triangle count. Per instruction issue costs are TTUOPEN 6, TTUMACROFUSE 6, TTUST 9, TTUGO 5, TTULD 5, TTULD.CLOSE 62, and TTUCCTL at 10 cycles. Validation is done via the Nsight Compute counter rtcore__rays.sum. (The installed ncu versions list no rtcore__ metric for AD102; checked 2026-11-09.)

## EoSS missing info
The README refers to `TTU/`, `Contention/`, `OSIRIS/`, `VisualModifier/` and `Unintentional/` directories, but they are not present. We need BVH builder and encoder, and the compute mode ray tracer ("Ecce Mono"), the OptiX JIT cubin dump, the in memory patch tool, and the TTU timing harness.

TTU encodings are packaged only for SM120 and our GPU is SM89. SM89 ISA JSON has all eight TTU classes but TTUST's second source and TTULD's second destination are ordinary GPR pairs Rc, while SM120 uses a uniform register field (Ra_URc). This means SM120 bytes cannot be directly copied so SM89 encodings need to be generated with EoSS encoder.

The README lists CUDA 13.0+ and ncu with counter permissions as prerequisites while docs/results.md validated SM89 using CUDA 12.8. On triton we installed CUDA 12.8 in `~/opt/cuda-12.8` (the driver supports up to 13.0), and ncu counters are open to non admins.

# SM89 ISA JSON findings
|Class|Notable fields|Reading|
|---|---|---|
|`TTUOPEN`|`/DUAL` modifier (bit 72)|Possibly a two session or paired mode. Unknown.|
|`TTUMACROFUSE`|5 bit count (`Sb`, bits 44:40)|At most 31 fused instructions.|
|`TTUST`|`TTU[ImmU16]` address, `Rb` and `Rc` each a 64 bit register pair|Each store writes 128 bits into a per session TTU address space, so `[0x40]` likely holds origin xyz plus one more float (tmin?), `[0x50]` direction plus tmax?. To verify.|
|`TTUGO`|none|Starts traversal asynchronously.|
|`TTULD`|`TTU[ImmU16]` address, `Rd` and `Rd2` 64 bit pairs, predicate output `Pu`, variable latency write scoreboard|Each load returns 128 bits plus a predicate (hit flag?). Consumers wait on the scoreboard.|
|`TTULD.CLOSE` / `TTUCLOSE`|close bit (bit 74)|Ends the session.|
|`TTUCCTL`|`.IVALL`|Cache control: invalidate all. Which cache is unknown.|

All TTU instructions are on virtual queue 21 (`VQ_TTU`).

# What we want to learn about the hardware
Each question is tagged with what its answer feeds: the performance model (PM), the binary instrumentation tool (BIT), an optimization opportunity (OPT), or the DSL's mapping decision (DSL).

## A: Interface and protocol
1. What every TTUST offset carries - origin, direction, tmin, tmax, root pointer, ray flags, cull mask, instance and SBT fields, and what is required - BIT OPT
2. What every TTULD offset returns - hit/miss, primitive ID, t, barycentrics, instance ID, geometry index, and any other information. What the Pu predicate means - BIT PM
3. Fusion rules - allowed numbers and orders of TTUST/TTULD, minimum legal group, and maximum count - BIT
4. Lane semantics - One ray per thread? What inactive or diverged lanes do, and if the cost is per ray or per warp - PM OPT
5. Session semantics - Can one session issue multiple TTUGOs? Can a warp keep several queries in flight? What does /DUAL do? SR133 values - PM OPT
6. Ray flags - closest hit vs terminate on first hit, culling, opaque handling, and if the TTU can return multiple hits per ray - PM DSL
7. TTUCCTL semantics - what it invalidates and what that costs - PM

## B: BVH format and scene building
1. Node layout - box node header, branching width, box quantization or compression, child pointer encoding, leaf and triangle block layout - BIT OPT
   - David's early tests suggest a 12 wide layout, leaves stored as one contiguous primitive list that leaf boxes point into by offset, and no duplicated triangles. To verify in BVH_1.
2. Primitive types besides triangles - custom AABB primitives, how control returns to SM for intersection, and any other native primitives - PM DSL
3. Two level scenes - instances, TLAS/BLAS, transforms - OPT
4. Starting traversal at an internal node - is it allowed, is the result identical to a root start for rays within the subtree, what happens when a ray leaves the subtree's box, and how much latency is saved per skipped level - OPT
5. Address and alignment constraints - low 4GB, node alignment, and maximum scene size - BIT

## C: Latency of single traversal
1. Cost per visited node and per tree level and if it depends on internal branching - PM
2. Cost per primitive test - triangle vs box vs custom primitive round trip to the SM - PM DSL
3. Cost per returned hit and when a ray reports more than one - PM DSL
4. Fixed overhead per traversal - open, store, go, load, and close - PM
5. Cold vs warm - first touch vs repeated traversal of the same nodes - PM

## D: Throughput and concurrency
1. How many traversals one RT core keeps in flight and the throughput curve vs warps per SM - PM OPT
2. Ray coherence - how throughput changes coherent vs incoherent - PM OPT
3. Overlap with SM work - can a warp or other warps perform useful work between TTUGO and the result? Does TTU contend with LSU/L1? - PM DSL
4. Scaling across SMs - Is there a shared bottleneck of L2 and memory - PM
5. RT cores per TPC - NVIDIA lists one RT core per SM (142 on the RTX 6000 Ada), but Cesar believes there is physically one per TPC, shared by its two SMs. If true, C_inflight and peak throughput halve per SM and TPC placement matters - PM OPT

## E: TTU memory hierarchy 
1. Which path node fetch take (L1, dedicated TTU cache, L2) and if SM memory traffic interferes - PM
2. Capacity and behavior of any TTU node cache - PM OPT
3. Bytes fetched per node and per traversal - behavior when BVH exceeds L2 (96MB on the RTX 6000 Ada) - PM DSL
4. Locality - does grouping rays that traverse the same subtree measurably reduce latency or DRAM traffic? - OPT
5. TPC sharing - two SMs form a TPC; when node data misses or is evicted from the RT core's cache, does it land in only its own SM's L1, in both SMs' L1, or neither? (raised by David) - PM OPT

## F: Traversal behavior
1. Traversal order (nearest child first? fixed order?) and early termination as tmax shrinks - PM OPT
2. Stack depth limits, short stack restarts, costs of very deep trees - PM
3. Sensitivity to overlapping and degenerate boxes - PM DSL
4. Numerical behavior - conservative box tests, watertight triangle tests, and precision of t - BIT

## G: Architecture generality
1. Which findings hold on the RTX 4070 Ti (SM89, smaller L2 and fewer SMs) and on a GPU from another generation, as well as the RTX 6000 Ada - PM DSL

# Potential performance model
This is a potential performance model which may significantly change after microbenchmarking the RT cores.

**Latency of one traversal**

T_ray ≈ t_fixed + Σ over visited nodes of t_node(cache state) + Σ over tested primitives of t_prim(type) + n_hits · t_hit

- `t_fixed`: open, stores, go, loads, close (Latency_1).
- `t_node`: per node cost, split by where the node came from (Latency_2, Latency_3, Memory_2).
- `t_prim`: per primitive test, triangle vs custom (Latency_4, BVH_3).
- `t_hit`: cost per reported hit (Latency_5).

**Throughput of one RT core**

X_rt ≈ min( C_inflight / T_ray , BW_nodes / bytes_per_ray , issue rate )

- `C_inflight`: queries in flight per RT core (Throughput_1, Throughput_2).
- `bytes_per_ray`: node and primitive bytes fetched (Memory_3).

**Whole kernel (the Amdahl part)**

T_kernel ≈ f(T_traversal, T_SM_work, overlap)

where overlap comes from Throughput_4. The DSL compares this against the cost of the best CUDA or CPU alternative for the same query.

# Microbenchmark suite
## Microbenchmark methodology
- **Harness.** Each benchmark is a compute kernel compiled with nvcc in which placeholder instructions are patched into TTU sequences with EoSS (`BinaryPatcher` + our SM89 TTU encodings). Timing uses `CS2R SR_CLOCKLO` reads; completion latency uses a scoreboard wait on the `TTULD` result, as in EoSS's `completion` mode.
- **Scenes** come from our own BVH builder (BVH_2) so that depth, width, leaf size, overlap and footprint are controlled exactly. Each scene has a CPU reference result.
- **Correctness first.** Every timing run also checks every returned result against the CPU reference. A run with wrong results is discarded.
- **Repetitions.** Warmup runs, then at least 30 repetitions; report the median and the 5th/95th percentiles. Fixed clocks where possible (`nvidia-smi --lock-gpu-clocks`), GPU otherwise idle.
- **Ground truth counters.** Cross check with Nsight Compute RT and memory counters (`rtcore__rays.sum` and the L1/L2/DRAM metrics) whenever permissions allow.
- **Records.** Each run writes a CSV plus a JSON metadata file (GPU, driver, clocks, git commit, scene parameters).
- **Safety.** Every run under a hard timeout, on a GPU nobody else is using.

## The benchmarks
### Phase 0: setup

These are prerequisites rather than research results.

| ID      | Name               | Method                                                                                                                                                   | Informs                                                    |
| ------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Setup_1 | Toolchain          | Install CUDA 12.8 on triton (matching EoSS's SM89 validation), check the driver supports it, then run `eoss gpu info`, the test suite and the plain harness validation with TTU classes excluded. Never `eoss full` or `scripts/shakedown.py`: they run each TTU class alone, which can hang the GPU. | Whether the EoSS pipeline works on our GPU.                |
| Setup_2 | SM89 TTU encodings | Generate all eight TTU classes for SM89 with the EoSS encoder, diff them field by field against the SM120 canonical sequence, and write `ttu_sm89.json`. | Every benchmark below.                                     |
| Setup_3 | First traversal    | One ray, one triangle, the canonical sequence under timeout; check the hit against the CPU reference and `rtcore__rays.sum`.                              | Confirms the SM89 port works.                              |
| Setup_4 | OptiX dumps        | Dump the JIT SASS and BVHs of the OptiX SDK samples on SM89 and extract their TTU sequences; also settles which bit TTULD.CLOSE sets. Get the code with cuda-gdb (raw bytes around `$pc` in raygen). Fallbacks, in order: (1) an `LD_PRELOAD` hook on `cuModuleLoadData`/`cuLibraryLoadData` (and their Ex variants) that saves every image OptiX loads; (2) the OptiX disk cache; (3) a device memory dump like Cesar's. Decode with the EoSS SM89 Decoder. | The full store and load protocol to decode in Interface_1, Interface_2 and BVH_1. |

### Phase 1: interface and protocol

| ID          | Name                        | Method                                                                                                                                                 | Informs                                                                                      |
| ----------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Interface_1 | Store map                   | For each `TTUST` offset seen in OptiX, zero it or perturb each 32 bit lane and observe how the result changes on known scenes.                         | A1: the ray descriptor layout; what the tool must capture per ray.                           |
| Interface_2 | Load map                    | Known scenes with known answers (hit at known t, known primitive and barycentrics); decode every word of every `TTULD` offset, and the `Pu` predicate. | A2: what the tool can report per ray; whether the hardware exposes any traversal statistics. |
| Interface_3 | Fusion rules                | Vary the number and order of stores and loads in the group and the fuse count, under timeout. Also raise the count to N+1 and place one non TTU instruction (a `CS2R` clock read) inside the group.                                                          | A3: legal sequence grammar; where the tool may and may not insert code.                      |
| Interface_4 | Lanes                       | Vary active masks and divergent rays within a warp.                                                                                                    | A4: per ray vs per warp cost; whether warp splitting can help.                               |
| Interface_5 | Sessions                    | Multiple `TTUGO`s in one session; several queries before any `TTULD`; `/DUAL`; read SR133 around each step.                                            | A5: whether a warp can overlap queries (latency hiding); session model for the tool.         |
| Interface_6 | Ray flags and multiple hits | Sweep flag bits from OptiX; test terminate on first hit, culling, and whether repeated `TTUGO`/`TTULD` can enumerate several hits.                     | A6: how "all hits" workloads (radius search, ICP) must be mapped.                            |
| Interface_7 | Cache control               | Time traversals before and after `TTUCCTL.IVALL`.                                                                                                      | A7, and a tool for cold measurements in later phases.                                        |

### Phase 2: BVH format and scene setup

| ID    | Name                      | Method                                                                                                                                                                                                                   | Informs                                                                                    |
| ----- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| BVH_1 | Node format decode        | Build BVHs with `optixAccelBuild` for scenes of known geometry, dump device memory, and decode node headers, boxes, child pointers and leaf blocks; confirm with Cesar's builder.                                        | B1: the format our builder must write and our replay tool must read.                       |
| BVH_2 | Controlled builder        | Write a builder that emits valid BVHs with exact depth, width, leaf size, box overlap and memory footprint, plus a CPU reference traverser.                                                                              | Every later benchmark.                                                                     |
| BVH_3 | Custom primitives         | AABB primitives whose intersection runs on the SM; measure the round trip.                                                                                                                                               | B2: whether non triangle workloads can use the RT core cheaply.                            |
| BVH_4 | Instancing                | Two level scenes with transforms.                                                                                                                                                                                        | B3: scene organization options for the DSL.                                                |
| BVH_5 | Start at an internal node | Write an internal node's address into the root slot. Compare against a root start for rays inside and outside that subtree's box: identical hits? Time saved per skipped level? Behavior when the ray exits the subtree. | B4: whether subtree or treelet queues work on today's hardware, and the latency they save. |
| BVH_6 | Address limits            | Place BVHs at different addresses and alignments, up to the largest scene that fits.                                                                                                                                     | B5: allocator requirements for the tool and the DSL runtime.                               |

### Phase 3: latency of a single traversal

|ID|Name|Method|Informs|
|---|---|---|---|
|Latency_1|Fixed overhead|Reproduce EoSS Table 4 on SM89; time an empty scene traversal.|`t_fixed`.|
|Latency_2|Depth sweep|One ray, trees of depth 1 to 12 (EoSS went to 4), one leaf on the path.|`t_node` per level.|
|Latency_3|Width and visits|Fixed depth, vary branching width; separately, vary how many nodes a ray must visit (overlapping siblings) at fixed depth.|Whether cost follows depth or nodes visited.|
|Latency_4|Leaf size|Vary primitives per leaf on the hit path.|`t_prim` for triangles.|
|Latency_5|Hit count|Rays that report 1 to many hits (from Interface_6's mechanism).|`t_hit`; the quantitative version of "few hits per ray".|
|Latency_6|Cold vs warm|Same ray repeated, with and without `TTUCCTL` between runs.|`t_node` split by cache state.|

### Phase 4: throughput and concurrency

|ID|Name|Method|Informs|
|---|---|---|---|
|Throughput_1|In flight capacity|Throughput vs warps per SM (1 to the maximum resident); apply Little's law.|`C_inflight`.|
|Throughput_2|Active lanes|Throughput vs active lanes per warp.|Per ray vs per warp cost; warp splitting.|
|Throughput_3|Queries per warp|If Interface_5 allows several queries per warp, measure the throughput gain.|Latency hiding strategy.|
|Throughput_4|Overlap with SM work|Insert ALU and memory work between `TTUGO` and the result consumer, and run a memory heavy kernel alongside.|The overlap term of `T_kernel`; whether the TTU and LSU contend.|
|Throughput_5|Coherence|Throughput for coherent, partially coherent and random rays over the same scene.|Batching and reordering optimizations.|
|Throughput_6|SM scaling|Throughput vs number of active SMs (1 to 142).|Shared bottlenecks (L2, DRAM).|
|Throughput_7|OptiX overhead|Same scene and rays through OptiX vs our direct TTU kernel.|The "remove OptiX overhead" optimization; how much frameworks cost.|
|Throughput_8|RT cores per TPC|Pin traversal warps to chosen SMs (blocks exit unless `%smid` matches). Compare saturated throughput on one SM, on two SMs of the same TPC, and on two SMs in different TPCs; sweep all SM pairs to find which pairs contend.|D5: whether the two SMs of a TPC share one RT core; the true SM to TPC map.|

### Phase 5: memory hierarchy

|ID|Name|Method|Informs|
|---|---|---|---|
|Memory_1|Fetch path|Run traversals while an SM kernel thrashes L1 or L2; read L1/L2 hit counters.|Where nodes are cached.|
|Memory_2|Node cache size|Repeated rays over a growing set of distinct nodes (a pointer chase inside the TTU); look for latency steps.|Capacity of any TTU cache, for `t_node(cache state)`.|
|Memory_3|Bytes per traversal|DRAM and L2 counters per ray for known node counts; BVHs from 1 MB to beyond L2.|`bytes_per_ray`; footprint cliffs.|
|Memory_4|Locality|Same rays issued grouped by subtree vs randomly interleaved.|Direct evidence for subtree queues.|
|Memory_5|TPC sharing|Map SM IDs to TPCs (`%smid`). Warm a BVH with traversals on one SM, then time traversals and plain loads of the node addresses on (a) the same SM, (b) its TPC sibling, (c) an SM in another TPC. Repeat while thrashing the sibling's L1.|E5: whether the TPC pair shares node data, and whether scheduling paired SMs on the same subtree helps.|

### Phase 6: traversal behavior

|ID|Name|Method|Informs|
|---|---|---|---|
|Traversal_1|Order and early exit|Scenes where the closest hit sits in the near or far child; time the difference.|Traversal order; value of BVH layouts that put likely hits first.|
|Traversal_2|Stack limits|Very deep and very unbalanced trees; look for latency jumps.|Builder rules (maximum depth).|
|Traversal_2a|Return trigger without a hit|Scenes where the ray hits nothing but must visit many nodes (deep or wide misses, overlapping empty boxes); see whether a query ever returns pending entries with no hit, and at what depth or node count. Find the internal stack depth, if there is one.|F2: when the RT core gives up and hands work back to the SM.|
|Traversal_2b|Which pending entries come back|Scenes with more than two closer pending subtrees at known distances; decode which two entries are returned (nearest, farthest, last pushed) and where the rest go.|F2: what the SM must keep, and the true stack model.|
|Traversal_2c|Queries vs hits at known depths|Place a known number of hits at known depths and distances; count queries per ray.|The queries per ray term of the model.|
|Traversal_2d|Ray flags and returns|Repeat 2a to 2c with terminate on first hit, culling and opaque flags.|Whether shadow style rays avoid resumes.|
|Traversal_2e|Cost per return|Latency of each query in a resumed ray vs the first; SM work between queries.|`t_resume` for the model.|
|Traversal_3|Overlap and degenerate geometry|Heavily overlapping boxes, flat boxes, tiny boxes (RTNN style), walls (RAGRT style).|Why some published mappings are slow; DSL rules.|
|Traversal_4|Precision|Rays grazing box and triangle edges; compare against exact arithmetic.|Whether results can be trusted for non graphics queries.|

### Phase 7: across architectures

|ID|Name|Method|Informs|
|---|---|---|---|
|Arch_1|Rerun everything|Repeat every microbenchmark and the BITTU validation on the RTX 4070 Ti and the other generation GPU (SM89 encodings carry over to the 4070 Ti; the other GPU needs its own from Setup_2).|Which model parameters are architectural constants and which scale with SM count and L2 size.|



# Binary instrumentation tool

Working name: **BITTU**

## Intended use
While the microbenchmarks can measure the TTU in isolation, a binary instrumentation tool is needed to measure performance inside real programs to determine behavior such as how many traversals they issue, how long each takes, how many hits each ray gets, how much of the kernel's time is traversal, and which rays and subtrees it touches. This information will be crucial in developing and validating a performance model.

## What it should do

| Capability                                                                                                                                                                                                                  | How                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Find TTU code** in any cubin, including JIT dumps: list sessions and fused groups, their shapes, and which TTU offsets they read and write.                                                                               | Decode with the EoSS ISA JSON (the standard disassemblers hide TTU instructions).                                               |
| **Count** traversals per warp, per kernel, per SM.                                                                                                                                                                          | Probe before each `TTUOPEN`.                                                                                                    |
| **Time** each traversal end to end.                                                                                                                                                                                         | Clock before `TTUOPEN`, clock after the first instruction that consumes a `TTULD` result. Subtract a calibrated probe overhead. |
| **Capture rays**: the values stored by each `TTUST` (origin, direction, tmin/tmax, flags), sampled.                                                                                                                         | Read the source registers before the fused group (they already hold the values).                                                |
| **Capture results**: hit, primitive, t, per ray.                                                                                                                                                                            | Read the `TTULD` destination registers after the group.                                                                         |
| **Derived metrics**: hits per ray, active lanes per traversal, fraction of kernel time spent waiting on traversal, traversals over time per SM.                                                                             | Host side analysis of the above.                                                                                                |
| **Replay**: run captured rays offline through a software traverser over the dumped BVH to recover nodes visited, depth reached, overlap, and which subtrees each ray touched (the hardware probably does not report these). | Needs the decoded node format (BVH_1). Validate replay against measured latency.                                                |
| **Hardware counters** per kernel as ground truth.                                                                                                                                                                           | CUPTI Profiling API or `ncu` (RT, L1, L2, DRAM metrics).                                                                        |

Modes: 
- **count** (cheapest)
- **sample** (1 in N warps or traversals)
- **trace** (everything, for small runs).

## Tool constraints

1. **Never insert code inside a fused group.** `TTUMACROFUSE(n)` fuses the next n instructions, so any inserted instruction breaks the group and hangs the GPU. This also rules out instrumenting `TTUMACROFUSE` itself, because trampoline style tools (NVBit included) relocate the instrumented instruction into a stub and follow it with a branch, which would be counted as fused. **Legal probe points: before `TTUOPEN`, and after the last instruction of the group.** Interface_3 tests whether raising the count to N+1 lets a single inline instruction sit inside the group; even if it does, an NVBit probe is far more than one instruction, and timing after the group already captures the full latency because the first consumer of a `TTULD` result is what waits.
2. **Do not disturb result scoreboards.** `TTULD` writes through a variable latency scoreboard; probes that read its destination registers must wait on it (place them after the original consumer, or add the matching wait).
3. **Respect session state.** Never issue anything that could open a second session or skip `TTULD.CLOSE`.
4. **Account for overhead.** Probes add cycles. Every timing mode needs a calibration run with empty probes, and results should be reported with and without the tool.

## Building on prior work
NVBit (NVIDIA Labs, MICRO 2019) is a dynamic binary instrumentation framework for NVIDIA GPUs, similar in spirit to Pin for CPUs. When a kernel loads, an NVBit tool can inspect every SASS instruction and inject calls to its own device functions before or after any of them, with access to register values. This lets it count, time and capture data inside unmodified binaries without recompiling them. Its core is closed source, so if its decoder does not understand TTU instructions we cannot fix it, and it is unknown whether it sees the kernels OptiX's JIT compiler produces. CUPTI is NVIDIA's profiling interface. It provides hardware counters per kernel (on SM120 this includes RT core counters such as `rtcore__rays.sum`; ncu lists none for AD102, see Findings; plus L1, L2 and DRAM metrics) and activity traces. Its SASS Metrics API also patches SASS, but only to collect NVIDIA's predefined metrics: it cannot run custom probes or read registers. Its PC sampling and Metric APIs do not support OptiX applications, while its Profiling API does.

BITTU will be an NVBit tool. When a module loads, BITTU decodes each function's instructions with a TTU decoder generated from the EoSS SM89 ISA JSON (the TTU opcode masks and field layouts exported as a C++ header), since NVBit itself may not recognize TTU instructions; if NVBit's SASS text hides them, BITTU reads the raw code bytes instead. From this it finds every session and fused group, then inserts probes only at legal points (before `TTUOPEN` and after the group) to count traversals, time them and capture ray and result registers into device buffers that the host collects. CUPTI's Profiling API (or `ncu`) runs alongside to collect per kernel counters as ground truth for BITTU's counts. EoSS's static patcher builds the microbenchmarks and gives kernels with known answers to validate BITTU against. Before committing to NVBit, four quick tests decide it: whether it lists TTU instructions correctly, whether empty probes around a fused group run without hanging, whether a probe reads the `TTUST` source registers correctly, and whether OptiX's JIT kernels are visible at all. If the last one fails, OptiX programs fall back to clock reads around `optixTrace` in our own raygen code.

# Repository layout
EoSS files stay untouched for easy merging. The public repo holds only code and data, in new top level directories:

```
ttu/
  encodings/ttu_sm89.json
  bvh/                    builder, decoder, CPU reference traverser
  harness/                CUDA kernels with patchable TTU sequences
  bench/<ID>/             one folder per benchmark: run.py, analyze.py, short usage README
  results/                raw data (gitignored; archived separately)
bittu/                    the instrumentation tool (BITTU)
model/                    model fitting scripts and notebooks
```

`MASTER_PLAN.md` (this document, repo root) is public, including its Findings section. The log, the protocol spec, per benchmark findings and all dumps stay in `notes/` (excluded via `.git/info/exclude`), because they hold working notes and NVIDIA code:

```
MASTER_PLAN.md            this document (public)
notes/
  RETTU Log.md            dated record of every change, test and finding
  protocol.md             living spec of the TTU interface (offsets, fields, rules)
  bvh_format.md           decoded node and leaf formats
  findings/<ID>.md        one page per benchmark: question, hypothesis, setup, result, what it means
  setup4/                 raw OptiX code dumps (NVIDIA code, never in git)
```

Each benchmark folder has `run.py` (writes CSV and JSON metadata) and `analyze.py` (produces the figure and the numbers that go into `notes/findings/<ID>.md`). Its README only says how to run it.

# Findings
Short dated list of what we have learned, in words only. Details and evidence are in the private notes. Add each new finding here with a date.

- **2026-11-09, TTULD.CLOSE is bit 74 on SM89.** Read from real SM89 OptiX code (and from a Turing warmup kernel inside the OptiX library). Bit 72, which Cesar's SM120 file gives as the close bit, is the `.DUAL` flag of TTUOPEN; no TTULD we have seen sets it.
- **2026-11-09, fuse count rule.** TTUMACROFUSE N equals the number of TTU instruction slots after it, including predicated ones that do not execute (OptiX's main traversal group has N = 17 with 6 predicated stores). The CLOSE is never predicated. In every group we have, the N instructions are contiguous TTU instructions, so we cannot yet tell whether non TTU instructions inside a group would count (Interface_3).
- **2026-11-09, SM89 traversal sequence.** OptiX's ray traversal on Ada is one fused group: TTUOPEN.DUAL, TTUMACROFUSE 17, ten TTUSTs (offsets 0xC0, 0xD0, 0x200 or 0x100, 0x180, 0x190, 0, 0x40, 0x50, 0x80), TTUGO, six TTULDs (0x300, 0x310, 0x320, 0x380, 0x3C0, 0x3D0, the last one closing). Stores set a read scoreboard and loads a write scoreboard, unlike Cesar's sequence. The smallest legal sequence NVIDIA ships is OPEN, FUSE 2, GO, LD.CLOSE with no store.
- **2026-11-09, store sources.** 0x40 and 0x50 carry origin plus tmin and direction plus tmax; 0xC0 and 0xD0 repeat them. The other stores are built from ray flags, the visibility mask, ray time and one OptiX launch constant. Raygen packs flags and mask into one word (flags << 16, mask << 8, then 1).
- **2026-11-09, OptiX always adds an instance level.** Even for a single GAS, store 0x100 points to a small instance node that OptiX builds itself (identity 3x4 transform), sitting next to its program dispatch table; the GAS root goes in store 0x180. So the RT core starts at an instance level even with one object. Setup_3 points 0x100 at OptiX's own node in the same process; building our own instance node is part of BVH_1. This is also why Cesar's 4 store sequence has no root pointer on SM89.
- **2026-11-09, shadow rays skip stores (future latency test).** The predicated stores depend on one test: they are skipped only when a ray has both TERMINATE_ON_FIRST_HIT and DISABLE_CLOSESTHIT (the usual shadow ray). Shadow style any hit queries send less to the RT core and may be cheaper; a Phase 3 latency benchmark should compare them with closest hit rays. Relevant to the model's "few hits per ray" question.
- **2026-11-09, some store words come from registers nothing initializes, and registers are not cleared at launch.** Half of 0x180 and all of 0x190 come from registers that OptiX never writes before the traversal in our pipeline. A plain CUDA test (fill the whole register file, then read every register at the start of the next kernel) found no zeroing: registers start with leftover or unpredictable values, not 0. So OptiX sends arbitrary data in those words and traversal is still correct, which suggests the RT core ignores them in this mode. Interface_1 should test that directly.
- **2026-11-09, the BVH does not need the low 4 GB.** OptiX's GAS on SM89 sat near 128 TiB and traced correctly; pointers are full 64 bit addresses in 64 or 128 byte units. The paper's low 4 GB rule may hold for Cesar's format or SM120; BVH_6 tests it.
- **2026-11-09, first GAS (one triangle).** 2304 bytes: a header with section offsets, a root node with 12 child slots (fits David's 12 wide result), what looks like a compressed triangle leaf, and the GAS's own GPU address inside it (not position independent, so it cannot be copied between processes).
- **2026-11-09, OPEN: a second copy of the traversal code at fnaddr minus 0x1C00.** 14 TTU instructions with no OPEN or FUSE before them. It is part of a larger copy of the traversal function with raygen code written over its middle, so it is most likely stale code left in memory, but it could be a resume path entered by an indirect branch. Not settled.
- **2026-11-09, no ray counter on AD102.** `rtcore__rays.sum` does not exist for AD102 in ncu 2024.1 or 2025.1, and no metric has a ray related name. Two stall metrics have a "non RTCORE" part, so their difference shows RT core stalls, not rays. Correctness checks use a CPU reference and an in process OptiX launch.
- **2026-11-09, NVIDIA hides OptiX traversal from every official tool.** Hooking the driver's code loaders saw nothing (OptiX loads code through a private interface); the OptiX disk cache holds only user programs (traversal is added at pipeline link); CUPTI and ncu list only user functions; cuda-gdb is blocked by the server's ptrace setting. Reading code from GPU memory with plain loads worked. This is the gap BITTU is meant to fill.
- **2026-11-09, possible EoSS issue: SM89 JSON lacks SEL with an immediate.** Opcode 0x807 (SEL Rd, Ra, imm, Pp) appears 20 times in OptiX's traversal code but is not in the SM89 ISA JSON (the register and constant forms are). The Decoder also prints constant operands without bank and offset and drops some immediates. Worth reporting upstream once checked against SM120.
- **2026-11-09, first TTU traversal from a plain CUDA kernel on SM89 (Setup_3).** One warp ran NVIDIA's own fused traversal group, unchanged, with registers we filled, against an OptiX built one triangle GAS and OptiX's instance node in the same process. No hang or fault. Hit and miss rays give clearly different results, the same in all 32 lanes. On a hit the loads return the barycentrics, the hit distance, the address of the hit triangle block and the hit instance; on a miss those words are all ones or zero. Two hit values (t and one barycentric) come back with the sign bit set, not yet explained.
- **2026-11-09, the RT core ignores the uninitialized store words here.** Random data instead of zeros in those words gives identical traversal results.
- **2026-11-09, what the main TTULDs return (one triangle).** A hit returns the hit distance negated (sign bit always set), the two barycentrics exactly to the float bit, with the sign of the first one marking which face was hit (set for a back face hit by OptiX's convention), a hit flag, the address of the hit triangle block, and the hit instance. A miss returns all ones in the distance and barycentric words and zeros elsewhere. One word is always negative zero on a hit and is still unexplained.
- **2026-11-09, first measured traversal latency on SM89.** One warp, one ray per lane, one triangle under OptiX's instance level, BVH already in cache: about 1,640 cycles from TTUOPEN to the last result (median of 30, spread about 10 cycles, all lanes equal), about 1,560 after subtracting the cost of the clock reads and an empty group slot. EoSS reports about 1,600 cycles at depth 1 on SM120, so the two are in the same range, but the setups differ (architecture, BVH format, instance level, timing method), so this is not yet a like for like comparison.
- **2026-11-09, face flag confirmed.** OptiX's own front face query agrees: the sign bit on the first barycentric is set exactly for back face hits.
- **2026-11-09, primitive index and leaf slot.** One TTULD word carries the hit primitive index in its low bits (checked up to 10,368) under a 3 bit flag field whose value changes for large scenes (meaning unknown). The leaf address word also carries which triangle slot inside the leaf block was hit. The word we guessed might be the primitive index is always zero.
- **2026-11-09, cold vs warm.** Reading 256 MB (more than the 96 MB L2) right before a timed launch raises one traversal of the one triangle scene from about 1,640 to about 2,630 cycles; an empty group slot is unaffected, so the cost is BVH data coming from DRAM.
- **2026-11-09, GAS node format (first decode).** 128 byte internal nodes with 12 child slots; one slot field marks empty, internal and leaf children; internal children of a node are stored together and found from one offset field; a header word gives the node count, and walking the tree with these rules reaches exactly that count in all 33 dumps (1 to 30,000 triangles). Leaves hold several triangles.
- **2026-11-09, latency vs depth.** One warp, with the BVH depth measured from the GAS dump each time: warm about 1,640 / 1,910 / 2,315 cycles at 1 / 2 / 3 internal levels (plus OptiX's instance level), about 2,640 for a scene whose ray is likely at level 4; cold about 2,650 / 3,230 / 3,890. Each level adds roughly 270 to 400 cycles warm and 600 to 750 cold. EoSS's SM120 numbers grow faster per level (about 1,600 to 4,400 over 4 levels); the setups differ, so this is a first comparison only.
- **2026-11-09, Throughput_1 on one SM.** With 1, 2, 4, 8, 16 warps on one SM (one ray per lane, same ray, BVH warm), the RT core finishes 0.020, 0.032, 0.047, 0.068 and 0.085 rays per cycle, about 12 cycles per ray at 16 warps, while each warp's latency rises from about 1,640 to about 2,960 cycles. Warps reach the group at staggered times because of our setup code, so the 16 warp rate is a lower bound.
- **2026-11-09, one SM throughput with aligned starts (coherent rays).** A blocking barrier plus a short gap lines up all warps within about 30 cycles (a warp issues a few instructions past a barrier before it stalls). The TTU group's register footprint caps one block at 20 warps per SM (96 registers per thread; the declared count must exceed the highest register used by at least 3). From 1 to 20 warps, with every lane tracing the same ray, throughput rises from 0.020 to 0.090 rays per cycle while per warp latency rises from about 1,640 to about 4,900 cycles; by Little's law about 14 warps (about 440 rays) are in flight at 20 warps. Warps finish at a steady 5.7 to 7.4 cycles per ray. Throughput is still rising at the ceiling, so the RT core is not saturated there; remapping the group to fewer registers would allow more warps.
- **2026-11-09, truly warm traversal.** Tracing the same ray twice in one kernel and timing the second: about 780, 830 and 870 cycles at 1, 2 and 3 internal levels, so about 43 cycles per level when everything is cached, and about 740 cycles of fixed cost. A traversal that is warm only across kernel launches costs about 1,640 to 2,300 cycles.
- **2026-11-09, memory latency and the per level cost.** A plain pointer chase on one SM gives L1 about 41 cycles, L2 about 292 to 298, DRAM about 616 to 654; L1 contents do not survive a kernel boundary. Each extra BVH level costs about one dependent access at the level where its node sits (about 43 in kernel = L1, 270 to 400 across launches = L2, 580 to 750 cold = DRAM), so the RT core's node fetches behave like loads through L1.
- **2026-11-09, Throughput_8: SM pairs share a traversal resource.** With 20 warps on each running SM (coherent rays), two SMs 2k and 2k+1 running at the same time each drop to about 70% of their solo rate (pair total about 1.45 times one SM), while two SMs further apart (0 and 2, 2 and 4) each keep their full rate. This fits one RT core (or another shared path) per TPC, as EoSS claims; the pairing of SM numbers into TPCs is inferred from the block placement pattern and this result, not documented.
- **2026-11-09, coherence (first test).** Giving every lane a different ray, to a different triangle and a different BVH path in a 144 triangle scene, changes neither throughput nor latency measurably (within about 3%) from 1 to 20 warps. The scene fits in L1 and all rays are parallel, so this shows that path divergence alone is free here; incoherence through a large memory footprint is not yet tested.
- **2026-11-09, coherence matters once the BVH leaves L1.** With random ray origins and directions in a cloud of random triangles, each lane's ray takes its own path. For a 3 KB BVH, random rays are as fast as coherent ones. For BVHs of 654 KB, 6.5 MB and 195 MB (bigger than L1, up to twice L2), random rays get 4 to 6 times less throughput than rays shared by all lanes at 20 warps, and 2 to 3.5 times higher latency at 1 warp. With random rays, 20 warps can deliver less throughput than 8 (cache thrashing).
- **2026-11-09, one TTU query is not always a whole ray.** In large scenes the last two loads of the group return values shaped like BVH node pointers, and in a few lanes the returned hit is real but not the closest one OptiX finds. This suggests the RT core hands back unexplored nodes and the SM must query again, which would also explain the input words OptiX leaves uninitialized on a first query. To be tested by resubmitting those words.
- **2026-11-09, one SM saturates (coherent rays).** After moving the TTU group to fewer registers (only register fields changed, checked against NVIDIA's bytes), one SM can hold 32 warps. The steady completion rate levels off at about 0.134 rays per cycle (7.4 cycles per ray) from 20 to 32 warps.
- **2026-11-09, Throughput_8 at saturation.** With 32 warps per SM, two sibling SMs (2k, 2k+1) each fall to 60 to 80% of their solo steady rate while two SMs further apart keep theirs. But a sibling pair delivers 1.25 to 1.4 times one saturated SM, not the same amount, so a single shared RT core with one SM's capacity per TPC is not confirmed; the one SM limit may be per SM, or the partial overlap of the two runs may inflate the pair total.
- **2026-11-09, L1 is hard to evict on purpose.** In a plain CUDA control, strided loads from one warp (up to 4 MB) never evicted a warmed 16 KB working set from L1, and a second random chain evicted it only partly and not monotonically with size. L1 replacement is not simple LRU; the Memory_1 test needs a proven eviction method first.
- **2026-11-09, rays can need several queries, and OptiX loops to resume them.** OptiX's traversal code resumes a ray by running the same TTU group again with the best hit distance as the new maximum and the returned pending nodes as input. Doing the same from our kernel gives the CPU's closest hit in every checked lane, for scenes up to 3 million triangles. Rays need 1.5 to 2 queries on average in moderately overlapping scenes and about 3.8 (up to 9) when triangles overlap heavily; scene size and ray direction matter little. Counting all queries, a finished ray costs about 2 to 10 times one query. (This corrects the earlier reading of "up to 4 node pointers".)
- **2026-11-09, a query hands back at most two pending entries.** With a known number of triangles stacked on a ray, the first query already returns the closest hit and at most two pending entries, each an entry distance plus a node pointer; the follow up query then ends at once because both entries lie beyond the hit. Extra queries cost real work only when the first hit found is not the closest.
- **2026-11-10, "2 pending entries" is not yet the stack depth.** A query returns at most 2 pending entries in every scene so far, but that is only the return format seen on a hit; the RT core may keep a deeper stack internally and return 2 only when it stops early. Traversal_2a to 2e test this.
- **2026-11-10, BVH_5: traversal can start at any internal node, with no new instruction.** Putting an internal node of the GAS in the root word (store 0x180, same flags as the root) makes the RT core search only that subtree. In 37 runs (flat grids of 144 and 2,500 triangles and a four layer scene, starts one and two levels below the root, straight and oblique rays) every ray returned exactly the CPU's closest hit among the triangles inside the start node's box, and a miss otherwise, including 92 rays that cross the start node's box and would hit something outside it. Rays that need more than one query resume correctly from a subtree start. OptiX's instance level stays in front.
- **2026-11-10, each skipped level saves one node fetch.** Starting one or two levels down saves 295 to 363 cycles per level when the BVH is in L2 and 610 to 650 when it comes from DRAM, the same as one dependent L2 or DRAM access in a pointer chase. A start two levels down in a 2,500 triangle scene costs about what a one level scene costs. For treelet or subtree queues this means the saving is real but per level, and a ray that misses inside its subtree learns nothing about the rest of the tree, so a correct closest hit still needs the rest searched. Memory_4 tests whether grouping rays by subtree adds a cache benefit on top.
- **2026-11-10, GAS node boxes decoded; empty slot rule corrected.** Each child slot holds an 8 bit lower and upper bound per axis relative to the node's origin, scaled by a per axis power of two in the node header; the decoded boxes match the scene geometry. The earlier rule for empty slots was wrong for children at the top of a node's z range (it dropped real nodes in a layered scene); with the corrected rule every saved dump walks to the node count in its header. OptiX also places nodes in a different memory order each run, while the tree itself is the same, so nodes are named by their slot path from the root.
