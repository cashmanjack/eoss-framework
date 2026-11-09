# Introduction
The goal of this project is to characterize the NVIDIA RT core which is expressed as TTU "Tree Traversal Unit" instructions at the SASS level. The RTX 6000 Ada (SM89) will be the primary target hardware for this effort. This project will consist of a suite of microbenchmarks designed to characterize a performance model of the RT core and a binary instrumentation tool to measure and validate performance characteristics.

The RAGRT project showed us that mapping a problem onto the RT cores can make the performance slower despite expectations. Currently the only way to find this out is to build the entire mapping then measure it. Search dominating runtime, data being natively 3D, scenes being sufficiently large, rays having few hits, and data being irregular enough that grids lose are the expected conditions which would justify RT speedup but these conditions are largely qualitative and not enough.

This project aims to replace the existing approach of reasoning about mapping performance with a performance model of the RT core based on actual measurements of the hardware, with the tooling needed to evaluate RT core performance in real programs.

# EoSS
EoSS is a framework which runs arbitrary SASS instructions on real GPUs bypassing the NVIDIA assembler. EoSS uses the readable ISA description "Rosetta String" from NVIDIA's tooling and parses it into a per architecture JSON. This includes every instruction class along with its opcode, operand bit fields, modifiers, scheduling properties and the constraint expressions that make an encoding legal. From this EoSS encodes using a constraint aware generator which builds valid 128 bit encodings for any class which includes the control bits, stall count, yield, scoreboard read, write, and wait masks. A pure Python CUDA ELF parser finds the .text sections in a compiled binary, locates the kernel code by byte anchors and patches instructions in place. Then, an nvcc compiled kernel with thousands of patchable "slots" bracketed by clock reads has its instructions patched by EoSS. EoSS then runs the binary and reads back the per slot timings. Finally EoSS provides a CLI tool and a SASS editor for testing and experimentation.

## EoSS Insights
EoSS discovered eight TTU instruction classes: TTUOPEN, TTUMACROFUSE, TTUST, TTUGO, TTULD, TTULD.CLOSE, TTUCLOSE, and TTUCCTL. 

TTU instructions are hidden by cuobjdump and nvdisasm and unreachable to PTX as only OptiX's JIT compiler emits them. The EoSS paper runs them from a plain CUDA compute kernel. The standard TTU sequence always starts with TTUOPEN then TTUMACROFUSE N and then N fused instructions including TTUST, TTUGO, TTULD, and TTULD.CLOSE. 

If the fuse count is not equal to the correct number of fused instructions then the GPU hangs. Sessions are per warp and are tracked using SR_TTU_TICKET_INFO (SR133). A second TTUOPEN on an open session faults. Omitting TTULD.CLOSE permanently hangs the GPU. The driver closes open sessions during preemption.

The BVH must live in the low 4GB of the address space as some pointers are 32 bit. Child pointers are absolute addresses.

Measured on SM120, traversal latency is about 1,600 cycles at depth 1, then 2,100, 2,900, and 4,400 cycles at depths 2, 3, and 4, and nearly flat in triangle count. Per instruction issue costs are TTUOPEN 6, TTUMACROFUSE 6, TTUST 9, TTUGO 5, TTULD 5, TTULD.CLOSE 62, and TTUCCTL at 10 cycles. Validation is done via the Nsight Compute counter rtcore__rays.sum.

## EoSS missing info
The README refers to `TTU/`, `Contention/`, `OSIRIS/`, `VisualModifier/` and `Unintentional/` directories, but they are not present. We need BVH builder and encoder, and the compute mode ray tracer ("Ecce Mono"), the OptiX JIT cubin dump, the in memory patch tool, and the TTU timing harness.

TTU encodings are packaged only for SM120 and our GPU is SM89. SM89 ISA JSON has all eight TTU classes but TTUST's second source and TTULD's second destination are ordinary GPR pairs Rc, while SM120 uses a uniform register field (Ra_URc). This means SM120 bytes cannot be directly copied so SM89 encodings need to be generated with EoSS encoder.

The README lists CUDA 13.0+ and ncu with counter permissions as prerequisites while docs/results.md validated SM89 using CUDA 12.8. Triton has CUDA 12.4 and no counter permissions for non admins.

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

## E: TTU memory hierarchy 
1. Which path node fetch take (L1, dedicated TTU cache, L2) and if SM memory traffic interferes - PM
2. Capacity and behavior of any TTU node cache - PM OPT
3. Bytes fetched per node and per traversal - behavior when BVH exceeds L2 (96MB on the RTX 6000 Ada) - PM DSL
4. Locality - does grouping rays that traverse the same subtree measurably reduce latency or DRAM traffic? - OPT

## F: Traversal behavior
1. Traversal order (nearest child first? fixed order?) and early termination as tmax shrinks - PM OPT
2. Stack depth limits, short stack restarts, costs of very deep trees - PM
3. Sensitivity to overlapping and degenerate boxes - PM DSL
4. Numerical behavior - conservative box tests, watertight triangle tests, and precision of t - BIT

## G: Architecture generality
1. Which findings hold on SM86 and SM120 as well as SM89 - PM DSL

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
| Setup_1 | Toolchain          | Install CUDA 12.8 on triton (matching EoSS's SM89 validation), check the driver supports it, then run `eoss gpu info`, the test suite and `eoss full --sm 89`. | Whether the EoSS pipeline works on our GPU.                |
| Setup_2 | SM89 TTU encodings | Generate all eight TTU classes for SM89 with the EoSS encoder, diff them field by field against the SM120 canonical sequence, and write `ttu_sm89.json`. | Every benchmark below.                                     |
| Setup_3 | First traversal    | One ray, one triangle, the canonical sequence under timeout; check the hit against the CPU reference and `rtcore__rays.sum`.                              | Confirms the SM89 port works.                              |
| Setup_4 | OptiX dumps        | Dump the JIT cubins and BVHs of the OptiX SDK samples on SM89 and extract their TTU sequences (Cesar's tool or a rebuild of it).                         | The full store and load protocol to decode in Interface_1, Interface_2 and BVH_1. |

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

### Phase 5: memory hierarchy

|ID|Name|Method|Informs|
|---|---|---|---|
|Memory_1|Fetch path|Run traversals while an SM kernel thrashes L1 or L2; read L1/L2 hit counters.|Where nodes are cached.|
|Memory_2|Node cache size|Repeated rays over a growing set of distinct nodes (a pointer chase inside the TTU); look for latency steps.|Capacity of any TTU cache, for `t_node(cache state)`.|
|Memory_3|Bytes per traversal|DRAM and L2 counters per ray for known node counts; BVHs from 1 MB to beyond L2.|`bytes_per_ray`; footprint cliffs.|
|Memory_4|Locality|Same rays issued grouped by subtree vs randomly interleaved.|Direct evidence for subtree queues.|

### Phase 6: traversal behavior

|ID|Name|Method|Informs|
|---|---|---|---|
|Traversal_1|Order and early exit|Scenes where the closest hit sits in the near or far child; time the difference.|Traversal order; value of BVH layouts that put likely hits first.|
|Traversal_2|Stack limits|Very deep and very unbalanced trees; look for latency jumps.|Builder rules (maximum depth).|
|Traversal_3|Overlap and degenerate geometry|Heavily overlapping boxes, flat boxes, tiny boxes (RTNN style), walls (RAGRT style).|Why some published mappings are slow; DSL rules.|
|Traversal_4|Precision|Rays grazing box and triangle edges; compare against exact arithmetic.|Whether results can be trusted for non graphics queries.|

### Phase 7: across architectures

|ID|Name|Method|Informs|
|---|---|---|---|
|Arch_1|Rerun core set|Latency_1, Latency_2, Throughput_1, Memory_3 and BVH_5 on SM86 and SM120 where we can get access.|Which model parameters are architectural constants.|



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
NVBit (NVIDIA Labs, MICRO 2019) is a dynamic binary instrumentation framework for NVIDIA GPUs, similar in spirit to Pin for CPUs. When a kernel loads, an NVBit tool can inspect every SASS instruction and inject calls to its own device functions before or after any of them, with access to register values. This lets it count, time and capture data inside unmodified binaries without recompiling them. Its core is closed source, so if its decoder does not understand TTU instructions we cannot fix it, and it is unknown whether it sees the kernels OptiX's JIT compiler produces. CUPTI is NVIDIA's profiling interface. It provides hardware counters per kernel (including RT core counters such as `rtcore__rays.sum` and L1, L2 and DRAM metrics) and activity traces. Its SASS Metrics API also patches SASS, but only to collect NVIDIA's predefined metrics: it cannot run custom probes or read registers. Its PC sampling and Metric APIs do not support OptiX applications, while its Profiling API does.

BITTU will be an NVBit tool. When a module loads, BITTU decodes each function's instructions with a TTU decoder generated from the EoSS SM89 ISA JSON (the TTU opcode masks and field layouts exported as a C++ header), since NVBit itself may not recognize TTU instructions; if NVBit's SASS text hides them, BITTU reads the raw code bytes instead. From this it finds every session and fused group, then inserts probes only at legal points (before `TTUOPEN` and after the group) to count traversals, time them and capture ray and result registers into device buffers that the host collects. CUPTI's Profiling API (or `ncu`) runs alongside to collect per kernel counters as ground truth for BITTU's counts. EoSS's static patcher builds the microbenchmarks and gives kernels with known answers to validate BITTU against. Before committing to NVBit, four quick tests decide it: whether it lists TTU instructions correctly, whether empty probes around a fused group run without hanging, whether a probe reads the `TTUST` source registers correctly, and whether OptiX's JIT kernels are visible at all. If the last one fails, OptiX programs fall back to clock reads around `optixTrace` in our own raygen code.

# Repository layout
EoSS files will be untouched for easy merging. New work lives in new top level directories:

```
docs/ttu/
  MASTER_PLAN.md          this document
  protocol.md             living spec of the TTU interface (offsets, fields, rules)
  bvh_format.md           decoded node and leaf formats
  findings/<ID>.md        one page per benchmark: question, setup, result, what it means
ttu/
  encodings/ttu_sm89.json
  bvh/                    builder, decoder, CPU reference traverser
  harness/                CUDA kernels with patchable TTU sequences
  bench/<ID>/             one folder per benchmark: README, run.py, analyze.py
  results/                raw data (gitignored; archived separately)
bittu/                    the instrumentation tool (BITTU)
model/                    model fitting scripts and notebooks
```

Each benchmark folder follows one template: `README.md` (question, hypothesis, method, expected outcome), `run.py` (writes CSV and JSON metadata), `analyze.py` (produces the figure and the numbers that go into `findings/`).
