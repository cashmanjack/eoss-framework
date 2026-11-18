# BITTU: a TTU aware instrumentation tool for NVIDIA RT cores

BITTU counts, times and records the tree traversal unit (TTU, the RT core's SASS level interface) groups inside GPU kernels, so that a performance model can be checked against what the hardware does. It is part of the RETTU project (Purdue ECE, Kulkarni group), built on EoSS (Baddouh, Payer, Rogers, ASPLOS 2027) and NVBit (Villa et al., MICRO 2019). Finding numbers (F...) below refer to the Findings section of `MASTER_PLAN.md` in the repository root.

## Status in one paragraph
Compute mode works for kernels with TTU code that we compile and run ourselves (decode, ray and result capture exact; timing exact to under 1 percent when a twin build with no-ops is available). It cannot instrument OptiX's own kernels, because NVBit never sees them (F12, F60); for OptiX programs the project uses a recorder (clock reads and counters written into the user's own device programs), which is not NVBit and whose patches are for third party programs and are not in this directory. Validated runs so far: one GPU (RTX 4070 Ti, SM89), the single level traversal group, one warp. No BITTU compute run on another GPU is recorded in the Findings.

## What is here
- `compute/`: the NVBit tool (`bittu.cu`, `inject_bittu.cu`, `bittu_rec.h`), the field table generated from the EoSS SM89 tables (`gen_tables.py` writes `ttu_tables.h`), a runner (`run_bittu.py`) and the timing check (`timing_fix.py`).
- `kill/`: the four feasibility tests (NVBit 1.8.1) with their own README (F60).

## What it does (compute mode)
1. NVBit silently drops TTU instructions from its instruction list (F60): the group appears only as one gap in the instruction offsets. The gap must match the offset and length of a group that our checker (`check_seq`, in `ttu/`) has already verified.
2. The group's bytes are read from a file cut from the checked binary (`BITTU_GROUP_FILE`) and decoded with the EoSS SM89 field tables. Anything other than exactly one TTUOPEN, one TTUMACROFUSE with the right count, TTUST, TTUGO and TTULD instructions, the close bit on the last TTULD only, and even register pairs is refused (not instrumented). This is a safety rule: a wrong group hangs the GPU (see the safety section of the project notes).
3. Probes are NVBit calls placed before the group and after the instruction that waits for its results; no TTU instruction is moved or touched. Per thread and group instance they record two clocks, two clocks at the same point (the probe cost), the eight ray registers, the load destination registers, the active masks and the instance count.
4. After every launch: summary (threads, instances, probe overhead, measured window, corrected window) and the first records.

Build and run (needs NVBit 1.8.1 from NVIDIA Labs, CUDA 12.8, CMake only for NVBit itself; NVBit is not included here):

    source ~/cuda128.sh
    bittu/compute/build.sh /dev/shm/bittu
    python bittu/compute/run_bittu.py BIN RAYS FILL OUT.json REPORT.txt      # BIN: a checked setup3 binary; run.py applies the safety checks

Environment: `BITTU_OUT` (report file), `BITTU_PRINT` (records printed), `BITTU_MODE=off` (decode only) or `empty` (the empty probe pair for the overhead subtraction), `BITTU_KERNEL`, `BITTU_GROUP_FILE`, `BITTU_GROUP_OFF`, `BITTU_GROUP_LEN`.
Safety: run only on an idle GPU, under `timeout`, smallest launch first, as in the project's rules; never run a TTU sequence that was not checked.

## Accuracy and overhead (all on an RTX 4070 Ti, one warp, the checked single level group)
| Item | Result | Finding |
|---|---|---|
| NVBit sees the TTU instructions | no: the 19 instruction group is a gap in its list (decoded by our tables instead) | F60 |
| Probes before and after a fused group | correct results, no hang | F60 |
| Decoding of the hidden group | exact: fuse count 17, six predicated stores, six loads, close bit on the last load only | F119 |
| Ray registers (8 words) and result registers (22 words) | exact in all 32 lanes, full warp masks, instance count right in all launches; results of the instrumented run equal the plain run | F119 |
| Cost of one probe | about 1,450 cycles clock to clock (1,441 to 1,459) | F119, F122 |
| Raw probe to probe window around the group | 11,335 cycles against 1,613 for the group measured by the harness clock without the tool: the raw window is about 7 times too long and not usable as a latency | F119 |
| Window around a block of no-ops of the same length (twin build, same probes) | 9,733 to 9,917 cycles (2 percent spread across scenes, 0.4 percent between two runs of one scene) | F122 |
| Corrected group time (window with the group minus the no-op window) at depths 1, 2, 3 | 1,522 / 1,855 / 2,163 cycles against 1,533 / 1,850 / 2,154 from the harness clock (group minus empty slots): within 11, 6, 9 cycles (0.7, 0.3, 0.4 percent) | F122 |
| Single probe subtraction (the first attempt) | wrong by a factor 6: a probe's register save and restore lies inside the window | F119 |
| OptiX recorder (not NVBit; clock reads and a counter in RTNN's own programs) | costs 7 to 15 percent of kernel time (event timing; F118's 1.1 to 5 times was host timer artefact); counts valid (lane efficiency identical to 0.003 at every sampling rate), per warp trace time divergence not observable | F118, F121 |

## Limits (read before relying on it)
- OptiX's own kernels are invisible to NVBit and to the official profilers (F12, F60). BITTU compute mode applies to kernels with TTU code compiled by us or others directly.
- A usable timing needs a twin build with no-ops in the group's place (F122); for OptiX kernels no twin exists.
- One warp, one group shape, one GPU in the validation (F119, F122). Many warps, other group shapes and the RTX 6000 Ada are not yet validated for BITTU.
- The tool must be built against the exact NVBit version (1.8.1) and the driver on the project's validated list.

## What may be released publicly
Checked 2026-11-18 (scan of the tracked files under `bittu/`: no 16 byte instruction, no hex run of 9 or more digits, no binary file, no dump word; `notes/push_scan.py` also runs before every push):
- Releasable (already tracked): all sources in `compute/` and `kill/`, `build.sh` files, `ttu_tables.h` (field positions and opcode numbers of the TTU instruction classes, generated from the EoSS SM89 tables, which are public in the upstream EoSS repository), this README, and the numbers in the table above (our own measurements).
- Not releasable, and not in the repository: the group byte files cut from binaries (`/dev/shm/bittu/group_*.bin`, NVIDIA instruction bytes, deleted after each run), dumps of OptiX's traversal code and of acceleration structures (`notes/`), and any binary compiled from the OptiX sequences. Tool reports contain register values (data) but no instruction bytes.
- Not included, obtained separately: NVBit (NVIDIA Labs; check its license before redistributing a built tool), CUDA, the EoSS package (upstream, MIT licensed).
- `run_bittu.py` imports the project's `eoss` package and `ttu/harness/setup3/run.py`; a standalone release needs those two pieces (our own code) with it.
