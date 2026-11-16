# Fence decomposition: results (predictions in model/predictions_fence.md, commit aaabc8c)

2026-11-16, triton GPU 1, stage 0 check of every arm first (no fault), then stage 1, 5 repetitions x 30 launches per cell in rotating order, medians, every run exit 0, exactly one TTU block per launch, no sibling timeout, results same every launch. Raw: notes/q21/fence2_stage1_20261116T133352.json (private).

Compiled fences (cuobjdump of the host binary and of the patched TTU kernel):
- fence.acq_rel.cta (plain block): one MEMBAR.ALL.CTA, nothing else.
- fence.acq_rel.gpu (plain block): MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL.
- own fence in the TTU kernel (build sibF, check_seq B15 ownfence PASS): the same three instructions MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL, copied from the compiled plain kernel, unpredicated, after the first CLOSE, before the spin on flag2, outside every session.

Cycles of the timed (second) session, 68 KB (N 1,024) / 132 KB (N 2,048):
single 100,429 / 176,148; twice 38,239 / 80,742.

| arm | what | cycles 68 / 132 KB | saved 68 / 132 | predicted | per repetition |
| b1 | sibling SM 1 polls, no fence (reference) | 39,344 / 88,853 | 0.61 / 0.50 | | 0.61 all; 0.48 to 0.52 |
| c1 | gpu fence on sibling SM 1 (F157 c) | 73,929 / 136,339 | 0.26 / 0.23 | | 0.25 to 0.27; 0.22 to 0.23 |
| cC | gpu fence on SM 20, another TPC | 39,351 / 87,819 | 0.61 / 0.50 | | 0.61; 0.48 to 0.54 |
| e | cta fence on sibling SM 1 | 39,439 / 89,232 | 0.61 / 0.49 | 0.62 / 0.54 (0.55 to 0.65 / 0.47 to 0.57) | 0.61; 0.48 to 0.51 |
| f | gpu fence by the TTU warp on its own SM 0 only | 74,360 / 132,269 | 0.26 / 0.25 | 0.31 / 0.27 (0.22 to 0.40 / 0.19 to 0.35) | 0.25 to 0.26; 0.25 |
| g | own gpu fence plus gpu fence on SM 1 | 100,104 / 175,791 | 0.00 / 0.00 | at most 0.12 / 0.10 | 0.00 all |
| h | TTUCCTL alone on the sibling | not run | | | |

(Medians over 5 x 30 launches; printed table in notes/q21/fence2_stage1_run_*.log.)

All three predictions fell inside their accepted ranges; the consistency check (f about equal to the sibling fence c1) held: 0.26 / 0.25 against 0.26 / 0.23.

Skipped (h): TTUCCTL alone on the sibling SM needs a second TTU block executing a TTUCCTL outside any session, a sequence type that check_seq does not validate and the safety rules do not cover (every TTU sequence we run has one TTUOPEN, one TTUMACROFUSE, one CLOSE). Not run.

Harness note: a plain block placed first on SM 0 (the TTU warp's SM) keeps the TTU block off SM 0 (0 of 3 launches ran with the one block build, 0 of 3 with 142 blocks; the same plain block on SM 1 lets it run). A plain CUDA probe (kernel B started after kernel A holds SM 0) also stalled. So (f) and (g) use a fence executed by the TTU warp itself (new build option --ownfence), not a plain block; this differs from (c) in who executes the fence (the TTU warp, which stalls on it, against a separate block).
