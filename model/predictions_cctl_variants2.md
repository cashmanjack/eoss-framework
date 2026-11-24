# TTUCCTL side and session variants, catalogue items 1 to 7: predictions, committed before any build or run

2026-11-24, triton GPU 1 (TTU_GPU=1). Approved by Jack: items 1 to 7 of notes/ttu_variant_catalogue.md, medium items not started.
Rules: before each run nvidia-smi and who, run only if no GPU compute process of anyone is present; one warp; timeout -k 2 30; one variant at a time; outcome classes: ran correctly, wrong result, fault, timeout or hang; a hang that does not clear stops everything. No process matching by name anywhere (CLAUDE.md rule of today): the runner uses its own child PIDs and nvidia-smi PIDs.
Setup as F434 and F435: 1,024 and 2,048 triangle scene, one warp, one block on SM 0, 1 launch, then 3 x 30 launches per size. "Saved" = 1 minus the variant's second session cycles over a fresh single session (single about 100,500 and 176,000). References: two sessions with nothing between saved 0.619 and 0.540; the validated TTUCCTL saved 0.000 and 0.000.
"Ran correctly" = exit 0, one TTU block, results same every launch, analyze.py PASS, and the 22 result registers equal to the unchanged base build of the same session (address words excluded).

Each variant is the checked base build with the listed instruction(s) edited at fixed positions; a narrow rule in check_seq accepts it by sha256 only after the base build passes and the diff to the base is exactly the listed edits (rule in notes/, hashes in a private file).

| item | variant (base build) | prediction (accept) |
|---|---|---|
| 1 | F0: TTUMACROFUSE with count 0 in place of the TTUCCTL (cctl build) | ran correctly (paper: executes standalone); saved as with nothing between the sessions: 0.62 and 0.54 (0.56 to 0.68 and 0.47 to 0.61) |
| 2 | P1, P2, P3: TTUCCTL pm_pred = 1, 2, 3 (cctl build) | ran correctly, still invalidates: saved 0.00 (at most 0.03), cycles of both sessions within 1 percent of the plain TTUCCTL build |
| 3 | O1 (batch NOP, usched DRAIN), O2 (NOP, W2), O3 (BATCH_START, W1), O4 (BARRIER_EXEMPT, W1) on the TTUCCTL (cctl build); only pairs of the spec's table | O1 and O2 ran correctly, saved 0.00; O3 and O4 ran correctly with probability 0.7, then saved 0.00; if not, a fault (no hang) |
| 4 | W1: TTUCCTL waits on scoreboard 1 (the TTULD results scoreboard, already complete), W2: waits on all six (a copy of the wait NOP before it) (cctl build) | both ran correctly, saved 0.00, cycles within 1 percent (the wait is already satisfied) |
| 5 | S1: all TTUST read scoreboard 0 to 2 (unused index), S2: to 1 (the TTULD results scoreboard) in both sessions (twice build) | both ran correctly, results equal, saved 0.62 and 0.54 (accept 0.58 to 0.66 and 0.50 to 0.58), cycles within 1 percent of the unchanged twice build |
| 6 | X1: the TTUST at offset 0xC0 (the copy of the 0x40 origin store) predicated off with @!PT in both sessions (twice build) | ran correctly with probability 0.6: results equal to the unchanged build, saved 0.62 and 0.54; otherwise wrong result (hit lost), not a hang (probability 0.3); fault 0.1 |
| 7 | D1: both TTUOPEN with the DUAL bit cleared (twice build) | ran correctly with probability 0.55 and results equal to the dual form; wrong result 0.2; fault 0.2; hang 0.05. If it runs: single and second session cycles within 5 percent of the dual form, saved fraction within 0.05 of 0.62 and 0.54 |

What would change the plan: a wrong result or fault stops that item's follow ups (no sibling variants are run to chase it); a hang stops everything.
