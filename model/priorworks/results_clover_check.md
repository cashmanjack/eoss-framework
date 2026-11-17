# CLOVER correctness check (2026-11-17, triton GPU 1)

Checker: model/priorworks/clover_check.cu (our code; includes CLOVER's linear-scans.cu with its main renamed; build line in the log). CPU brute force in double precision, every point is a query (as CLOVER's own runs), a neighbour set is wrong if any returned index repeats, is out of range or is farther than the true K-th distance (ties allowed); the reported distance list is compared with the true sorted list (squared distances). Raw output in the log.

| algorithm | data | K | queries wrong |
|---|---|---|---|
| bitonic | uniform 3,000 | 30 | 0 of 3,000 |
| bitonic | uniform 20,000 | 8, 16, 30, 31, 33, 96, 100, 160 | 0 |
| bitonic | uniform 20,000 | 32 | 20,000 of 20,000 |
| bitonic | uniform 3,000 / 20,000 | 128 | 2,210 of 3,000 / 19,512 of 20,000 |
| bitonic | bunny 20,000 | 30 / 64 / 128 | 0 / 18,955 / 15,826 |
| bitonic | dragon 20,000 | 128 | 13,632 |
| warpwise, hubs | uniform 3,000 (30), 20,000 (32, 64), bunny 20,000 (128) | | 0 |
| bitonic | Porto 3,000 | 30 | 0 |

Reading: bitonic returns wrong neighbours at K = 32, 64 and 128 on every set tried and is right at the other K values tried (including 96 and 160); cause not looked at. CLOVER's own mesh run uses K = 128, so a bitonic run there returns wrong neighbours; whether its published timings are affected is not known. warpwise and hubs were right in every run. The Treelogy kd tree (algorithm 7) is NOT checked: its indices come back in a buffer that the dispatcher does not return (my 3,000 of 3,000 mismatch is not evidence about the algorithm). hubs_ws and the FAISS variants are not compiled (no FAISS).
Meshes: CLOVER reads vertex text files from ../meshes (directory not shipped, file name must contain the vertex count); neither the repo nor its README says where the paper's meshes come from (the paper itself was not read).
