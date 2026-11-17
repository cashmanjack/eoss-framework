# CLOVER correctness check, second checker (2026-11-17, triton GPU 1)

Every point is a query, N = 20,000, brute force in double precision on the CPU (model/priorworks/clover_check2.cu). A query is wrong if an index repeats, is out of range or a returned point is farther than the true K-th distance by more than 1e-6 relative (squared). "tie" = the true K-th and (K+1)-th squared distances are within 1e-6 relative, so the K-th neighbour is not unique. Raw output in notes/q36/raw (private).

| algorithm | data | K | duplicate points | queries with a boundary tie | wrong queries | wrong with a tie | wrong without a tie | mean recall | worst relative excess over the K-th |
|---|---|---|---|---|---|---|---|---|---|
| bitonic | bunny (unique points) | 8 | 0 | 736 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 31 | 0 | 220 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 32 | 0 | 265 | 19880 | 265 | 19615 | 0.590 | 116 |
| bitonic | bunny (unique points) | 33 | 0 | 155 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 64 | 0 | 207 | 18959 | 167 | 18792 | 0.901 | 27 |
| bitonic | bunny (unique points) | 96 | 0 | 212 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 100 | 0 | 257 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 128 | 0 | 221 | 15841 | 191 | 15650 | 0.969 | 19.4 |
| bitonic | bunny (unique points) | 160 | 0 | 217 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 192 | 0 | 175 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | bunny (unique points) | 256 | 0 | 182 | 9920 | 63 | 9857 | 0.992 | 4.7 |
| bitonic | uniform | 8 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 31 | 0 | 1 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 32 | 0 | 0 | 20000 | 0 | 20000 | 0.131 | 312 |
| bitonic | uniform | 33 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 64 | 0 | 2 | 20000 | 2 | 19998 | 0.863 | 1.08 |
| bitonic | uniform | 96 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 100 | 0 | 3 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 128 | 0 | 2 | 19513 | 1 | 19512 | 0.972 | 0.138 |
| bitonic | uniform | 160 | 0 | 6 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 192 | 0 | 7 | 0 | 0 | 0 | 1.000 | 0 |
| bitonic | uniform | 256 | 0 | 8 | 15161 | 4 | 15157 | 0.994 | 0.061 |
| patched bitonic | bunny (unique points) | 31 | 0 | 220 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 32 | 0 | 265 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 33 | 0 | 155 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 64 | 0 | 207 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 96 | 0 | 212 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 128 | 0 | 221 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 160 | 0 | 217 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 192 | 0 | 175 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | bunny (unique points) | 256 | 0 | 182 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 31 | 0 | 1 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 32 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 33 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 64 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 96 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 128 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 160 | 0 | 6 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 192 | 0 | 7 | 0 | 0 | 0 | 1.000 | 0 |
| patched bitonic | uniform | 256 | 0 | 8 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | bunny (unique points) | 32 | 0 | 265 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | bunny (unique points) | 64 | 0 | 207 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | bunny (unique points) | 128 | 0 | 221 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | bunny (unique points) | 256 | 0 | 182 | 20000 | 182 | 19818 | 0.000 | 427 |
| warpwise | uniform | 32 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | uniform | 64 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | uniform | 128 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| warpwise | uniform | 256 | 0 | 8 | 20000 | 8 | 19992 | 0.000 | 62.6 |
| hubs | bunny (unique points) | 32 | 0 | 265 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | bunny (unique points) | 64 | 0 | 207 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | bunny (unique points) | 128 | 0 | 221 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | bunny (unique points) | 256 | 0 | 182 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | uniform | 32 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | uniform | 64 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | uniform | 128 | 0 | 2 | 0 | 0 | 0 | 1.000 | 0 |
| hubs | uniform | 256 | 0 | 8 | 0 | 0 | 0 | 1.000 | 0 |
