# Replay for model v2 (CPU only)

CPU traversal over the decoded OptiX BVH (`notes/bvh_format.md`): per ray nodes fetched, leaf slots entered, triangles tested and queries (hit only return, nearest first).
- `assign.cpp`, `replay.cpp`: build with `g++ -O2`. `make_table.py GAS TRIS OUT`: node table from a GAS dump (needs `ASSIGN=` path of the assign binary). Triangles are put into leaf slots by box matching (the triangle to leaf mapping is not decoded).
- `validate.py WORK measure`: compares with queries per ray measured on the GPU (existing checked resume loop kernel) and with OptiX hits.

Custom primitive (box) GAS and the leaf slot assignment (F166, F167). make_table puts primitives into leaf slots by box matching. On a FRESHLY BUILT tree (RTNN, 12 configurations, F167) this is exact: the replay with the true counts equals it for every query. On a REFIT tree whose primitives have moved (Mochi, F166) it is wrong (-8.6 to -50 percent on the calls per query once sibling boxes overlap). Use the true counts there: the slot type byte holds the primitives of the slot in its low 3 bits (0 means 8; the sum over all slots equals the primitive count); model/mochi_leafcount.py and model/rtnn_leafcount.py rebuild the table's leaf starts from it (primitive ids are not decoded, only counts, which is all a point query count needs).
