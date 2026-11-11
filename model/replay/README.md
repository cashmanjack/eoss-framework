# Replay for model v2 (CPU only)

CPU traversal over the decoded OptiX BVH (`notes/bvh_format.md`): per ray nodes fetched, leaf slots entered, triangles tested and queries (hit only return, nearest first).
- `assign.cpp`, `replay.cpp`: build with `g++ -O2`. `make_table.py GAS TRIS OUT`: node table from a GAS dump (needs `ASSIGN=` path of the assign binary). Triangles are put into leaf slots by box matching (the triangle to leaf mapping is not decoded).
- `validate.py WORK measure`: compares with queries per ray measured on the GPU (existing checked resume loop kernel) and with OptiX hits.
