"""Summarize the clover_check2 raw outputs (notes/q36/raw/clover2_*, clover2p_* = patched bitonic) into a table. Usage: python3 model/priorworks/clover_summarize.py > model/priorworks/results_clover_check2.md"""
import re, glob
rows = []
for f in sorted(glob.glob('/home/min/a/cashman3/eoss/notes/q36/raw/clover2*_a*_k*_*.txt')):
    s = open(f).read(); m = re.search(r'alg (\d) N (\d+) K (\d+) (\S+): duplicate points in the data (\d+); queries with a tie at the K-th boundary (\d+); WRONG queries (\d+) \(with boundary tie (\d+), without (\d+)\).*?mean recall (\S+); worst relative excess of a returned squared distance over the K-th (\S+), mean (\S+)', s)
    if not m: continue
    tag = 'patched bitonic' if '/clover2p_' in f else {'0': 'bitonic', '1': 'warpwise', '2': 'hubs'}[m[1]]
    data = 'uniform' if m[4] == 'uniform' else 'bunny (unique points)'
    rows.append((tag, data, int(m[3]), int(m[2]), int(m[5]), int(m[6]), int(m[7]), int(m[8]), int(m[9]), float(m[10]), float(m[11])))
rows.sort(key=lambda r: (['bitonic', 'patched bitonic', 'warpwise', 'hubs'].index(r[0]), r[1], r[2]))
print('# CLOVER correctness check, second checker (2026-11-17, triton GPU 1)\n\nEvery point is a query, N = 20,000, brute force in double precision on the CPU (model/priorworks/clover_check2.cu). A query is wrong if an index repeats, is out of range or a returned point is farther than the true K-th distance by more than 1e-6 relative (squared). "tie" = the true K-th and (K+1)-th squared distances are within 1e-6 relative, so the K-th neighbour is not unique. Raw output in notes/q36/raw (private).\n')
print('| algorithm | data | K | duplicate points | queries with a boundary tie | wrong queries | wrong with a tie | wrong without a tie | mean recall | worst relative excess over the K-th |\n|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print('| %s | %s | %d | %d | %d | %d | %d | %d | %.3f | %.3g |' % (r[0], r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9], r[10]))
