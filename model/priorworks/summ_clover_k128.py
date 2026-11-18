"""Summary of the CLOVER K 128 and Arkade dataset runs (notes/q38/raw/clk128_*, 2026-11-17). Prints markdown. warpwise times are for Q = 10,000 random queries and also scaled linearly to all N points (linearity checked in F227 item 8)."""
import glob, re, os, statistics as st
R = os.path.expanduser('~/eoss/notes/q38/raw/')
rows = {}
for f in sorted(glob.glob(R + 'clk128_*_N*_Q*_k*_*.txt')):
    m = re.match(r'clk128_(\w+?)_(.+)_N(\d+)_Q(\d+)_k(\d+)_', os.path.basename(f)); tag, ds, N, Q, K = m[1], m[2], int(m[3]), int(m[4]), int(m[5])
    t = open(f).read(); ns = [int(x) for x in re.findall(r'^(\d+), rep', t, re.M)]; c = re.search(r'sampled (\d+) wrong (\d+) far (\d+) invalid (\d+)', t)
    if ns and c: rows[(ds, K, tag)] = (st.median(ns) / 1e6, len(ns), N, Q, int(c[2]) + int(c[3]) + int(c[4]), int(c[1]))
print('| dataset | K | N | warpwise ms (10,000 queries) | warpwise ms scaled to all N | hubs ms (all N queries, median of runs) | wrong in 1000 sampled (warp / hubs) |\n|---|---|---|---|---|---|---|')
for (ds, K) in sorted({(a, b) for a, b, _ in rows}, key=lambda x: (x[1] != 128, x[0], x[1])):
    w = rows.get((ds, K, 'warp')); h = rows.get((ds, K, 'hubs'))
    print('| %s | %d | %d | %s | %s | %s | %s / %s |' % (ds, K, (w or h)[2], '%.1f' % w[0] if w else '-', '%.0f' % (w[0] * w[2] / w[3]) if w else '-', '%.1f [%d]' % (h[0], h[1]) if h else '-', w[4] if w else '-', h[4] if h else '-'))
