"""Meeting figures (CPU only, existing data, no new runs) -> notes/figures/meeting/*.png.   /usr/bin/python3 model/q20/meeting_figures.py
Sources are named in each block; numbers taken from the Findings are marked (F nnn)."""
import json, glob, os, re, statistics as st, math
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.expanduser('~/eoss/'); O = R + 'notes/figures/meeting/'
B, Or, G, K, Rd = '#1f5fa8', '#d1621b', '#4a8f3c', '#555555', '#b03a2e'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130, 'axes.titlesize': 12, 'axes.titleweight': 'bold'})
def save(fig, n, rect=None): fig.tight_layout(rect=rect); fig.savefig(O + n, dpi=150); plt.close(fig); print('wrote', n)

# 1 latency staircase (4070 Ti pointer chase, notes/io/core/chase_ca_*.json)
pts = []
for f in glob.glob(R + 'notes/io/core/chase_ca_*.json'):
    j = json.load(open(f)); pts.append((j['size_kb'] / 1024., st.median(j['cycles_per_load'])))
pts.sort(); fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(*zip(*pts), 'o-', color=B, ms=5); ax.set_xscale('log')
for lo, hi, c, t in ((0.01, 0.09, G, 'L1\n41 cycles'), (0.1, 40, B, 'L2\n285 cycles'), (80, 600, Rd, 'DRAM\n660 cycles')):
    ax.axvspan(lo, hi, color=c, alpha=.07); ax.text(math.sqrt(lo * hi), 520 if c != Rd else 560, t, ha='center', color=c, fontsize=10)
ax.axvline(48, color=K, ls=':', lw=1); ax.annotate('L2 = 48 MB', (48, 150), xytext=(-6, 0), textcoords='offset points', ha='right', color=K, fontsize=9)
ax.set_ylim(0, 700); ax.set_xlabel('working set of a cyclic pointer chase (MB)'); ax.set_ylabel('cycles per load'); ax.set_title('Latency is a staircase with one step per cache level (RTX 4070 Ti)')
save(fig, 'm1_latency_staircase.png')

# 2 throughput vs active TPCs (triton, notes/setup3/runs/throughput6/report.txt, par64 = long queries, one block per SM, blocks fill even SMs first)
rows = []
for l in open(R + 'notes/setup3/runs/throughput6/report.txt'):
    p = [x.strip() for x in l.split('|')]
    if p[0] == 'par64' and p[1].isdigit(): rows.append((int(p[1]), float(p[3].split()[-1])))
a = [r for r in rows if r[0] <= 71]; b = [r for r in rows if r[0] >= 71]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.1), gridspec_kw={'width_ratios': [1.5, 1]})
ax[0].plot(*zip(*a), 'o-', color=B, ms=4, label='measured'); ax[0].plot([0, 71], [0, 71], color=K, ls=':', lw=1, label='one unit of throughput per TPC')
ax[0].set_xlabel('active TPCs (one SM each)'); ax[0].set_ylabel('throughput (units of one SM)'); ax[0].legend(frameon=False, fontsize=9); ax[0].set_title('Linear in TPCs up to 71')
ax[1].plot([r[0] - 71 for r in b], [r[1] for r in b], 'o-', color=Or, ms=4); ax[1].set_ylim(0, 80); ax[1].set_xlabel('second SMs added (all 71 TPCs active)'); ax[1].set_title('A second SM adds almost nothing')
save(fig, 'm2_throughput_vs_tpcs.png')

# 3 per element cost bars (cycles; log). Callback side: F131 microbenchmarks (4070 Ti); RT side: F5 / fig1 slopes (triton), F127 (candidate), F137 (hand off), Throughput_1 (7.4 cycles per ray, one SM saturated)
items = [('warp instruction', .25, B), ('L1 tag request', 1.0, B), ('scattered L2 sector', 2.9, B), ('double precision instruction', 19, B),
         ('custom primitive candidate (per SM)', 4.9, Or), ('ray at saturation (one SM)', 7.4, Or), ('BVH level, node in L1', 43, Or), ('BVH level, node in L2', 337, Or), ('BVH level, node in DRAM', 620, Or), ('callback hand off (per warp call)', 875, Rd)]
fig, ax = plt.subplots(figsize=(8.4, 5.8)); y = np.arange(len(items))[::-1]
ax.barh(y, [i[1] for i in items], color=[i[2] for i in items]); ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9.5)
for yy, i in zip(y, items): ax.text(i[1] * 1.12, yy, '%g' % i[1], va='center', fontsize=9)
ax.set_xlabel('cycles (blue: SM instruction and cache, orange: RT unit, red: hand off)', fontsize=9.5); ax.set_xlim(.1, 3000); ax.set_title('Per element costs span four orders of magnitude'); ax.grid(axis='y', alpha=0)
fig.text(.01, .01, 'Sources (Findings):\n instruction .25, L1 tag 1.0, L2 sector 2.9, double precision 19: F131 (4070 Ti microbenchmarks)\n candidate 4.9 per SM: F116 (RTNN slope; F127 gives 4 to 9)\n ray 7.4 at saturation: F24, F31 (5.7 to 7.4 per ray, one SM)\n BVH level 43 (L1): F25, F26; 337 (L2), 620 (DRAM): F26 ranges 270 to 400, 580 to 750; point values from fig1 slopes\n hand off 875 per warp call: F49 (850 to 900), used in F137', fontsize=7.4, color=K, va='bottom')
save(fig, 'm3_per_element_cost.png', rect=(0, .2, 1, 1))

# 4 node state capacity (F150, notes/q19/capacity_table.txt), sibling results F155 (notes/q20/sib_matrix.log medians; its json was overwritten by the F157 run) F157 (notes/q20/sib_fence.json) and F159 fence decomposition (notes/q21/fence2_stage1_20261116T133352.json). Measurement firm; interpretation (which state, which part of the fence) left open in the caption.
rows = [l.split() for l in open(R + 'notes/q19/capacity_table.txt') if l.strip() and l.split()[0].isdigit()]
kb = [float(r[2]) for r in rows]; sv = [float(r[7]) for r in rows]
nm = 'single twice flag0 sib4 ctl4 sib16 ctl16 flag0p sib4p ctl4p'.split(); f155 = {}
for l in open(R + 'notes/q20/sib_matrix.log'):
    t = l.split('|')[0].split()
    if len(t) == 11 and t[0].isdigit(): f155[int(t[0])] = dict(zip(nm, map(int, t[1:])))
fen = json.load(open(R + 'notes/q20/sib_fence.json'))
fen2 = json.load(open(R + 'notes/q21/fence2_stage1_20261116T133352.json')); fenh = json.load(open(R + 'notes/q22/fence_h_stage1_20261116T170734.json'))
def med2(a, N): return st.median(c for x in fen2['%s|%d' % (a, N)] for c in x['cycles'])
def sav2(a, N): return 1 - med2(a, N) / med2('single', N)
def med(a, N): return st.median(c for x in fen['%s|%d' % (a, N)] for c in x['cycles'])
Ns = [1024, 2048, 3072, 4096]; KB = [68, 132, 199, 263]
fig, ax = plt.subplots(figsize=(7.6, 6.6)); ax.plot(kb, sv, 'o-', color=B, ms=4, label='second session, alone (F150)')
ax.plot(KB, [1 - f155[N]['sib4'] / f155[N]['single'] for N in Ns], 's--', color=Rd, ms=6, label='sibling block that streams loads and fences (F155)')
ax.plot(KB, [1 - f155[N]['ctl4'] / f155[N]['single'] for N in Ns], '^:', color=G, ms=6, label='same block on another TPC (F155)')
ax.plot(KB[:2], [1 - med('c1', N) / med('single', N) for N in (1024, 2048)], 'D', color=Rd, mfc='none', ms=8, label='sibling that only waits, then ONE fence (F157)')
ax.plot(KB[:2], [1 - med('a1', N) / med('single', N) for N in (1024, 2048)], 'v', color='#7b3fa0', ms=8, label='sibling that only waits, no fence (F157)')
for a_, mk, col, lab in (('e1', 'o', G, 'cta scope fence on the sibling SM (F159)'), ('f0', 'X', Or, 'gpu scope fence on the TTU warp\'s own SM (F159)'), ('g', 'P', 'k', 'gpu scope fences on both SMs of the TPC (F159)')):
    ax.plot(KB[:2], [sav2(a_, N) for N in (1024, 2048)], mk, color=col, mfc=col if a_ != 'e1' else 'none', ms=9, label=lab)
medh = lambda a, N: st.median(c for x in fenh['%s|%d' % (a, N)] for c in x['cycles'])
ax.plot(KB[:2], [1 - medh('h', N) / medh('single', N) for N in (1024, 2048)], '*', color=Rd, ms=13, label='TTUCCTL alone on the sibling SM, no memory barrier (F161)')
feni = json.load(open(sorted(glob.glob('/home/min/a/cashman3/eoss/notes/q22/fence_i_stage1_*.json'))[-1]))
medi = lambda a, N: st.median(c for x in feni['%s|%d' % (a, N)] for c in x['cycles'])
ax.plot(KB[:2], [1 - medi('i', N) / medi('single', N) for N in (1024, 2048)], 'd', color=Rd, mfc='none', ms=10, label='TTUCCTL on the TTU warp\'s own SM, no memory barrier (F162)')
ax.axvspan(80, 96, color=K, alpha=.15); ax.text(88, .03, 'L1 cliff\n80 to 96 KB', ha='center', fontsize=8.5, color=K); ax.set_xscale('log')
ax.set_xlabel('BVH size, GAS bytes read by one ray (KB)'); ax.set_ylabel('saving of the second session (1 minus second / first)'); ax.set_ylim(0, .75)
ax.set_title('Second session saving: a gpu fence on either SM of the TPC removes about 55 percent,\non both all; a TTUCCTL alone on either SM removes all; a cta fence none',fontsize=9.5); ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.14), ncol=1)
fig.text(.01, .005, 'Fence tests at 68 and 132 KB only. Not separated: the gpu barrier against the L1 invalidate it comes with,\nand one L1 pool shared over the TPC against separate per SM state (F159). F161, F162: TTUCCTL on the sibling or on the TTU warp own SM removes the whole saving.', fontsize=7.5, color=K); save(fig, 'm4_node_state_capacity.png', rect=(0, .04, 1, 1))

# 5 BITTU against the harness clock (F122, notes/q11/timing_fix.json)
j = json.load(open(R + 'notes/q11/timing_fix.json')); x = [v['harness_s1'] - v['harness_s0'] for v in j.values()]; y = [v['corrected'] for v in j.values()]
fig, ax = plt.subplots(figsize=(4.8, 4.5)); ax.plot([1400, 2300], [1400, 2300], color=K, lw=1); ax.plot(x, y, 'o', color=B, ms=8)
for xx, yy, n in zip(x, y, ('1 level', '2 levels', '3 levels')): ax.annotate('%s: %+.1f%%' % (n, (yy / xx - 1) * 100), (xx, yy), xytext=(8, -12), textcoords='offset points', fontsize=9)
ax.set_xlabel('harness clock, group minus empty slots (cycles)'); ax.set_ylabel('BITTU, corrected (cycles)'); ax.set_xlim(1400, 2300); ax.set_ylim(1400, 2300); ax.set_title('BITTU timing is within 1%\nof the harness (3 scenes)', fontsize=11); save(fig, 'm5_bittu_vs_harness.png')

# 6 predicted over measured, every workload and GPU. Filled = prediction committed before the measurement, hollow = after the fact. Pairs: core set tables of model/results_4070ti.md (F103), RTNN (model/results_rtnn_4070ti.md), kNN floor + hand off (F137, model/results_knn_fix_4070ti.md), bounce speedups (notes/io/predict), Arkade (F145, F149), TrueKNN (F156)
P = []   # (workload, gpu, predicted, measured, first)
for p, m in zip([1641, 1652, 1908, 1920, 2315, 2315, 2639], [1612, 1635, 1930, 1984, 2234, 2248, 2564]): P.append(('latency warm', 'io', p, m, True))
for p, m in zip([2648, 2680, 3228, 3284, 3887, 4058, 4635], [2593, 2663, 3351, 3245, 4004, 3981, 4645]): P.append(('latency cold', 'io', p, m, True))
for p, m in zip([.0195, .031, .0445, .058, .080, .090], [.0200, .0316, .0450, .0586, .0811, .0910]): P.append(('throughput 1 warp to 20', 'io', p, m, True))
for p, m in ((.080, .216), (.490, .467), (.909, .747), (.965, 1.583), (6.261, 4.176), (9.260, 6.413)): P.append(('RTNN launch time', 'io', p, m, True))
for p, m in zip([.374, .552, 1.761, 6.019], [.661, .979, 3.502, 11.016]): P.append(('kNN rounds 1 to 4', 'io', p, m, True))
for s in (100000, 3000000):
    for n_, p, m in json.load(open(R + 'notes/io/predict/predict_bounce_%d.json' % s))['rows']: P.append(('bounce ray speedup', 'io', p, m, True))
P += [('Arkade device buffer', 'io', 3.71, 3.70, True), ('Arkade device buffer', 'triton', 1.57, 1.97, False), ('TrueKNN rounds sum', 'triton', 2.623, 4.955, True), ('TrueKNN rounds sum', 'triton', 8.840, 10.394, True)]
names = []; [names.append(p[0]) for p in P if p[0] not in names]
fig, ax = plt.subplots(figsize=(8.6, 4.8)); rng = np.random.default_rng(1)
for p in P:
    xx = names.index(p[0]) + rng.uniform(-.18, .18); yy = math.log2(p[2] / p[3]); c = B if p[1] == 'io' else Or
    ax.scatter(xx, yy, s=34, marker='o' if p[1] == 'io' else 's', facecolors=c if p[4] else 'none', edgecolors=c, linewidths=1.4)
for v in (-1, 1): ax.axhline(v, color=K, ls=':', lw=1)
ax.axhline(0, color=K, lw=1); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=28, ha='right', fontsize=9); ax.set_ylabel('log2 (predicted / measured)'); ax.set_ylim(-2, 2)
ax.set_title('Latency, throughput: within 10%. Callback programs: up to 3x off'); ax.text(len(names) - .6, 1.05, 'predicted 2x too high', ha='right', fontsize=8.5, color=K); ax.text(len(names) - .6, -1.2, 'predicted 2x too low', ha='right', fontsize=8.5, color=K)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=B, label='4070 Ti (io)'), Line2D([], [], marker='s', ls='', color=Or, label='6000 Ada (triton)'), Line2D([], [], marker='o', ls='', mfc=K, mec=K, label='filled: predicted first'), Line2D([], [], marker='o', ls='', mfc='none', mec=K, label='hollow: after the fact')], frameon=False, fontsize=8.5, loc='lower left', ncol=2)
save(fig, 'm6_predicted_vs_measured.png')

# 7 time breakdown per workload (F134, every measured row of model/results_latency_term.md): range over the rows as a bar with end caps, a single measurement as a dot; NO midpoints.
rows = {}
for l in open(R + 'model/results_latency_term.md'):
    t = [x.strip() for x in l.split('|')]
    if len(t) > 9 and re.match(r'(rtnn|knn_round[1-4]$|bh_|ark_)', t[1]) and t[7].endswith('%') and t[9].endswith('%') and t[1] not in rows: rows[t[1]] = [float(x.rstrip('%')) for x in (t[7], t[8], t[9])]
grp = [('RTNN (12 configs)', 'rtnn'), ('kNN rounds 1 to 4', 'knn'), ('RT-BarnesHut 1M', 'bh_g1M'), ('RT-BarnesHut 100k', 'bh_g100k'), ('Arkade (3 configs)', 'ark')]
fig, ax = plt.subplots(figsize=(7.6, 4.6)); y0 = np.arange(len(grp))[::-1] * 1.0
for gi, (lab, key) in enumerate(grp):
    R_ = [v for k, v in rows.items() if k.startswith(key) and v]
    for ci, (cn, c) in enumerate((('callbacks on the SM', Or), ('traversal (RT unit)', B), ('other', '#777777'))):
        v = [r[ci] for r in R_]; lo, hi = min(v), max(v); yy = y0[gi] + (1 - ci) * .25
        if hi - lo < 1e-9: ax.plot([lo], [yy], 'o', color=c, ms=6, label=cn if gi == 0 else None)
        else:
            ax.plot([lo, hi], [yy, yy], '-', color=c, lw=2.5, label=cn if gi == 0 else None); ax.plot([lo, lo, None, hi, hi], [yy - .07, yy + .07, None, yy - .07, yy + .07], '-', color=c, lw=1.5)
        ax.text(hi + 1.5, yy, ('%d' % lo) if hi - lo < 1e-9 else '%d to %d' % (lo, hi), va='center', fontsize=8, color=c)
ax.set_yticks(y0); ax.set_yticklabels([g[0] for g in grp]); ax.set_xlabel('percent of warp time (bar: range of rows, dot: one row)'); ax.set_xlim(0, 100); ax.grid(axis='y', alpha=0)
ax.set_title('Callbacks, not traversal, dominate most RT programs'); ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3)
fig.text(.01, .005, 'F134. RT-BarnesHut 100k: other = 30 is a first launch cost (28). Probes add 0 to 31 percent to launch time (F133).', fontsize=7.5, color=K); save(fig, 'm7_time_breakdown.png')

# 8 Arkade waterfall on io (F145 pinned 24.1 ms, F154 device buffer 3.70 and float square 0.36)
fig, ax = plt.subplots(figsize=(6.6, 4.2)); v = [24.1, 3.70, 0.36]; lab = ['original\n(pinned host output)', 'device buffer\noutput', 'float distance\n(same neighbours)']
ax.bar(range(3), v, color=[Rd, Or, G], width=.55); ax.set_yscale('log'); ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylabel('launch time (ms)'); ax.set_ylim(.1, 60)
for i, x_ in enumerate(v): ax.text(i, x_ * 1.15, '%g ms' % x_, ha='center', fontsize=10)
ax.annotate('6.5 times', (.5, 9), ha='center', color=K, fontsize=10); ax.annotate('10 times', (1.5, 1.1), ha='center', color=K, fontsize=10); ax.grid(axis='x', alpha=0)
ax.set_title('Arkade on the 4070 Ti: two code changes, 67 times faster'); save(fig, 'm8_arkade_waterfall.png')
