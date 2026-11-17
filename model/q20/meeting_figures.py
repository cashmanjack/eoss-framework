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
         ('custom primitive candidate (per SM)', 4.9, Or), ('ray at saturation (one SM)', 7.4, Or), ('BVH level, node in L1', 43, Or), ('BVH level, node in L2', 337, Or), ('BVH level, node in DRAM', 620, Or), ('warp call, saturated SM (F204)', 96, Rd), ('callback hand off, one warp alone', 875, Rd)]
fig, ax = plt.subplots(figsize=(8.4, 5.8)); y = np.arange(len(items))[::-1]
ax.barh(y, [i[1] for i in items], color=[i[2] for i in items]); ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9.5)
for yy, i in zip(y, items): ax.text(i[1] * 1.12, yy, '%g' % i[1], va='center', fontsize=9)
ax.set_xlabel('cycles (blue: SM instruction and cache, orange: RT unit, red: hand off)', fontsize=9.5); ax.set_xlim(.1, 3000); ax.set_title('Per element costs span four orders of magnitude', fontsize=12); ax.grid(axis='y', alpha=0)
fig.text(.01, .01, 'Sources (Findings):\n instruction .25, L1 tag 1.0, L2 sector 2.9, double precision 19: F131 (4070 Ti microbenchmarks)\n candidate 4.9 per SM: F116 (RTNN slope; F127 gives 4 to 9)\n ray 7.4 at saturation: F24, F31 (5.7 to 7.4 per ray, one SM)\n BVH level 43 (L1): F25, F26; 337 (L2), 620 (DRAM): F26 ranges 270 to 400, 580 to 750; point values from fig1 slopes\n hand off 875 = latency a lone warp waits per call: F49 (850 to 900), used in F137; \n 96 = cost per warp call on a saturated SM, other warps overlap it (9 times below the lone warp latency): F204', fontsize=7.4, color=K, va='bottom')
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
f174 = json.load(open(sorted(glob.glob('/home/min/a/cashman3/eoss/notes/q32/own_sm_stage1_*.json'))[-1]))   # last file = the 5 x 30 run
med4 = lambda a, N: st.median(c for x in f174['%s|%d' % (a, N)] for c in x['cycles'])
ax.plot(KB[:2], [1 - med4('own8', N) / med4('single', N) for N in (1024, 2048)], 'h', color=Or, ms=10, label='1 MB of ordinary loads on the TTU\'s own SM between the sessions (F174)')
ax.plot(KB[:2], [1 - med4('own9', N) / med4('single', N) for N in (1024, 2048)], 'h', color=Or, mfc='none', ms=10, label='same wait (70 us), no loads (F174)')
ax.axvspan(80, 96, color=K, alpha=.15); ax.text(88, .03, 'L1 cliff\n80 to 96 KB', ha='center', fontsize=8.5, color=K); ax.set_xscale('log')
ax.set_xlabel('BVH size, GAS bytes read by one ray (KB)'); ax.set_ylabel('saving of the second session (1 minus second / first)'); ax.set_ylim(0, .75)
ax.set_title('Second session saving: a gpu fence on either SM of the TPC removes about 55 percent,\non both all; a TTUCCTL alone on either SM removes all; a cta fence none;\n1 MB of ordinary loads on the own SM removes about half',fontsize=9.5); ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.14), ncol=1)
fig.text(.01, .005, 'Fence, TTUCCTL and F174 tests at 68 and 132 KB only. F161, F162: TTUCCTL on the sibling or on the TTU warp own SM removes the whole saving.\nF174: the 70 us wait without loads also costs part of the saving (0.15 at 132 KB, open). Not separated: gpu barrier against the L1 invalidate (F159).', fontsize=6.8, color=K); save(fig, 'm4_node_state_capacity.png', rect=(0, .04, 1, 1))

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
# refreshed 2026-11-17: F207 real data rows (predicted first; F207 form), measured / predicted ratios from model/priorworks/results_priorworks.md
for r_ in (2.53, 2.23, 2.49, 1.69): P.append(('RTNN real data (F207)', 'triton', 1, r_, True))
for r_ in (3.12, 161.90, 90.72, 0.56): P.append(('TrueKNN real data (F207)', 'triton', 1, r_, True))
for r_ in (4.04, 1.73, 7.69, 3.22, 3.79, 1.31): P.append(('Arkade real data (F207)', 'triton', 1, r_, True))
names = []; [names.append(p[0]) for p in P if p[0] not in names]
fig, ax = plt.subplots(figsize=(8.6, 4.8)); rng = np.random.default_rng(1)
for p in P:
    xx = names.index(p[0]) + rng.uniform(-.18, .18); yy = math.log2(p[2] / p[3]); c = B if p[1] == 'io' else Or
    if yy < -3: ax.scatter(xx, -2.9, s=40, marker='v', color=Rd); continue
    ax.scatter(xx, yy, s=34, marker='o' if p[1] == 'io' else 's', facecolors=c if p[4] else 'none', edgecolors=c, linewidths=1.4)
for v in (-1, 1): ax.axhline(v, color=K, ls=':', lw=1)
ax.axhline(0, color=K, lw=1); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=28, ha='right', fontsize=9); ax.set_ylabel('log2 (predicted / measured)'); ax.set_ylim(-3, 2)
ax.set_title('Latency, throughput: within 10%. Callback programs on real data:\n2 to 8 times off, TrueKNN up to 160 times (red triangles, off scale)', fontsize=11); ax.text(len(names) - .6, 1.05, 'predicted 2x too high', ha='right', fontsize=8.5, color=K); ax.text(len(names) - .6, -1.3, 'predicted 2x too low', ha='right', fontsize=8.5, color=K)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=B, label='4070 Ti (io)'), Line2D([], [], marker='s', ls='', color=Or, label='6000 Ada (triton)'), Line2D([], [], marker='o', ls='', mfc=K, mec=K, label='filled: predicted first'), Line2D([], [], marker='o', ls='', mfc='none', mec=K, label='hollow: after the fact')], frameon=False, fontsize=8.5, loc='lower left', ncol=2)
save(fig, 'm6_predicted_vs_measured.png')

# 7 time breakdown per workload (F134, every measured row of model/results_latency_term.md): range over the rows as a bar with end caps, a single measurement as a dot; NO midpoints.
rows = {}
for l in open(R + 'model/results_latency_term.md'):
    t = [x.strip() for x in l.split('|')]
    if len(t) > 9 and re.match(r'(rtnn|knn_round[1-4]$|bh_|ark_)', t[1]) and t[7].endswith('%') and t[9].endswith('%') and t[1] not in rows: rows[t[1]] = [float(x.rstrip('%')) for x in (t[7], t[8], t[9])]
# refreshed 2026-11-17: real data rows of F220 (probe split of the warp time, triton): callbacks / traversal window / outside the trace window
for i_, r_ in enumerate(([15, 77, 9], [28, 71, 1], [25, 74, 2], [34, 66, 0])): rows['zreal_rtnn%d' % i_] = r_
for i_, c_ in enumerate((80, 80, 73, 95, 95, 82)): rows['zreal_ark%d' % i_] = [c_, None, None]
grp = [('RTNN (12 configs)', 'rtnn'), ('kNN rounds 1 to 4', 'knn'), ('RT-BarnesHut 1M', 'bh_g1M'), ('RT-BarnesHut 100k', 'bh_g100k'), ('Arkade (3 configs)', 'ark'), ('RTNN real data (F220)', 'zreal_rtnn'), ('Arkade real data (F220)', 'zreal_ark')]
fig, ax = plt.subplots(figsize=(7.6, 4.6)); y0 = np.arange(len(grp))[::-1] * 1.0
for gi, (lab, key) in enumerate(grp):
    R_ = [v for k, v in rows.items() if k.startswith(key) and v]
    for ci, (cn, c) in enumerate((('callbacks on the SM', Or), ('traversal (RT unit)', B), ('other', '#777777'))):
        v = [r[ci] for r in R_ if r[ci] is not None]
        if not v: continue
        lo, hi = min(v), max(v); yy = y0[gi] + (1 - ci) * .25
        if hi - lo < 1e-9: ax.plot([lo], [yy], 'o', color=c, ms=6, label=cn if gi == 0 else None)
        else:
            ax.plot([lo, hi], [yy, yy], '-', color=c, lw=2.5, label=cn if gi == 0 else None); ax.plot([lo, lo, None, hi, hi], [yy - .07, yy + .07, None, yy - .07, yy + .07], '-', color=c, lw=1.5)
        ax.text(hi + 1.5, yy, ('%d' % lo) if hi - lo < 1e-9 else '%d to %d' % (lo, hi), va='center', fontsize=8, color=c)
ax.set_yticks(y0); ax.set_yticklabels([g[0] for g in grp]); ax.set_xlabel('percent of warp time (bar: range of rows, dot: one row)'); ax.set_xlim(0, 100); ax.grid(axis='y', alpha=0)
ax.set_title('Callbacks, not traversal, dominate most RT programs'); ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3)
fig.text(.01, .005, 'F134, F220 (real data rows; Arkade real data: traversal at most 27 percent, other not split; probes lengthen the launch by 23 to 32 percent). RT-BarnesHut 100k: other = 30 is a first launch cost (28). Probes add 0 to 31 percent to launch time (F133).', fontsize=7.5, color=K); save(fig, 'm7_time_breakdown.png')

# 8 Arkade waterfall on io (F145 pinned 24.1 ms, F154 device buffer 3.70 and float square 0.36)
fig, ax = plt.subplots(figsize=(6.6, 4.2)); v = [24.1, 3.70, 0.36]; lab = ['original\n(pinned host output)', 'device buffer\noutput', 'float distance\n(same neighbours)']
ax.bar(range(3), v, color=[Rd, Or, G], width=.55); ax.set_yscale('log'); ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylabel('launch time (ms)'); ax.set_ylim(.1, 60)
for i, x_ in enumerate(v): ax.text(i, x_ * 1.15, '%g ms' % x_, ha='center', fontsize=10)
ax.annotate('6.5 times', (.5, 9), ha='center', color=K, fontsize=10); ax.annotate('10 times', (1.5, 1.1), ha='center', color=K, fontsize=10); ax.grid(axis='x', alpha=0)
ax.set_title('Arkade on the 4070 Ti: two code changes, 67 times faster'); save(fig, 'm8_arkade_waterfall.png')

# ---- 2026-11-17 additions: m9 model table, m10 real data validation, m11 RT against CLOVER. Existing data only (model/model_table.py, results_priorworks_diag.md, results_critical_path.md, results_clover_vs_rt.md).
import importlib.util
def inside(r): return sum(1 for x in r if 0.5 <= x <= 2)
# 9 model table
sets = [('F123 soup 1.5M random', .094, .702, 0), ('F123 Bistro random', .074, .244, 0), ('F123 soup 3M bounce', .072, .318, 0), ('F205 Sponza random', .083, .062, 1), ('F205 Conference random', .014, .064, 1), ('F205 Rungholt random', .132, .421, 1),
        ('F208 Sponza camera', .051, .065, 1), ('F208 Conference camera', .086, .091, 1), ('F208 Rungholt camera', .053, .070, 1), ('F208 Sponza bounce', .031, .021, 1), ('F208 Conference bounce', .055, .061, 1), ('F208 Rungholt bounce', .075, .153, 1)]
fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={'width_ratios': [1.15, 1]}); y = np.arange(len(sets))[::-1]
for yy, (n_, m_, b_, bl) in zip(y, sets):
    ax[0].plot([m_, b_], [yy, yy], color=G if m_ < b_ else Rd, lw=1.2, alpha=.6)
    ax[0].plot(m_, yy, 'o', color=B, mfc=B if bl else 'none', ms=8); ax[0].plot(b_, yy, 's', color=K, ms=7)
ax[0].set_yticks(y); ax[0].set_yticklabels([s[0] for s in sets], fontsize=8.5); ax[0].set_xlabel('rms log error of the predicted speedup'); ax[0].grid(axis='y', alpha=0)
ax[0].plot([], [], 'o', color=B, label='model (filled: predicted first)'); ax[0].plot([], [], 's', color=K, label='no change (predict 1.0)'); ax[0].legend(frameon=False, fontsize=8.5, loc='lower right')
ax[0].set_title('Speedup of a ray order: model better in 10 of 12 sets', fontsize=10.5)
ab = [('core set latency, throughput (4090)', 1.01, 1.04, None), ('one SM, held out scenes (14%)', 1.11, 1.19, 1.14), ('RTNN soups, F127 form', 1.67, 1.67, None), ('TrueKNN soups, launch total (F156)', 1.18, 1.89, None), ('whole GPU, 32 warps (Throughput_6)', 1.16, 1.85, None),
      ('RTNN real data, launch time (F220)', 1.11, 1.67, 1.28), ('RTNN real data, search compute (F207)', 1.69, 2.53, 2.36), ('TrueKNN real data, max with chain (F219)', 1.49, 155, 2.0), ('Arkade real data (F207)', 1.31, 7.69, 3.5), ('Mochi frames (F168)', 2.1, 3.3, None)]
yb = np.arange(len(ab))[::-1]
for yy, (n_, lo, hi, md) in zip(yb, ab):
    ax[1].plot([lo, hi], [yy, yy], '-', color=B, lw=3); ax[1].plot([lo, lo, None, hi, hi], [yy - .12, yy + .12, None, yy - .12, yy + .12], '-', color=B, lw=1.5)
    if md: ax[1].plot(md, yy, 'o', color=Or, ms=7)
    ax[1].text(hi * 1.12, yy, ('%.2g' % lo) if lo == hi else '%.3g to %.3g' % (lo, hi), va='center', fontsize=8, color=K)
ax[1].set_xscale('log'); ax[1].set_xlim(.9, 700); ax[1].axvline(2, color=K, ls=':', lw=1); ax[1].set_yticks(yb); ax[1].set_yticklabels([s[0] for s in ab], fontsize=8.5); ax[1].grid(axis='y', alpha=0)
ax[1].set_xlabel('factor off (bar: range of rows, dot: median)'); ax[1].set_title('Absolute time: 1.1 to 1.9 times on soups,\n2 to 3.5 times on real programs', fontsize=10.5)
fig.suptitle('The model predicts speedups, not absolute time on real programs', fontweight='bold', fontsize=12.5); save(fig, 'm9_model_table.png', rect=(0, 0, 1, .95))

# 10 real data validation, per row ratios of the three forms (measured / predicted)
forms = {'F207 form': {'RTNN': [2.53, 2.23, 2.49, 1.69], 'TrueKNN': [3.12, 161.90, 90.72, 0.56], 'Arkade': [4.04, 1.73, 7.69, 3.22, 3.79, 1.31]},
         'max with slowest warp (F219)': {'RTNN': [2.53, 2.23, 2.49, 1.69], 'TrueKNN': [0.67, 155.39, 0.46, 0.55], 'Arkade': [4.04, 1.73, 7.69, 3.22, 3.79, 1.31]},
         'measured callback cost (F216)': {'RTNN': [1.89, 1.97, 2.03, 2.42], 'TrueKNN': [4.83, 0.59, 127.81, 1.36], 'Arkade': [2.81, 3.48, 2.77, 3.33, 3.41, 4.70]}}
fig, ax = plt.subplots(figsize=(8.6, 5.2)); cols = [K, B, Or]; off = [-.22, 0, .22]
for fi, (fn, d) in enumerate(forms.items()):
    n_in = sum(inside(v) for v in d.values()); n_all = sum(len(v) for v in d.values())
    for pi, (pn, v) in enumerate(d.items()):
        ax.scatter([pi + off[fi] + np.random.default_rng(pi).uniform(-.04, .04, len(v))], [math.log2(x) for x in v], s=36, color=cols[fi], label='%s: %d of %d inside factor 2' % (fn, n_in, n_all) if pi == 0 else None)
ax.axhspan(-1, 1, color=G, alpha=.1); ax.axhline(0, color=K, lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(['RTNN (4 rows)', 'TrueKNN (4 rows)', 'Arkade (6 rows)']); ax.set_ylabel('log2 (measured / predicted)')
ax.legend(frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, .02)); ax.text(2.45, .15, 'factor 2 band', fontsize=8.5, color=G, ha='right'); ax.set_title('Real data (3DRoad, Porto, bunny, dragon, Gowalla, Glove 3D, Manuscript):\nno form closes more than 5 of 14 rows; every form leaves TrueKNN rows 100 times off', fontsize=10.5)
save(fig, 'm10_real_data_validation.png')

# 11 RT against CLOVER (F227): times in ms, log scale
k5 = [('3DRoad\nk 5', dict(warpwise=7022, bitonic=322, hubs=34.2, rt_total=85.8, rt_launch=17.6)), ('Porto 1M\nk 5', dict(warpwise=37801, bitonic=2407, hubs=94.4, rt_total=1282.5, rt_launch=1068.7)), ('uniform 1M\nk 5', dict(warpwise=38850, bitonic=2708, hubs=89.8, rt_total=65.1, rt_launch=9.3)),
      ('bunny\nk 50', dict(warpwise=6076, bitonic=360, hubs=30.9, rt_total=35.5, rt_launch=1.03)), ('dragon\nk 50', dict(warpwise=627000, bitonic=32454, hubs=409.7, rt_total=60.5, rt_launch=7.55))]
series = [('warpwise', K), ('bitonic', '#8a6d3b'), ('hubs', G), ('rt_total', Or), ('rt_launch', '#f0b27a')]
fig, ax = plt.subplots(figsize=(10.4, 5)); w = .16
for si, (s, c) in enumerate(series):
    ax.bar(np.arange(5) + (si - 2) * w, [d[s] for _, d in k5], w, color=c, label={'rt_total': 'RT total (TrueKNN wall; RTNN sort + geometry + search)', 'rt_launch': 'RT launches only', 'bitonic': 'CLOVER bitonic (patched)', 'hubs': 'CLOVER hubs', 'warpwise': 'CLOVER warpwise'}[s])
ax.set_yscale('log'); ax.set_xticks(range(5)); ax.set_xticklabels([n for n, _ in k5]); ax.set_ylabel('time (ms, log)'); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper left')
ax.set_title('RT beats the linear scan by 30 to 600 times, but CLOVER hubs is as fast or faster in 3 of 5 rows\nby total time (RT wins uniform 1M and dragon); TrueKNN at k 5 has 3 to 8 percent wrong queries', fontsize=10.5)
fig.text(.01, .005, 'F227. One GPU (RTX 6000 Ada). CLOVER: its own timer, 1,000 sampled queries 0 wrong. RTNN radius answers not checked; dragon warpwise from 100,000 queries x 36.', fontsize=7.5, color=K); save(fig, 'm11_rt_vs_clover.png', rect=(0, .02, 1, 1))
