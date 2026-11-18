"""Paper figures for notes/paper/characterization.md (numbers from the Findings named in each caption; raw data only where noted).
/usr/bin/python3 model/paper_figs.py OUTDIR      writes fig01..fig09 as PDF (vector) and PNG (300 dpi). Existing data only, no GPU."""
import json, os, statistics as st, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import conc4_analyze
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'figure.dpi': 100})
C = {'ada': '#0072B2', 'ti': '#D55E00', 'rtx4090': '#009E73', 'gray': '#666666', 'sky': '#56B4E9', 'vermil': '#E69F00'}   # Okabe-Ito
GPU = {'ada': 'RTX 6000 Ada', 'ti': '4070 Ti', 'rtx4090': '4090'}
def save(fig, name):
    fig.tight_layout(pad=0.5)
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(OUT, name + '.' + ext), dpi=300)
    plt.close(fig); print('wrote', name)
def bars(ax, groups, series, w=0.8):
    n = len(series); x = np.arange(len(groups))
    for i, (lab, vals, col) in enumerate(series):
        b = ax.bar(x + (i - (n - 1) / 2) * w / n, vals, w / n, label=lab, color=col, linewidth=0)
    ax.set_xticks(x); ax.set_xticklabels(groups)

# 1 latency: (a) second session by BVH levels (F25, F104, F176), (b) one triangle: second session, L2 warm, fully cold (F25, F104, F176)
f, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4))
a.plot([1, 2, 3], [780, 830, 870], 'o-', color=C['ada'], label=GPU['ada'] + ' (F25)'); a.plot([1, 2, 3], [782, 827, 867], 's--', color=C['ti'], label=GPU['ti'] + ' (F104)'); a.plot([1], [782], '^', color=C['rtx4090'], label=GPU['rtx4090'] + ' (F176)')
a.set_xlabel('internal BVH levels'); a.set_ylabel('cycles, second trace of the same ray'); a.set_xticks([1, 2, 3]); a.set_ylim(700, 900); a.legend(frameon=False); a.set_title('(a) L1 resident: 740 + about 43 per level', loc='left')
bars(b, ['second trace\n(L1 warm)', 'L2 warm', 'cold (DRAM)'], [(GPU['ada'], [780, 1641, 2632], C['ada']), (GPU['rtx4090'], [782, 1630, 2627], C['rtx4090'])])
b.set_ylabel('cycles, one triangle'); b.legend(frameon=False); b.set_title('(b) one triangle (F25, F104, F176)', loc='left')
save(f, 'fig01_latency')

# 2 cost per element: latency bound regression (F67/F163, 6000 Ada) against saturated unit rate (F194, 4090)
f, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.3), gridspec_kw={'width_ratios': [4, 3]})
a.bar(['intercept', 'node', 'leaf', 'triangle'], [934, 343, 113, 52], color=C['ada'], linewidth=0); a.set_ylabel('cycles per query / element'); a.set_title('(a) cold launch regression, 60 scenes (F67, F163)', loc='left', fontsize=7)
b.bar(['node', 'leaf slot', 'triangle'], [1.3, 0.0, 0.67], color=C['rtx4090'], linewidth=0); b.set_ylabel('cycles per element'); b.set_title('(b) saturated unit, 4090 (F194)', loc='left', fontsize=7)
b.annotate('about 0', (1, 0.03), ha='center', va='bottom', fontsize=7)
save(f, 'fig02_cost_per_element')

# 3 TPC sharing: (a) pair over single SM against query length (F62, F105, F104), (b) whole chip scaling (F104)
f, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.4))
a.plot([1, 64], [1.53, 1.04], 'o-', color=C['ada'], label=GPU['ada']); a.plot([1, 64], [1.38, 1.04], 's--', color=C['ti'], label=GPU['ti']); a.axhline(2, color=C['gray'], ls=':', lw=0.8); a.text(40, 1.94, 'no sharing', fontsize=6, color=C['gray'])
a.set_xscale('log', base=2); a.set_xlabel('triangles per query'); a.set_ylabel('throughput of an SM pair over one SM'); a.set_ylim(1, 2.1); a.legend(frameon=False, loc='center right'); a.set_title('(a) sibling SMs', loc='left')
b.plot([0, 75], [0, 75], color=C['gray'], ls=':', lw=0.8); b.plot([71], [70.6], 'o', color=C['ada'], label=GPU['ada'] + ', 71 SMs'); b.plot([30], [29.8], 's', color=C['ti'], label=GPU['ti'] + ', 30 SMs (one per TPC)'); b.plot([60], [31.1], 's', mfc='none', color=C['ti'], label=GPU['ti'] + ', 60 SMs (both SMs of each TPC)')
b.set_xlabel('SMs running'); b.set_ylabel('speedup over one SM'); b.legend(frameon=False, loc='upper left'); b.set_title('(b) long coherent queries', loc='left')
save(f, 'fig03_tpc_sharing')

# 4 callback cost: probe cycles per any hit call, original against register heap, unloaded and loaded (F138, 4070 Ti); one warp lone callback 875 (F49)
f, a = plt.subplots(figsize=(3.4, 2.5))
bars(a, ['round 1', 'round 2', 'round 3', 'round 4'], [('original, loaded', [55000, 36000, 36000, 23000], C['ti']), ('original, one warp per SM', [11000, 7900, 6900, 5300], C['vermil']), ('register heap, loaded', [4600, 2900, 2900, 2800], C['ada']), ('register heap, one warp per SM', [2400, 1700, 1500, 1500], C['sky'])])
a.set_yscale('log'); a.set_ylabel('cycles per any hit call'); a.legend(frameon=False, fontsize=6); a.set_title('kNN any hit, 4070 Ti (F138)', loc='left')
save(f, 'fig04_callback_cost')

# 5 partial warp cost per ray against a full warp (F230, 4090)
f, a = plt.subplots(figsize=(3.2, 2.3)); a.plot([32, 16, 8, 1], [1.0, 1.04, 1.15, 5.4], 'o-', color=C['rtx4090']); a.set_xscale('log', base=2); a.set_xticks([1, 8, 16, 32]); a.set_xticklabels(['1', '8', '16', '32']); a.invert_xaxis()
a.set_xlabel('active lanes in the callback warp'); a.set_ylabel('cost per ray, over a full warp'); a.set_title('4090 (F230)', loc='left'); save(f, 'fig05_partial_warp')

# 6 saved fraction against BVH size: 6000 Ada (F150, values in the Finding), 4090 (raw notes/node4090/node_cap, F256). KB = triangles x 64.2 B (F150 sizes 68 KB at 1,024 and 525 KB at 8,192)
ada = [(2, .58), (68, .62), (132, .54), (164, .39), (199, .23), (231, .08), (263, .23), (327, .18), (394, .13), (458, .05), (525, .04), (1024, .03), (4096, .00)]
cap = json.load(open(os.path.join(ROOT, 'notes/node4090/node_cap_20261118T101103.json'))); Ns = sorted({int(k.split('|')[1]) for k in cap})
med = lambda k: st.median(c for x in cap[k] for c in x)
x4 = [n * 64.2 / 1000 for n in Ns]; s4 = [1 - med('twice|%d' % n) / med('single|%d' % n) for n in Ns]; c4 = [1 - med('cctl|%d' % n) / med('single|%d' % n) for n in Ns]
f, a = plt.subplots(figsize=(3.6, 2.6)); a.plot(*zip(*ada), 'o-', color=C['ada'], ms=3, label=GPU['ada'] + ' (F150)'); a.plot(x4, s4, 's-', color=C['rtx4090'], ms=3, label=GPU['rtx4090'] + ' (F256)'); a.plot(x4, c4, 'x:', color=C['gray'], ms=3, label='4090, TTUCCTL between sessions')
a.set_xscale('log'); a.set_xlabel('BVH size (KB)'); a.set_ylabel('second session saving'); a.legend(frameon=False, fontsize=6); save(f, 'fig06_saved_vs_size')

# 7 what removes the saving (68 / 132 KB): F155 to F162, F174 (6000 Ada) and F256 (4090)
arms = ['nothing', 'cta fence\nsibling', 'gpu fence\nsibling', 'gpu fence\nown SM', 'gpu fence\nboth', 'TTUCCTL\nsibling', 'TTUCCTL\nown SM']
A = {68: [.62, .61, .26, .26, .00, .00, .00], 132: [.54, .49, .23, .25, .00, .00, .00]}; G = {68: [.62, .62, .26, .26, .00, .00, .00], 132: [.54, .54, .22, .25, .00, .00, .00]}   # 4090 own TTUCCTL .00; F161 .00 (rerun)
f, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for i, kb in enumerate((68, 132)):
    bars(ax[i], arms, [(GPU['ada'], A[kb], C['ada']), (GPU['rtx4090'], G[kb], C['rtx4090'])]); ax[i].set_title('BVH %d KB' % kb, loc='left'); ax[i].tick_params(axis='x', labelsize=5.5)
ax[0].set_ylabel('second session saving'); ax[0].legend(frameon=False)
save(f, 'fig07_fence_decomposition')
f, a = plt.subplots(figsize=(3.2, 2.3)); bars(a, ['68 KB', '132 KB'], [('Ada, wait only', [.53, .42], C['sky']), ('Ada, 1 MB loads', [.29, .27], C['ada']), ('4090, wait only', [.57, .45], C['vermil']), ('4090, 1 MB loads', [.27, .27], C['rtx4090'])])
a.set_ylabel('second session saving'); a.legend(frameon=False, fontsize=6); a.set_title('own SM L1 stream (F174, F256)', loc='left'); save(f, 'fig07b_own_stream')

# 8 request merging: model against measured L1 sectors per traversal (F135, BarnesHut), two points per size, one held out
f, a = plt.subplots(figsize=(3.2, 2.4)); a.plot([3.17, 4.39], [3.26, 3.86], 'o', color=C['ti']); a.plot([2.5, 5], [2.5, 5], color=C['gray'], ls=':', lw=0.8)
a.annotate('100,000 bodies (post hoc)', (3.17, 3.26), (3.3, 2.9), fontsize=6); a.annotate('1,000,000 (held out)', (4.39, 3.86), (3.2, 4.3), fontsize=6)
a.set_xlabel('measured sectors per traversal'); a.set_ylabel('predicted'); a.set_title('lane sharing model (F135)', loc='left'); save(f, 'fig08_request_merging')

# 9 sibling slowdown (Conc_4 raw, F254): (a) RT ratio against stream footprint, (b) against stream lines in flight
out = conc4_analyze.main(os.path.join(ROOT, 'notes/conc4'))
f, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6))
for pl, col, lab in (('sib', C['rtx4090'], 'stream on TPC siblings'), ('high', C['ti'], 'stream on other TPCs')):
    for thr, ls, mk in ((False, '-', 'o'), (True, '--', 's')):
        s = sorted([o for o in out if o['pl'] == pl and o['thr'] == thr and o['F']], key=lambda o: o['F'])
        if s: a.plot([o['F'] for o in s], [o['r'] for o in s], ls, marker=mk, ms=3, color=col, label='%s%s' % (lab, ', 5 G lines/s limit' if thr else ''))
a.axvline(72, color=C['gray'], ls=':', lw=0.8); a.text(74, 0.95, 'L2 72 MB', fontsize=6, color=C['gray'], va='top'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('stream footprint (MB)'); a.set_ylabel('random ray rate, with stream over alone'); a.legend(frameon=False, fontsize=5.5); a.set_title('(a) 4090, F254', loc='left')
for pl, mk in (('sib', 'o'), ('high', 's')):
    for kind, col, sel in (('L2 (F <= 72 MB)', C['ada'], lambda o: 0 < o['F'] <= 72), ('DRAM (F >= 512 MB)', C['ti'], lambda o: o['F'] >= 512)):
        s = [o for o in out if o['pl'] == pl and sel(o)]; b.scatter([o['O'] for o in s], [o['r'] for o in s], marker=mk, c=col, s=14, label='%s, %s' % (pl, kind))
b.set_xscale('log'); b.set_yscale('log'); b.set_xlabel('stream lines in flight (lines per s x latency)'); b.set_ylabel('ray rate ratio'); b.legend(frameon=False, fontsize=5.5); b.set_title('(b) not one curve (Spearman -0.73)', loc='left')
save(f, 'fig09_sibling_slowdown')
