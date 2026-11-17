"""Arkade callback load latency matrix (2026-11-17): table of the LDP / CBK / CHASE lines, median of the repeats. Raw: notes/q37/raw (private).
Usage: /usr/bin/python3 model/priorworks/analyze_ldp.py > model/priorworks/results_arkade_callback_loads.md"""
import glob, re, collections, numpy as np
R = '/home/min/a/cashman3/eoss/notes/q37/raw/'; G = collections.defaultdict(list)
for f in sorted(glob.glob(R + 'ldp_*.txt')):
    m = re.search(r'ldp_(ld0?)_(\w+)_q(\d+)_c(\d+)_', f); b, ds, q, c = m[1], m[2], int(m[3]), int(m[4]); s = open(f).read()
    g = lambda pat: float(re.search(pat, s)[1]) if re.search(pat, s) else float('nan')
    d = dict(ms=g(r'KERNEL queries \d+ ms (\S+)'), a=g(r'a_cyc_per_call (\S+)'), b=g(r'b_cyc_per_exec (\S+)'), bn=g(r'b_execs (\S+)'), an=g(r'a_calls (\S+)'),
             wc=g(r'f0_cycles (\S+)') / g(r'f0_calls (\S+)'), ch_alone=g(r'alone cycles_per_step (\S+)'), ch_dur=g(r'during_launch cycles_per_step (\S+)'), sms=g(r'sms_used (\S+)'), wpsm=g(r'mean_warps_in_flight_per_used_sm (\S+)'),
             life=g(r'mean_life_ms (\S+)'), q50a=g(r'a_q50 (\S+)'), q50b=g(r'b_q50 (\S+)'))
    G[(b, ds, q, c)].append(d)
md = lambda L, k: float(np.median([x[k] for x in L]))
print('# Arkade callback loads under load (float build, device buffer, KN 50, NORM 2; 2026-11-17, triton GPU 1; median of 3 runs; cycles are clock64 SM cycles)\n')
print('a = dependent load of the primitive data (data_pts[primID]) per callback lane call; b = scan of the 50 result slots per execution (49 dependent loads on the payload); warp call = CB probe: cycles of one whole callback call of a warp (leader lane).')
print('Chase level 0 = none; 1 = 18 blocks, 8 = 144 blocks of 128 threads (one chain per thread, 1 GiB) started after the warm up launch. Chase cycles per step: thread 0 of each block, alone before the launch / during the timed launch.\n')
print('| dataset | queries | chase | launch ms | a cycles/call | b cycles/exec | whole warp call cycles | a / a(32 queries, no chase) | b / b(32, no chase) | warp call / warp call (32, no chase) | warps per used SM | chase cycles/step alone | during |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for ds in ('manuscript', 'gowalla'):
    base = G[('ld', ds, 32, 0)]; a0, b0, w0 = md(base, 'a'), md(base, 'b'), md(base, 'wc')
    for q in (32, 320, 3200, 10000, 40000, 160000):
        for c in (0, 1, 8):
            L = G.get(('ld', ds, q, c))
            if not L: continue
            print('| %s | %d | %d | %.3f | %.0f | %.0f | %.0f | %.2f | %.2f | %.2f | %.1f | %s | %s |' % (ds, q, c, md(L, 'ms'), md(L, 'a'), md(L, 'b'), md(L, 'wc'), md(L, 'a') / a0, md(L, 'b') / b0, md(L, 'wc') / w0, md(L, 'wpsm'),
                  '%.0f' % md(L, 'ch_alone') if c else '', '%.0f' % md(L, 'ch_dur') if c else ''))
print('\n## Probe overhead: the same code without the segment probes (ld0: only the CB probes) against the probe build, no chase\n')
print('| dataset | queries | launch ms probe build | launch ms without segment probes | warp call cycles probe build | without | probe overhead on a warp call |\n|---|---|---|---|---|---|---|')
for ds in ('manuscript', 'gowalla'):
    for q in (10000, 160000):
        A = G.get(('ld', ds, q, 0)); B = G.get(('ld0', ds, q, 0))
        if A and B: print('| %s | %d | %.3f | %.3f | %.0f | %.0f | %.2f |' % (ds, q, md(A, 'ms'), md(B, 'ms'), md(A, 'wc'), md(B, 'wc'), md(A, 'wc') / md(B, 'wc')))
print('\n## Share of one warp call taken by (a) + (b) at 32 queries, no chase, and growth of the rest\n')
for ds in ('manuscript', 'gowalla'):
    for q in (32, 10000, 160000):
        L = G[('ld', ds, q, 0)]; fb = md(L, 'bn') / md(L, 'an'); sh = (md(L, 'a') + fb * md(L, 'b')) / md(L, 'wc')
        print('%s q %d: b executes in %.2f of the lane calls; (a + that share of b) / warp call cycles = %.2f (lane cycles of a serial lane vs warp call cycles, see note)' % (ds, q, fb, sh))
