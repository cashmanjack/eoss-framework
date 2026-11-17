"""Omm_1 interleaved analysis: model/omm_ilv.py notes/omm1. Per process: median over rounds per variant; ratios to A and D within the process; spread of the ratio across the 3 processes."""
import glob, json, os, re, statistics as st, sys
d = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else 'ilv'; P = {}
for f in sorted(glob.glob(os.path.join(d, PFX + '_N*_p[1-9]_*.json'))):
    m = re.match(PFX + r'_N(\d+)_p(\d)_', os.path.basename(f)); P.setdefault(int(m[1]), []).append(json.load(open(f)))
for N, L in sorted(P.items()):
    print('N =', N, '(%d processes, %d rounds, %d rays)' % (len(L), L[0]['rounds'], L[0]['big']))
    print('cfg | ms (median, all procs) | within-process round spread (IQR/median) | time/A (min..max over procs, spread) | time/D | calls/ray | calls vs A | GAS MB | rays differing from A | hit frac')
    for k in L[0]['variants']:
        v = [p['variants'][k] for p in L]; med = [st.median(x['ms']) for x in v]; medA = [st.median(p['variants']['A']['ms']) for p in L]; medD = [st.median(p['variants']['D']['ms']) for p in L]
        rA = [a / b for a, b in zip(med, medA)]; rD = [a / b for a, b in zip(med, medD)]; q = sorted(v[0]['ms']); iqr = (q[3 * len(q) // 4] - q[len(q) // 4]) / st.median(q)
        sp = (max(rA) - min(rA)) / st.median(rA)
        print('%s | %.3f | %.1f%% | %.3f (%.3f..%.3f, %.1f%%)%s | %.3f | %.4f | %.3f | %.2f | %s | %.4f' % (k, st.median(med), 100 * iqr, st.median(rA), min(rA), max(rA), 100 * sp, '' if sp < 0.03 else ' NOISY', st.median(rD), v[0]['calls_mean'],
              v[0]['calls_mean'] / L[0]['variants']['A']['calls_mean'], v[0]['gas_bytes'] / 1e6, ','.join(str(x['differ_from_A']) for x in v), v[0]['hit_frac']))
