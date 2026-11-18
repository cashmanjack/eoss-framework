"""Score the calibrated blind pipeline runs on the 8 F245 datasets (Phase 13; predictions model/priorworks/predictions_phase13.md, commit 915e927). CPU only.
Usage: python3 -I model/priorworks/score_blind_cal.py > model/priorworks/results_blind_cal.md
Reads model/blind/f245_<name>/predictions_*.json and results_*.json (the latest pair per dataset). Variants as pre registered: uncal (cb = cbref, F250), S (stub difference, v1's definition),
D (direct), each at 2 percent (primary) and 10 percent (diagnostic) of the queries. The post hoc variant L (two point linear extrapolation of the 2 and 10 percent real subset launch times) was NOT pre registered."""
import glob, json, os, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'blind')
NAMES = [('3droad', '3DRoad'), ('porto1m', 'Porto 1M'), ('uni1m', 'uniform 1M'), ('bunny', 'bunny'), ('dragon', 'dragon'), ('arkR_gowalla', 'Gowalla'), ('arkR_glove3d', 'Glove 3D'), ('arkR_manuscript', 'Manuscript')]
rows = []
for key, nm in NAMES:
    d = os.path.join(R, 'f245_' + key); pf = sorted(glob.glob(d + '/predictions_*.json')); rf = sorted(glob.glob(d + '/results_*.json'))
    if not pf or not rf: print('missing', key, file=__import__('sys').stderr); continue
    P = json.load(open(pf[-1])); Rr = json.load(open(rf[-1])); N = P['N']
    for K in ('5', '50', '128'):
        p = P['pred'][K]; r = Rr[K]; cp, cd = p.get('cal_p', {}), p.get('cal_d', {}); m = r.get('search_ms')
        if not m or 'ms_S' not in cp: rows.append(dict(ds=nm, K=int(K), ok=False)); continue
        L = None
        if 'T_real_ms' in cd and cd['nq'] != cp['nq']:
            b = (cd['T_real_ms'] - cp['T_real_ms']) / (cd['nq'] - cp['nq']); a = cp['T_real_ms'] - b * cp['nq']; L = max(a + b * N, 1e-6)
        fx = p['fixed_ms']; hubs = r['hubs_ms']; tot = r.get('total_dev_ms'); truth = ('RTNN' if tot < hubs else 'hubs') if tot and hubs else None
        v = dict(uncal=p['uncal']['ms'], S=cp['ms_S'], D=cp['ms_D'], S10=cd.get('ms_S'), D10=cd.get('ms_D'), L=L)
        calls = {k: ('RTNN' if x + fx < hubs else 'hubs') for k, x in v.items() if x is not None}
        rows.append(dict(ds=nm, K=int(K), ok=True, meas=m, v=v, fixed_cal=fx, fixed_meas=r.get('fixed_ms'), hubs=hubs, tot=tot, truth=truth, calls=calls, floor=cp.get('floor_binds_S'), callA=r['callA'], callB=r['callB'], nq=cp['nq']))
ok = [x for x in rows if x['ok']]; n = len(ok)
print('# Calibrated blind pipeline on the 24 F245 cells (Phase 13, triton GPU 1, 2026-11-18)\n')
print('Predictions: predictions_phase13.md (commit 915e927), pipeline code commit 3899e81; each dataset went through blind_pipeline.py (commit order: predictions with the calibration numbers, then hubs and the call, then the full RTNN runs; the commits are in git, files in model/blind/f245_*). NOT blind: the F245 numbers are known. Ratio = measured RTNN search compute (median of 3) over predicted launch; inside 0.5 to 2 is a hit. %d of 24 cells have complete data.\n' % n)
print('| dataset | K | measured ms | uncal | S 2% | D 2% | S 10% | D 10% | L (post hoc) | ratio uncal | ratio S | ratio D | ratio D 10% | ratio L | cal fixed / meas fixed | hubs | RTNN total | winner | call (S) | right |\n|' + '---|' * 20)
f = lambda x: '-' if x is None else ('%.3g' % x)
for x in ok:
    rt = lambda k: None if x['v'][k] is None else x['meas'] / x['v'][k]
    print('| %s | %d | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f / %.1f | %.1f | %s | %s | %s | %s |' % (x['ds'], x['K'], x['meas'], f(x['v']['uncal']), f(x['v']['S']), f(x['v']['D']), f(x['v']['S10']), f(x['v']['D10']), f(x['v']['L']),
          f(rt('uncal')), f(rt('S')), f(rt('D')), f(rt('D10')), f(rt('L')), x['fixed_cal'], x['fixed_meas'] or 0, x['hubs'], '%.1f' % x['tot'] if x['tot'] else '-', x['truth'], x['callB'], 'yes' if x['callB'] == x['truth'] else 'no'))
print()
def stat(k):
    s = [x['meas'] / x['v'][k] for x in ok if x['v'][k] is not None]
    if not s: return None
    s = np.array(s); return int(((s >= .5) & (s <= 2)).sum()), len(s), float(np.median(np.abs(np.log(s)))), float(np.median(s))
print('Within a factor 2 (hits of cells, median absolute log error, median ratio measured / predicted):\n')
for k, lab in (('uncal', 'uncalibrated v1 (F250 form)'), ('S', 'S, 2 percent (pre registered primary)'), ('D', 'D, 2 percent'), ('S10', 'S, 10 percent (diagnostic)'), ('D10', 'D, 10 percent (diagnostic)'), ('L', 'L two point extrapolation (post hoc, not pre registered)')):
    s = stat(k); print('- %s: %s' % (lab, 'no data' if s is None else '%d of %d inside, median |log| %.2f (factor %.1f), median ratio %.2f' % (s[0], s[1], s[2], np.exp(s[2]), s[3])))
print('\nCalibrated fixed part (query sort + point sort + geometry of the 2 percent run) against the measured query sort + geometry of the full run: within 1.5 times in %d of %d cells (ratio range %.2f to %.2f).' % (
    sum(1 for x in ok if x['fixed_meas'] and 1 / 1.5 <= x['fixed_cal'] / x['fixed_meas'] <= 1.5), n, min(x['fixed_cal'] / x['fixed_meas'] for x in ok if x['fixed_meas']), max(x['fixed_cal'] / x['fixed_meas'] for x in ok if x['fixed_meas'])))
print('S floor (real build not slower than the stub on the 2 percent subset, cb set to 0) binds in %d of %d cells.' % (sum(1 for x in ok if x['floor']), n))
ah = sum(x['truth'] == 'hubs' for x in ok); print('\nWinner (RTNN with sort + geometry + search against hubs, measured in this run): RTNN wins %d cells, hubs %d; always hubs is right in %d of %d.' % (n - ah, ah, ah, n))
for k, lab in (('uncal', 'uncalibrated launch + calibrated fixed'), ('S', 'S + calibrated fixed (the pipeline call)'), ('D', 'D 2 percent + calibrated fixed'), ('D10', 'D 10 percent + calibrated fixed'), ('L', 'L + calibrated fixed (post hoc)')):
    c = [x for x in ok if k in x['calls']]
    if c: print('- call %s: right in %d of %d cells (wrong: %s)' % (lab, sum(x['calls'][k] == x['truth'] for x in c), len(c), '; '.join('%s K %d called %s' % (x['ds'], x['K'], x['calls'][k]) for x in c if x['calls'][k] != x['truth']) or 'none'))
print('- call A (S launch only, the old pipeline rule): right in %d of %d cells.' % (sum(x['callA'] == x['truth'] for x in ok), n))
