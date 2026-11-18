"""Winner prediction with model v1 unchanged plus the replay counts on the 24 F245 cells (2026-11-18). CPU only. NOT BLIND: the F245 times are known (predictions_phase12.md part D, committed before this ran).
Usage: python3 -I model/priorworks/winner_v1.py PREP.json FAIR_JSONL > model/priorworks/results_winner_v1.md
Model: model_v1_rtnn.py (frozen v1 RTNN form, callback cost = the uniform reference's, K generalized from 50). Predicted = search launch only. Measured = RTNN's search compute (the quantity v1 predicts), median of 3 from F245.
Winner call A: RTNN if predicted launch < measured hubs, else hubs. Call B: RTNN if predicted launch + measured sort + geometry (NOT predicted) < measured hubs. Truth: RTNN (sort + geometry + search, F245 device scope) against hubs (F245)."""
import sys, json, statistics as st, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import model_v1_rtnn as m1
D = '/local/scratch/a/cashman3/thirdparty/pw/data/'
prep = json.load(open(sys.argv[1])); fair = {(c['ds'], c['K']): c for c in map(json.loads, open(sys.argv[2]))}
names = {'3droad': '3DRoad', 'porto1m': 'Porto 1M', 'uni1m': 'uniform 1M', 'bunny': 'bunny', 'dragon': 'dragon', 'arkR_gowalla': 'Gowalla', 'arkR_glove3d': 'Glove 3D', 'arkR_manuscript': 'Manuscript'}
rows = []
for ds, info in prep.items():
    N = info['N']; pts = np.array(open(D + info['file']).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)[:N].astype(np.float32).astype(np.float64); tree = cKDTree(pts)
    for K in (5, 50, 128):
        r = info['r99'][str(K)]; C, S = m1.replay_counts(pts, r, tree); p = m1.predict(N, K, C, S); c = fair[(ds, K)]
        runs = [x for x in c['rtnn']['runs'] if x.get('rc') == 0 and 'total_dev' in x]; srch = st.median(x['search'] for x in runs); fixed = st.median(x['sort'] + x['geom'] for x in runs); tot = srch + fixed; hub = c['hubs']['ms_med']
        rows.append(dict(ds=names[ds], K=K, pred=p['ms'], thr=p['thr_ms'], chain=p['chain_ms'], meanq=p['mean_q'], maxq=p['max_q'], meas=srch, fixed=fixed, total=tot, hubs=hub, callA='RTNN' if p['ms'] < hub else 'hubs', callB='RTNN' if p['ms'] + fixed < hub else 'hubs', truth='RTNN' if tot < hub else 'hubs'))
        print(ds, K, file=sys.stderr, flush=True)
lg = lambda a, b: abs(np.log(a / b))
print('# Winner prediction with model v1 unchanged (2026-11-18, triton, CPU only, NOT BLIND: F245 is known)\n')
print('Predictions: predictions_phase12.md part D (commit 9fcc6c3). Model: model_v1_rtnn.py (v1 RTNN form, callback cost = uniform reference, K generalized from 50), replay counts from a kd tree on the F245 points and r99 radii. Units ms. Predicted = search launch only; measured = RTNN search compute (F245 median of 3).\n')
print('| dataset | K | mean q | max q | throughput | chain | predicted launch | measured search | measured / predicted | RTNN sort+geom (not predicted) | RTNN total | hubs | call A | call B | actual winner |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print('| %s | %d | %.1f | %.0f | %.3f | %.3f | %.3f | %.2f | %.2f | %.1f | %.1f | %.1f | %s | %s | %s |' % (r['ds'], r['K'], r['meanq'], r['maxq'], r['thr'], r['chain'], r['pred'], r['meas'], r['meas'] / r['pred'], r['fixed'], r['total'], r['hubs'], r['callA'], r['callB'], r['truth']))
ratio = np.array([r['meas'] / r['pred'] for r in rows]); inside = int(((ratio >= .5) & (ratio <= 2)).sum()); n = len(rows)
print('\nLaunch time: %d of %d cells inside a factor 2; median absolute log error %.2f (factor %.1f); median ratio %.2f; chain term binds in %d cells.' % (inside, n, np.median(np.abs(np.log(ratio))), np.exp(np.median(np.abs(np.log(ratio)))), np.median(ratio), sum(r['chain'] > r['thr'] for r in rows)))
for nm, key in (('A (launch only against hubs)', 'callA'), ('B (launch + measured sort and geometry against hubs)', 'callB')):
    ok = sum(r[key] == r['truth'] for r in rows); print('Winner, call %s: right in %d of %d cells.' % (nm, ok, n))
print('Baseline "always hubs": right in %d of %d cells. Truth: RTNN wins %d cells.' % (sum(r['truth'] == 'hubs' for r in rows), n, sum(r['truth'] == 'RTNN' for r in rows)))
for key in ('callA', 'callB'):
    wrong = [(r['ds'], r['K'], r[key], r['truth']) for r in rows if r[key] != r['truth']]; print('Wrong calls %s: %s' % (key, '; '.join('%s K %d (called %s, truth %s)' % w for w in wrong) or 'none'))
