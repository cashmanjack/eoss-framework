"""Phase 11 B: turn the fair run lines (notes/q40/fair_*.jsonl, private raw) into the table (2026-11-18).
Usage: python3 -I fair_table.py JSONL [private]    default prints the PUBLIC table (no TrueKNN exactness column); 'private' adds it (for notes/ only).
Time = device time for building the index and searching, data already on the device, result left there (see predictions_phase11.md part B): hubs = CLOVER's own span (includes its index build), warpwise = 10,000 random queries scaled linearly to all N,
RTNN knn mode = query sort + geometry + search compute (median of 3), TrueKNN = printed total of the program's rounds (1 run). Exactness = share of 5,000 sampled queries whose list is exactly the true K nearest (distance ties are not errors); RTNN also lists the share of short lists."""
import sys, json, statistics
def f(x, d=1): return '-' if x is None else ('%.*f' % (d, x))
rows = [json.loads(l) for l in open(sys.argv[1])]; priv = len(sys.argv) > 2 and sys.argv[2] == 'private'
conv = {}
if len(sys.argv) > 3:   # RTNN exactness under two conventions from fair_rtnn_conv.py (truth tree from RTNN's own points)
    for l in open(sys.argv[3]):
        c = json.loads(l)
        if c['per_batch']: conv[(c['ds'], c['K'])] = c['per_batch'][0]
names = {'3droad': '3DRoad', 'porto1m': 'Porto 1M', 'uni1m': 'uniform 1M', 'bunny': 'bunny', 'dragon': 'dragon', 'arkR_gowalla': 'Gowalla', 'arkR_glove3d': 'Glove 3D', 'arkR_manuscript': 'Manuscript'}
hdr = '| dataset | N | K | hubs ms | hubs exact | warpwise ms (scaled) | warpwise wrong in 300 | RTNN knn ms (sort+geom+search) | RTNN exact (coincident points excluded) | RTNN exact (coincident points count) | RTNN short lists | TrueKNN s (rounds)' + (' | TrueKNN exact' if priv else '') + ' | fastest |'
print(hdr); print('|' + '---|' * (hdr.count('|') - 1))
for c in rows:
    h = c['hubs']; w = c['warp']; r = c['rtnn']; t = c['tk']
    rt = [x['total_dev'] for x in r['runs'] if x.get('rc') == 0 and 'total_dev' in x]; rtm = statistics.median(rt) if rt else None
    wm = w['ms_all'][0] if w.get('ms_all') else None
    tk = t['total_s'] * 1000 if t.get('total_s') else None
    cand = {'hubs': h.get('ms_med'), 'warpwise': wm, 'RTNN': rtm, 'TrueKNN': tk}; cand = {k: v for k, v in cand.items() if v}
    best = min(cand, key=cand.get) if cand else '-'
    rtn = f(rtm) if rtm else ('failed rc %s' % r['runs'][0].get('rc') if r.get('runs') else 'not run')
    tks = ('%s (%d)' % (f(t['total_s'], 3), t['rounds'])) if t.get('total_s') else 'did not finish (rc %s)' % t.get('rc')
    cv = conv.get((c['ds'], c['K'])); 
    if cv: r['exact'] = cv['exact_C']; r['exact_B'] = cv['exact_B']; r['short'] = cv['short']
    ex = lambda d, k: (f(100 * d[k], 2) + ' %') if d.get(k) is not None else '-'
    print('| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s%s | %s |' % (names[c['ds']], c['N'], c['K'], f(h.get('ms_med')), ex(h, 'exact'), f(wm, 0), (str(w['check'][1]) if w.get('check') else '-'),
          rtn, ex(r, 'exact'), ex(r, 'exact_B'), ex(r, 'short'), tks, (' | ' + (ex(t, 'exact') if t.get('exact') is not None else '-')) if priv else '', best))
