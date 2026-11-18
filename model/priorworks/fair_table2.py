"""Phase 12 A: the F245 table with ONE timing scope (2026-11-18). Public output (no TrueKNN exactness).
Usage: python3 -I fair_table2.py FAIR_JSONL(q40, 24 cells) SCOPE_JSONL(q42) RAW_DIR_q40 CONV_JSONL
Device scope (all methods): data already on the device, index or BVH build plus search, no host set up, no result copy.
 hubs, warpwise: CLOVER's own span (device synchronized, includes hubs' index build); warpwise = 10,000 random queries scaled linearly to all N.
 RTNN: query sort + create and upload geometry + search compute (its own timers, median of 3 runs of F245).
 TrueKNN: BVH build + every round's launch (device synchronized) + every round's refit (device synchronized), from a copy of the original sample with timers added (results identical); 1 run.
Host column (separate): hubs and warpwise = host wall time of CLOVER's whole call (allocation, host to device copies, search, result copy back; median of reps, warpwise for its 10,000 queries not scaled);
 RTNN = 'time total search time' minus the three device timers (mostly the result copy to the host; median of 3); TrueKNN = whole loop total printed by the sample minus the device scope (host scan of the managed buffer, parameter updates, synchronization gaps)."""
import sys, json, re, statistics, glob
def f(x, d=1): return '-' if x is None else ('%.*f' % (d, x))
med = statistics.median
fair = [json.loads(l) for l in open(sys.argv[1])]; scope = {(c['ds'], c['K']): c for c in map(json.loads, open(sys.argv[2]))}; RAW = sys.argv[3]
conv = {}
for l in open(sys.argv[4]):
    c = json.loads(l)
    if c['per_batch']: conv[(c['ds'], c['K'])] = c['per_batch'][0]
names = {'3droad': '3DRoad', 'porto1m': 'Porto 1M', 'uni1m': 'uniform 1M', 'bunny': 'bunny', 'dragon': 'dragon', 'arkR_gowalla': 'Gowalla', 'arkR_glove3d': 'Glove 3D', 'arkR_manuscript': 'Manuscript'}
T1 = ['| dataset | N | K | hubs | warpwise (scaled) | RTNN knn | TrueKNN | fastest | hubs host | warpwise host (10,000 q) | RTNN host | TrueKNN host | TrueKNN old total |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
T2 = ['| dataset | K | hubs exact | warpwise wrong in 300 | RTNN exact (coincident points excluded) | RTNN exact (coincident points count) | RTNN short lists |', '|---|---|---|---|---|---|---|']
labels = []
for c in fair:
    ds, K = c['ds'], c['K']; s = scope[(ds, K)]; h = c['hubs']; w = c['warp']; r = c['rtnn']
    runs = [x for x in r['runs'] if x.get('rc') == 0 and 'total_dev' in x]; rtm = med([x['total_dev'] for x in runs]) if runs else None
    # RTNN host part from the raw files of the three runs
    tag = None; hosts = []
    for i in range(3):
        fn = glob.glob('%s/rtnn_%s_k%d_*_%d.txt' % (RAW, ds, K, i)); 
        if not fn: continue
        t = open(fn[-1]).read(); m = re.search(r'time total search time: ([\d.e+-]+) ms', t)
        if m and i < len(runs): hosts.append(float(m[1]) - runs[i]['total_dev'])
    rth = med(hosts) if hosts else None
    hub = med(s['hubs']['dev_ms']) if s['hubs']['dev_ms'] else h['ms_med']; hubh = med(s['hubs']['host_ms']) if s['hubs']['host_ms'] else None
    wm = w['ms_all'][0] if w.get('ms_all') else None; wh = med(s['warp']['host_ms']) if s['warp']['host_ms'] else None
    t = s['tk']; tkd = t['device_scope_s'] * 1000 if t.get('device_scope_s') is not None else None; tkh = (t['total_s'] - t['device_scope_s']) * 1000 if tkd is not None and t.get('total_s') else None; tko = t['total_s'] * 1000 if t.get('total_s') else None
    cand = {'hubs': h['ms_med'], 'warpwise': wm, 'RTNN': rtm, 'TrueKNN': tkd}; cand = {k: v for k, v in cand.items() if v}; best = min(cand, key=cand.get)
    cand_old = dict(cand, TrueKNN=tko) if tko else {k: v for k, v in cand.items() if k != 'TrueKNN'}; best_old = min({k: v for k, v in cand_old.items() if v}, key=lambda k: cand_old[k])
    labels.append((names[ds], K, best_old, best))
    T1.append('| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (names[ds], c['N'], K, f(h['ms_med']), f(wm, 0), f(rtm), f(tkd), best, f(hubh), f(wh), f(rth), f(tkh), f(tko)))
    cv = conv.get((ds, K)) or {}
    ex = lambda x: (f(100 * x, 2) + ' %') if x is not None else '-'
    T2.append('| %s | %d | %s | %s | %s | %s | %s |' % (names[ds], K, ex(h.get('exact')), (str(w['check'][1]) if w.get('check') else '-'), ex(cv.get('exact_C')), ex(cv.get('exact_B')), ex(cv.get('short'))))
print('All times in ms, device scope unless the column says host. hubs here is the F245 median (the rerun with the host timer agrees, see notes).\n'); print('\n'.join(T1)); print(); print('\n'.join(T2))
ch = [(a, b, o, n) for a, b, o, n in labels if o != n]
print('\nFastest label changed by the scope (old TrueKNN total -> device scope): %d of %d cells: %s' % (len(ch), len(labels), '; '.join('%s K %d: %s to %s' % x for x in ch) or 'none'))
