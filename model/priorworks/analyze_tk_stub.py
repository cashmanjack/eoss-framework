"""Phase 15 item 3 scoring: TrueKNN replay bound with a free callback at K 50 and 128 (predictions model/priorworks/predictions_phase15.md, e57d74e).
Usage: /usr/bin/python3 model/priorworks/analyze_tk_stub.py notes/q45/tkstub_TS.jsonl > model/priorworks/results_tk_stub.md"""
import json, sys, statistics, re
NAME = {'3droad': '3DRoad', 'porto1m': 'Porto 1M', 'uni1m': 'uniform 1M', 'bunny': 'bunny', 'dragon': 'dragon', 'arkR_gowalla': 'Gowalla', 'arkR_glove3d': 'Glove 3D', 'arkR_manuscript': 'Manuscript'}
fair = {}
for l in open('/home/min/a/cashman3/eoss/model/priorworks/results_fair_table.md'):
    c = [x.strip() for x in l.split('|')]
    if len(c) > 9 and c[2].isdigit() and c[3].isdigit() and re.match(r'^[\d.]+$', c[4]) and c[1] in NAME.values() and c[8] in ('hubs', 'RTNN', 'TrueKNN') and (c[1], int(c[3])) not in fair:
        fair[(c[1], int(c[3]))] = dict(hubs=float(c[4]), rtnn=float(c[6]), tk=float(c[7]))
cells = [json.loads(l) for l in open(sys.argv[1])]
print('# TrueKNN at K 50 and 128: replay bound with a free callback (2026-11-19, triton GPU 1; file %s)\n' % sys.argv[1].split('/')[-1])
print('Original = Vani\'s program, recorder build (adds one byte store per query per round in raygen), device scope = build + launches + refits (F247). Stub = intersection program returns at once, same radii, same refits, masks of the original replayed (raygen reads the mask). Stub median of 3. Hubs and RTNN knn from model/priorworks/results_fair_table.md (F245); F245 TrueKNN device column for reference. ms.\n')
print('| dataset | K | rounds | rounds, radii and active counts equal in all stub runs | original (this session) | F245 TrueKNN | stub median (min to max) | stub / original | hubs | RTNN knn | stub < hubs | stub < RTNN | stub build + refit share | callback cost a fix may leave for TrueKNN to tie hubs (share of the current callback cost) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
rows = []
for c in cells:
    n = NAME[c['ds']]; f = fair.get((n, c['K'])); o = c['rec']['device_scope_s'] * 1e3 if c['rec'].get('device_scope_s') else None
    if not c['stub'] or o is None: print('| %s | %d | | not run (rc %s) | %s | | | | | | | | | |' % (n, c['K'], c['rec']['rc'], o)); continue
    s = [x['device_scope_s'] * 1e3 for x in c['stub']]; sm = statistics.median(s); same = all(x['same_rounds'] and x['same_active'] and x['same_radii'] for x in c['stub'] ); bs = statistics.median((x['build_s'] + x['refit_s']) * 1e3 for x in c['stub'])
    rows.append((n, c['K'], o, sm, f, same))
    allow = '%.1f percent' % (100 * (f['hubs'] - sm) / (o - sm)) if sm < f['hubs'] else 'none (the stub alone is slower)'
    print('| %s | %d | %d | %s | %.1f | %.1f | %.1f (%.1f to %.1f) | %.3f | %.1f | %.1f | %s | %s | %.2f | %s |' % (n, c['K'], c['rec']['rounds'], 'yes' if same else 'NO', o, f['tk'], sm, min(s), max(s), sm / o, f['hubs'], f['rtnn'], 'yes' if sm < f['hubs'] else 'no', 'yes' if sm < f['rtnn'] else 'no', bs / sm, allow))
print()
for K in (50, 128):
    R = [r for r in rows if r[1] == K]
    if not R: continue
    print('K %d: cells run %d; stub below hubs in %d; stub below RTNN in %d; stub below the original in %d; replay exact in %d; median stub / original %.3f (range %.3f to %.3f).' % (K, len(R), sum(r[3] < r[4]['hubs'] for r in R), sum(r[3] < r[4]['rtnn'] for r in R), sum(r[3] < r[2] for r in R), sum(r[5] for r in R), statistics.median(r[3] / r[2] for r in R), min(r[3] / r[2] for r in R), max(r[3] / r[2] for r in R)))
R = rows
print('All: stub above hubs in %d of %d cells; stub above RTNN in %d of %d.' % (sum(r[3] > r[4]['hubs'] for r in R), len(R), sum(r[3] > r[4]['rtnn'] for r in R), len(R)))
