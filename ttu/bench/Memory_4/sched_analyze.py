"""Memory_4a: analysis of setup3 --sched runs (one JSON per run).

    sched_analyze.py OUT.json ...   [--json]

correct: every ray's closest hit vs the OptiX reference. Cost per finished ray = sum of launch spans
(first warp start to last warp end) / rays; latency per ray = sum over its queries of its warp's latency.
Lane use = active lanes / (launches x 32 W): the share of lane slots doing a real query.
"""
import json, os, statistics as st, struct, sys

f32 = lambda h: struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]


def analyze(p):
    j = json.load(open(p)); ref = j['ref']; n = len(ref)
    bad = 0
    for r, (bt, bp) in enumerate(j['ray_best']):
        h = ref[r]
        ok = (bt == '0xffffffff') if not h['hit'] else (bt != '0xffffffff' and bp == h['prim'] and abs(f32(bt) - f32(h['t'])) <= 1e-6 * max(1.0, f32(h['t'])))
        bad += not ok
    cyc = 0; act = 0; nl = 0; per_round = []
    for rd in j['rounds']:
        c = sum(max(e for _, e in l['warp_clk']) - min(s for s, _ in l['warp_clk']) for l in rd['launches'])
        cyc += c; per_round.append((len(rd['launches']), c)); act += sum(l['active'] for l in rd['launches']); nl += len(rd['launches'])
    q = [abs(x) for x in j['ray_queries']]
    return dict(file=os.path.basename(p), scene=j['scene'], warps=j['warps'], sched=j['sched'], flush=j['flush_mb'], rays=n, wrong=bad,
                unfinished=sum(x < 0 for x in j['ray_queries']), queries_per_ray=st.mean(q), max_queries=max(q), rounds=len(j['rounds']),
                launches=nl, lane_use=act / (nl * 32 * j['warps']), cycles_per_ray=cyc / n, rays_per_cycle=n / cyc,
                lat_mean=st.mean(j['ray_lat']), round_launches=[a for a, _ in per_round], round_share=[c / cyc for _, c in per_round])


if __name__ == '__main__':
    for p in [a for a in sys.argv[1:] if not a.startswith('--')]:
        r = analyze(p)
        if '--json' in sys.argv:
            print(json.dumps(r)); continue
        print('%-32s %s W%-2d %-7s flush %-3g | %d rays, wrong %d, unfinished %d | q/ray %.3f (max %d), %d rounds, %d launches, lane use %.0f%% | '
              '%.1f cyc/ray (%.4f rays/cyc) | lat/ray %.0f | round launches %s' % (
                  r['file'], r['scene'], r['warps'], r['sched'], r['flush'], r['rays'], r['wrong'], r['unfinished'], r['queries_per_ray'], r['max_queries'],
                  r['rounds'], r['launches'], 100 * r['lane_use'], r['cycles_per_ray'], r['rays_per_cycle'], r['lat_mean'], r['round_launches']))
