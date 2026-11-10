"""Memory_4 analysis of one timing run (setup3 file:F rays, --resume K, --passes P).

    analyze.py OUT.json [--pass p] [--json]

Uses pass p (default: the last). Per chunk: launch 0 = first query of every lane, then resume
rounds; a lane takes part in its first nq launches (nq = its query count); finished lanes still run
with tmax 0 in later rounds of their chunk (counted in the time, reported as idle lane queries).
  correct      final closest hit (prim, t) vs the OptiX reference of the same process
  throughput   finished rays per SM cycle = rays / sum over all launches of (last warp end - first warp start)
  latency/ray  per ray, sum over its queries of its warp's query latency (TTUOPEN to last result)
"""
import json, os, statistics as st, struct, sys

f32 = lambda h: struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]


def chunk_files(path, j, p):
    for c in range(j['nchunks']):
        yield path if (p == 0 and c == 0) else '%s.p%dc%d' % (path, p, c)


def launches_per_chunk(path):
    """[(pass, chunk, n_launches)] in launch order, for mapping ncu rows."""
    j = json.load(open(path)); out = []
    for p in range(j['passes']):
        for c, f in enumerate(chunk_files(path, j, p)):
            k = json.load(open(f))
            out.append((p, c, len(k['launches']) + len(k.get('resume_rounds', []))))
    return out


def analyze(path, p=None):
    j = json.load(open(path)); p = j['passes'] - 1 if p is None else p
    L = 32 * j['warps']; ref = j['ref']
    cyc = 0; lat = []; nq_all = []; bad = 0; unfinished = 0; idle = 0; lane_q = 0; first_cyc = 0; qlat = []
    for c, f in enumerate(chunk_files(path, j, p)):
        k = json.load(open(f))
        assert k.get('chunk', c) == c and k['ttu_status'] == 'no error' and k['resume_status'] == 'no error', f
        rounds = [k['launches'][0]['warp_clk']] + [r['warp_clk'] for r in k['resume_rounds']]
        span = [max(e for _, e in r) - min(s for s, _ in r) for r in rounds]
        cyc += sum(span); first_cyc += span[0]
        wl = [[e - s for s, e in r] for r in rounds]
        qlat += [x for r in wl for x in r]
        for i in range(L):
            nq = k['lane_queries'][i]
            if nq < 0:
                unfinished += 1; nq = -nq
            nq_all.append(nq); lane_q += len(rounds)
            idle += len(rounds) - nq
            lat.append(sum(wl[q][i // 32] for q in range(nq)))
            bt, bp = k['lane_best'][i]; r = ref[c * L + i]
            ok = (bt == '0xffffffff') if not r['hit'] else (bt != '0xffffffff' and bp == r['prim'] and abs(f32(bt) - f32(r['t'])) <= 1e-6 * max(1.0, f32(r['t'])))
            bad += not ok
    n = len(nq_all)
    return dict(scene=j['scene'], warps=j['warps'], flush=j['flush_mb'], passes=j['passes'], used_pass=p, rays=n, wrong=bad, unfinished=unfinished,
                queries_per_ray=st.mean(nq_all), max_queries=max(nq_all), idle_lane_queries=idle / lane_q,
                cycles=cyc, rays_per_cycle=n / cyc, cycles_per_ray=cyc / n, first_query_share=first_cyc / cyc,
                lat_mean=st.mean(lat), lat_median=st.median(lat), query_lat_mean=st.mean(qlat))


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    p = int(sys.argv[sys.argv.index('--pass') + 1]) if '--pass' in sys.argv else None
    a = [x for x in a if x != str(p)] if p is not None else a
    for path in a:
        r = analyze(path, p)
        if '--json' in sys.argv:
            print(json.dumps(r)); continue
        print('%-30s %s W%-2d flush %-3g pass %d/%d | %d rays, wrong %d, unfinished %d | queries/ray %.2f (max %d), idle lane queries %.0f%% | '
              '%.4f rays/cycle (%.1f cyc/ray, first queries %.0f%% of time) | latency/ray mean %.0f median %.0f, per query %.0f' % (
                  os.path.basename(path), r['scene'], r['warps'], r['flush'], r['used_pass'], r['passes'], r['rays'], r['wrong'], r['unfinished'],
                  r['queries_per_ray'], r['max_queries'], 100 * r['idle_lane_queries'], r['rays_per_cycle'], r['cycles_per_ray'],
                  100 * r['first_query_share'], r['lat_mean'], r['lat_median'], r['query_lat_mean']))
