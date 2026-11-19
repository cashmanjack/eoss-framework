"""Throughput_6 on the RTX 5090 (pc14): throughput against the number of active SMs, 32 warps per SM, ONE BLOCK PER TPC (SM 2k of TPC k, the assumed pairing) and then all SMs.
    PC14_TTU_APPROVED=... TTU_GPU=0 python3 thr6_pc14.py WORK [scene]       scene: par64 (par:64:s, rays t2, long coherent queries) | tri1 (one triangle, ray hit)
The Ada method (Throughput_6/run.py: the first G blocks) cannot be used: on this GPU the first blocks land on SM pairs, so G blocks would be G/2 TPCs. Here every launch has NSM blocks and an
--smfilter build; the host's --sm-set lets only the chosen SMs run (the others exit before any TTU instruction). Stops at the first fault, wrong result or missing block (core_set.go).
Rate per SM = 1,024 rays / block span (SM cycles); aggregate = sum over blocks; reported with the overlap check of the global timer."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'io_core'))
import core_set as C  # noqa: E402
work = sys.argv[1]; scn = sys.argv[2] if len(sys.argv) > 2 else 'par64'
scene, ray = {'par64': ('par:64:s', 't2'), 'tri1': ('tri1', 'hit')}[scn]
NSM = C.NSM; tab = C.CFG.sm_table; assert tab and len(tab) == NSM
fl = ['--clock', '--sync', '--gtime', '--smfilter', '--remap', '--compact-out', '--regs', '64', '--max-out', '61']
b0, b1 = C.build('t6p', 0, fl), C.build('t6p', 1, fl)
W = 32
def run(b, g, sms, out, warps=W, need=None):
    return C.go(b, ray, out, ['--scene', scene, '--warps', str(warps), '--blocks', str(NSM), '--repeat', '6', '--sm-set', ','.join(map(str, sms))], need_blocks=g if need is None else need)
tpc = lambda k: 2 * k
C.go(b0, ray, work + '/t6p_s0.json', ['--scene', scene, '--warps', str(W), '--blocks', str(NSM), '--repeat', '2', '--sm-set', ','.join(str(i) for i in range(NSM))], need_blocks=NSM)   # stage 0, every block runs
C.go(b1, ray, work + '/t6p_first.json', ['--scene', scene, '--warps', '1', '--blocks', str(NSM), '--repeat', '1', '--sm-set', '0'], need_blocks=1)       # smallest TTU launch: one block, one warp
rows = []
for g in (1, 2, 4, 8, 16, 32, 64, NSM // 2) + (('all',)):
    sms = list(range(NSM)) if g == 'all' else [tpc(k) for k in range(g)]
    gg = NSM if g == 'all' else g
    j = run(b1, gg, sms, '%s/t6p_%s_g%s.json' % (work, scn, g))
    assert C.ref_ok(j)[1] == 0, 'wrong result at G %s' % g
    agg = []; ov = []; per = []
    for l in j['launches'][1:]:
        spans = [max(e for _, e in b['warp_clk']) - min(s for s, _ in b['warp_clk']) for b in l['per_block']]
        agg.append(sum(1024.0 / s for s in spans)); per += [1024.0 / s for s in spans]
        gs = [min(s for s, _ in b['gt']) for b in l['per_block']]; ge = [max(e for _, e in b['gt']) for b in l['per_block']]; ov.append(min(ge) > max(gs))
    rows.append(dict(G=g, blocks=gg, agg=st.median(agg), per_sm=st.median(per), overlap='%d/%d' % (sum(ov), len(ov))))
    print('   ', rows[-1], flush=True)
base = rows[0]['agg']
for r in rows: r['x_one_sm'] = r['agg'] / base
json.dump(rows, open('%s/thr6_%s.json' % (work, scn), 'w'))
print('G, blocks, rays/cycle/SM, aggregate x one SM, overlap'); [print(r['G'], r['blocks'], round(r['per_sm'], 5), round(r['x_one_sm'], 2), r['overlap']) for r in rows]
