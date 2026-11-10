"""Memory_4: memory traffic per finished ray from the ncu runs of run.py.

    ncu_analyze.py WORK/s<N>/n [--json]

For each W, state, order: rows of the used pass (warm: pass 1 of 2; cold: the only pass) of
group A (L1) and group B (L2, DRAM). RT core L1 traffic = all L1 sectors minus LSU pipe sectors
(our template's own loads and stores are LSU). L2 and DRAM: minus the stage 0 baseline per launch
(same template, no TTU) times the number of launches. Sectors are 32 bytes.
"""
import glob, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncu_parse  # noqa: E402
from analyze import launches_per_chunk  # noqa: E402


def used_rows(base):
    j = json.load(open(base + '.json')); rows = ncu_parse.rows(base + '.csv')
    lp = launches_per_chunk(base + '.json')
    p = j['passes'] - 1; out = []; i = 0
    if len(rows) == sum(n for pp, _, n in lp if pp == p) and len(rows) < sum(n for _, _, n in lp):   # run with --ncu-skip: only the last pass
        lp = [x for x in lp if x[0] == p]
    assert sum(n for _, _, n in lp) == len(rows), (base, sum(n for _, _, n in lp), len(rows))
    for pp, c, n in lp:
        if pp == p:
            out += rows[i:i + n]
        i += n
    assert all(r['profiler__replayer_passes'] == 1 for r in rows), 'more than one replay pass'
    return out, 32 * j['warps'] * j['nchunks']


def total(rows, k):
    return sum(r[k] for r in rows)


def one(d, w, stt, order):
    A, nr = used_rows(os.path.join(d, '%s_w%d_%s_A' % (order, w, stt)))
    B, _ = used_rows(os.path.join(d, '%s_w%d_%s_B' % (order, w, stt)))
    bA, _ = used_rows(os.path.join(d, 'base_w%d_%s_A' % (w, stt)))
    bB, _ = used_rows(os.path.join(d, 'base_w%d_%s_B' % (w, stt)))
    per = lambda rows, k: st.mean(r[k] for r in rows)
    l1 = total(A, 'l1tex__t_sectors.sum') - total(A, 'l1tex__t_sectors_pipe_lsu.sum')
    l1hit = total(A, 'l1tex__t_sectors_lookup_hit.sum') - total(A, 'l1tex__t_sectors_pipe_lsu_lookup_hit.sum')
    lsu_ratio = (total(A, 'l1tex__t_sectors_pipe_lsu.sum') / len(A)) / per(bA, 'l1tex__t_sectors_pipe_lsu.sum')
    sub = lambda k: total(B, k) - len(B) * per(bB, k)
    fill = sub('l1tex__m_xbar2l1tex_read_sectors.sum')
    l2r = sub('lts__t_sectors_srcunit_tex_op_read.sum'); l2h = sub('lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum')
    dram = sub('dram__bytes_read.sum')
    return dict(w=w, state=stt, order=order, rays=nr, launches=len(A), launches_B=len(B), lsu_vs_baseline=lsu_ratio,
                l1_sectors_per_ray=l1 / nr, l1_hit=l1hit / l1, l1_fill_sectors_per_ray=fill / nr,
                l2_read_sectors_per_ray=l2r / nr, l2_hit=l2h / l2r if l2r else float('nan'), dram_bytes_per_ray=dram / nr)


if __name__ == '__main__':
    d = sys.argv[1]
    for w in (4, 8, 16):
        for stt in ('warm', 'cold'):
            for o in ('grouped', 'interleaved', 'b_grouped_warp', 'b_grouped_ray', 'b_interleaved_ray'):
                if not glob.glob(os.path.join(d, '%s_w%d_%s_B.csv' % (o, w, stt))):
                    continue
                r = one(d, w, stt, o)
                if '--json' in sys.argv:
                    print(json.dumps(r)); continue
                print('W%-2d %-4s %-11s | %d rays, %d launches (B %d), LSU/baseline %.3f | RT L1 %.1f sectors/ray, hit %.1f%% | '
                      'L1 fills %.1f sectors/ray | L2 reads %.1f sectors/ray, hit %.1f%% | DRAM %.0f B/ray' % (
                          w, stt, o, r['rays'], r['launches'], r['launches_B'], r['lsu_vs_baseline'], r['l1_sectors_per_ray'], 100 * r['l1_hit'],
                          r['l1_fill_sectors_per_ray'], r['l2_read_sectors_per_ray'], 100 * r['l2_hit'], r['dram_bytes_per_ray']))
