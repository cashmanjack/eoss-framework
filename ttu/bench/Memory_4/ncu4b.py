"""Memory_4b ncu: RT core traffic per finished ray for the --sched runs of run4b.py.

    ncu4b.py WORK/s3000000/n
"""
import json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncu_parse  # noqa: E402


def rows(base):
    r = ncu_parse.rows(base + '.csv'); j = json.load(open(base + '.json'))
    assert len(r) == j['launches_total'], (base, len(r), j['launches_total'])
    assert all(x['profiler__replayer_passes'] == 1 for x in r)
    return r, len(j['ref'])


d = sys.argv[1]; out = []
for stt in ('warm', 'cold'):
    bA, _ = rows(os.path.join(d, 'stage0_%s_A' % stt)); bB, _ = rows(os.path.join(d, 'stage0_%s_B' % stt))
    per = lambda R, k: st.mean(x[k] for x in R); tot = lambda R, k: sum(x[k] for x in R)
    for o in ('interleaved', 'grouped'):
        for m in ('inplace', 'compact'):
            A, n = rows(os.path.join(d, '%s_%s_%s_A' % (o, m, stt))); B, _ = rows(os.path.join(d, '%s_%s_%s_B' % (o, m, stt)))
            l1 = tot(A, 'l1tex__t_sectors.sum') - tot(A, 'l1tex__t_sectors_pipe_lsu.sum')
            l1h = tot(A, 'l1tex__t_sectors_lookup_hit.sum') - tot(A, 'l1tex__t_sectors_pipe_lsu_lookup_hit.sum')
            sub = lambda k: tot(B, k) - len(B) * per(bB, k)
            r = dict(state=stt, order=o, sched=m, rays=n, launches=len(A), lsu_vs_base=(tot(A, 'l1tex__t_sectors_pipe_lsu.sum') / len(A)) / per(bA, 'l1tex__t_sectors_pipe_lsu.sum'),
                     l1=l1 / n, l1_hit=l1h / l1, fill=sub('l1tex__m_xbar2l1tex_read_sectors.sum') / n, l2=sub('lts__t_sectors_srcunit_tex_op_read.sum') / n,
                     l2_hit=sub('lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum') / sub('lts__t_sectors_srcunit_tex_op_read.sum'),
                     dram=sub('dram__bytes_read.sum') / n, dram_raw=tot(B, 'dram__bytes_read.sum') / n)
            k = (tot(A, 'l1tex__t_sectors_pipe_lsu.sum') / len(A)) / per(bA, 'l1tex__t_sectors_pipe_lsu.sum')   # template traffic scaled by its LSU sectors
            r['dram_scaled'] = (tot(B, 'dram__bytes_read.sum') - k * len(B) * per(bB, 'dram__bytes_read.sum')) / n
            out.append(r)
            print('%-4s %-11s %-7s | %d rays, %d launches, LSU/base %.2f | RT L1 %.1f sect/ray, hit %.1f%% | fills %.1f | L2 reads %.1f, hit %.1f%% | DRAM %.0f B/ray (LSU scaled baseline %.0f, raw %.0f)' % (
                stt, o, m, n, r['launches'], r['lsu_vs_base'], r['l1'], 100 * r['l1_hit'], r['fill'], r['l2'], 100 * r['l2_hit'], r['dram'], r['dram_scaled'], r['dram_raw']))
json.dump(out, open(os.path.join(d, 'ncu4b.json'), 'w'))
