"""ncu_an.py RUNS_DIR SET WPS K TS_A TS_B... : per ray RT L1 sectors, RT L1 hit rate, L1 fills, L2 reads and hit rate, DRAM bytes, from the two single pass groups of run.py ncu.
Rows in each csv: for every arm (order of ARMS in run.py) the warm launch then the measured launch; the measured launches are used. Group A = L1, B = fills, L2, DRAM."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Memory_4')); import ncu_parse  # noqa: E402
d, SET, W, K = sys.argv[1:5]; ARMS = ['nat_R', 'grp_R', 'pn_R', 'pg_R', 'pp_R', 'nat_T', 'grp_T', 'pn_T', 'pg_T', 'pp_T']
def pick(g):
    fs = sorted(glob.glob('%s/ncu_%s_k%s_w%s_%s_*.csv' % (d, SET, K, W, g))); f = fs[-1]; r = ncu_parse.rows(f); assert len(r) == 20, (f, len(r)); return f, r
fa, A = pick('A'); fb, B = pick('B'); n = json.load(open(sorted(glob.glob('%s/ncuprof_%s_k%s_w%s_A_*.json' % (d, SET, K, W)))[-1]))['rays']
print(os.path.basename(fa), os.path.basename(fb), 'rays', n)
print('%-6s %8s %8s %8s %8s %8s %8s %8s' % ('arm', 'RT L1/r', 'L1hit%', 'fills/r', 'L2rd/r', 'L2hit%', 'DRAM B/r', 'ms(ncu)'))
out = {}
for j, a in enumerate(ARMS):
    x, y = A[2 * j + 1], B[2 * j + 1]; rt = x['l1tex__t_sectors.sum'] - x['l1tex__t_sectors_pipe_lsu.sum']; rth = x['l1tex__t_sectors_lookup_hit.sum'] - x['l1tex__t_sectors_pipe_lsu_lookup_hit.sum']
    r = dict(rt_l1=rt / n, l1_hit=rth / rt, fills=y['l1tex__m_xbar2l1tex_read_sectors.sum'] / n, l2=y['lts__t_sectors_srcunit_tex_op_read.sum'] / n,
             l2_hit=y['lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum'] / y['lts__t_sectors_srcunit_tex_op_read.sum'], dram=y['dram__bytes_read.sum'] / n, ms=x['gpu__time_duration.sum'] / 1e6, lsu=x['l1tex__t_sectors_pipe_lsu.sum'] / n)
    out[a] = r; print('%-6s %8.1f %8.1f %8.1f %8.1f %8.1f %8.0f %8.3f' % (a, r['rt_l1'], 100 * r['l1_hit'], r['fills'], r['l2'], 100 * r['l2_hit'], r['dram'], r['ms']))
json.dump(out, open('%s/ncu_summary_%s_k%s_w%s_%s.json' % (d, SET, K, W, os.path.basename(fa).split('_')[-1].replace('.csv', '')), 'x'))
