"""Scores model/predictions_4090_batch6.md (commit 25789e1, before the runs): ncu csv from notes/ncu4090 (private raw), plain event times from notes/p10f/raw (F255 raw). python3 model/score_4090_batch6.py"""
import csv, glob, io, re, statistics as st
def load(f):
    t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for r in rows: d.setdefault(int(r['ID']), {'n': r['Kernel Name'][:30]})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return [v for k, v in sorted(d.items()) if v['n'] == 'optixLaunch']
def files(p): return sorted(glob.glob('notes/ncu4090/ncu_%s_2*.csv' % p))
med = st.median
print('RTNN (ncu optixLaunch with 100,000 rays), plain medians from notes/p10f/raw')
for n, pred in (('r05', 0.191), ('r025', 0.058)):
    for f in files('rtnn_' + n):
        L = [x for x in load(f) if x['launch__grid_size'] > 500]; print(f, [round(x['gpu__time_duration.sum'] / 1e6, 4) for x in L], 'all launches', len(load(f)))
    k = [x for x in load(files('rtnn_' + n)[-1]) if x['launch__grid_size'] > 500][0]['gpu__time_duration.sum'] / 1e6
    R = sorted(glob.glob('notes/p10f/raw/rtnn_%s_*.txt' % n)); sc = med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in R]); ev = med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in R])
    print(' r %s: scored %.4f event %.4f kernel %.4f | host outside event %.4f, gap inside event %.4f | scored/pred %.2f event/pred %.2f kernel/pred %.2f kernel share %.2f' % (n, sc, ev, k, sc - ev, ev - k, sc / pred, ev / pred, k / pred, k / sc))
ev = {}
for v in ('orig', 'reg'):
    R = sorted(glob.glob('notes/p10f/raw/knn_%s_*.txt' % v)); L = [[float(x) for x in re.findall(r'LAUNCH round \d+ active \d+ ms (\S+)', open(f).read())][:4] for f in R]; ev[v] = [med([l[i] for l in L]) for i in range(4)]
res = {}
for rep in (0, 1):
    for v in ('orig', 'reg'):
        res[v, rep] = load(files('knn_' + v)[rep])[:4]
for rep in (0, 1):
    print('\nrepetition', rep + 1, '(orig file %s, reg file %s)' % (files('knn_orig')[rep][-19:-4], files('knn_reg')[rep][-19:-4]))
    print('| round | orig ms | reg ms | speedup | L2 MB orig / reg | byte ratio | speedup / byte ratio | L2 TB/s orig / reg | lts pct orig / reg | sector hit orig / reg | DRAM MB orig / reg | dram pct orig / reg | warp inst M orig / reg | inst ratio | longsb orig / reg | event ms orig / reg | ncu / event orig / reg |')
    T = [0, 0]
    for r in range(4):
        o, g = res['orig', rep][r], res['reg', rep][r]; t = lambda x: x['gpu__time_duration.sum'] / 1e6; T[0] += t(o); T[1] += t(g)
        ls = lambda x: x['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']; allst = lambda x: sum(v for k, v in x.items() if 'stalled' in k)
        print('| %d | %.4f | %.4f | %.2f | %.0f / %.0f | %.2f | %.2f | %.2f / %.2f | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.2f | %.0f%% / %.0f%% | %.3f / %.3f | %.2f / %.2f |' % (r + 1, t(o), t(g), t(o) / t(g), o['lts__t_bytes.sum'] / 1e6, g['lts__t_bytes.sum'] / 1e6, o['lts__t_bytes.sum'] / g['lts__t_bytes.sum'], t(o) / t(g) / (o['lts__t_bytes.sum'] / g['lts__t_bytes.sum']),
              o['lts__t_bytes.sum'] / o['gpu__time_duration.sum'] / 1e3, g['lts__t_bytes.sum'] / g['gpu__time_duration.sum'] / 1e3, o['lts__throughput.avg.pct_of_peak_sustained_elapsed'], g['lts__throughput.avg.pct_of_peak_sustained_elapsed'],
              o['lts__t_sector_hit_rate.pct'], g['lts__t_sector_hit_rate.pct'], o['dram__bytes.sum'] / 1e6, g['dram__bytes.sum'] / 1e6, o['dram__throughput.avg.pct_of_peak_sustained_elapsed'], g['dram__throughput.avg.pct_of_peak_sustained_elapsed'],
              o['smsp__inst_executed.sum'] / 1e6, g['smsp__inst_executed.sum'] / 1e6, o['smsp__inst_executed.sum'] / g['smsp__inst_executed.sum'], 100 * ls(o) / allst(o), 100 * ls(g) / allst(g), ev['orig'][r], ev['reg'][r], t(o) / ev['orig'][r], t(g) / ev['reg'][r]))
    print('rounds 1 to 4 kernel total: orig %.3f reg %.3f speedup %.2f ; event total orig %.3f reg %.3f speedup %.2f' % (T[0], T[1], T[0] / T[1], sum(ev['orig']), sum(ev['reg']), sum(ev['orig']) / sum(ev['reg'])))
print('\nSM clock (nvidia-smi 50 ms, median MHz while busy >1000):')
for f in sorted(glob.glob('notes/ncu4090/ncu_*_2*.clk')): 
    c = [int(m) for m in re.findall(r', (\d+) MHz', open(f).read())]; b = [x for x in c if x > 1000]; print(f[-40:], 'median busy', med(b) if b else None, 'n', len(b))
