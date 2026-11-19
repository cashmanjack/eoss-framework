"""Scores part 1 of model/predictions_5090_batch7.md (commit 2362bd4): second ncu metric list on the 5090 kNN builds. python3 model/score_5090_batch7a.py"""
import csv, glob, io
def load(f):
    t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for r in rows: d.setdefault(int(r['ID']), {'n': r['Kernel Name'][:30]})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return [v for k, v in sorted(d.items()) if v['n'] == 'optixLaunch'][:4]
F = lambda p: sorted(glob.glob('notes/pc14_workloads/ncu_knn2_%s_2*.csv' % p))[-1]
O, G = load(F('orig')), load(F('reg'))
print('| round | write share orig / reg | sectors M orig / reg (atom+red orig / reg) | bytes/32 vs sectors orig / reg | sectors per us per pct orig / reg | reg over orig | L1 sector hit orig / reg | tex lookup hit rate orig / reg |')
for r in range(4):
    row = []
    for x in (O[r], G[r]):
        rd, wr, at, rdc = x['lts__t_sectors_op_read.sum'], x['lts__t_sectors_op_write.sum'], x['lts__t_sectors_op_atom.sum'], x['lts__t_sectors_op_red.sum']; tot = rd + wr + at + rdc
        hit, miss = x['lts__t_sectors_srcunit_tex_lookup_hit.sum'], x['lts__t_sectors_srcunit_tex_lookup_miss.sum']
        row.append(dict(ws=wr / tot, tot=tot / 1e6, ar=at + rdc, b=x['lts__t_bytes.sum'] / 32 / tot, rate=tot / (x['gpu__time_duration.sum'] / 1e3) / x['lts__throughput.avg.pct_of_peak_sustained_elapsed'], l1=x['l1tex__t_sector_hit_rate.pct'], hr=100 * hit / (hit + miss), ms=x['gpu__time_duration.sum'] / 1e6))
    o, g = row
    print('| %d | %.1f%% / %.1f%% | %.1f / %.1f (%d / %d) | %.3f / %.3f | %.0f / %.0f | %.2f | %.1f / %.1f | %.1f / %.1f |' % (r + 1, 100 * o['ws'], 100 * g['ws'], o['tot'], g['tot'], o['ar'], g['ar'], o['b'], g['b'], o['rate'], g['rate'], g['rate'] / o['rate'], o['l1'], g['l1'], o['hr'], g['hr']))
