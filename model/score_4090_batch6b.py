"""Scores the addendum of model/predictions_4090_batch6.md (commit 7f14040): second ncu run, notes/ncu4090/ncu2_* (private raw). python3 model/score_4090_batch6b.py"""
import csv, glob, io
def load(f):
    t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for r in rows: d.setdefault(int(r['ID']), {'n': r['Kernel Name'][:30]})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return [v for k, v in sorted(d.items()) if v['n'] == 'optixLaunch'][:4]
F = lambda p: sorted(glob.glob('notes/ncu4090/ncu2_knn_%s_2*.csv' % p))
for rep in (0, 1):
    O, G = load(F('orig')[rep]), load(F('reg')[rep]); print('repetition', rep + 1)
    print('| round | write share of sectors orig / reg | sectors M orig / reg (read, write, atom, red orig) | sectors per us per pct orig / reg | reg over orig | L1 sector hit orig / reg | lookup hit rate orig / reg |')
    for r in range(4):
        row = []
        for x in (O[r], G[r]):
            rd, wr = x['lts__t_sectors_op_read.sum'], x['lts__t_sectors_op_write.sum']; tot = rd + wr + x['lts__t_sectors_op_atom.sum'] + x['lts__t_sectors_op_red.sum']
            row.append((wr / tot, tot / 1e6, rd / 1e6, wr / 1e6, tot / (x['gpu__time_duration.sum'] / 1e3) / x['lts__throughput.avg.pct_of_peak_sustained_elapsed'], x['l1tex__t_sector_hit_rate.pct'], x['lts__t_sectors_srcunit_tex_lookup_hit.sum'] / (x['lts__t_sectors_srcunit_tex_lookup_hit.sum'] + x['lts__t_sectors_srcunit_tex_lookup_miss.sum']) * 100, x['lts__t_bytes.sum'] / 32 / 1e6, x['lts__t_sectors_op_atom.sum'] + x['lts__t_sectors_op_red.sum']))
        o, g = row
        print('| %d | %.1f%% / %.1f%% | %.2f / %.2f (rd %.2f wr %.2f; reg rd %.2f wr %.2f; atom+red %d / %d; bytes/32 %.2f / %.2f) | %.2f / %.2f | %.2f | %.1f / %.1f | %.1f / %.1f |' % (r + 1, 100 * o[0], 100 * g[0], o[1], g[1], o[2], o[3], g[2], g[3], o[8], g[8], o[7], g[7], o[4], g[4], g[4] / o[4], o[5], g[5], o[6], g[6]))
