"""Read an ncu --csv --page raw log (one row per ttu_one launch) into dicts of floats."""
import csv, io

KEYS = ['l1tex__t_sectors.sum', 'l1tex__t_sectors_lookup_hit.sum', 'l1tex__t_sectors_pipe_lsu.sum', 'l1tex__t_sectors_pipe_lsu_lookup_hit.sum', 'l1tex__t_sectors_pipe_tex.sum',
        'l1tex__m_xbar2l1tex_read_sectors.sum', 'lts__t_sectors_srcunit_tex_op_read.sum', 'lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum',
        'dram__bytes_read.sum', 'sm__cycles_elapsed.max', 'profiler__replayer_passes', 'gpu__time_duration.sum',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'dram__throughput.avg.pct_of_peak_sustained_elapsed', 'lts__t_sectors.sum', 'dram__bytes_write.sum']


def rows(path):
    txt = open(path).read()
    i = txt.index('"ID"')
    r = list(csv.reader(io.StringIO(txt[i:])))
    head, data = r[0], [x for x in r[2:] if x and x[0].isdigit()]
    out = []
    for x in data:
        d = dict(zip(head, x))
        out.append({k: float(d[k].replace(',', '')) for k in KEYS if k in d})
    return out


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        for i, d in enumerate(rows(p)):
            print(p.split('/')[-1], i, {k.split('__')[1].replace('.sum', '').replace('.max', ''): int(v) for k, v in d.items()})
