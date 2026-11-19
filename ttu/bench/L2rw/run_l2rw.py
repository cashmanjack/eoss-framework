#!/usr/bin/env python3
"""Runs the l2rw matrix (patterns 0..2, write quarters 0..4, footprints MB) and one ncu run per config. Raw jsonl and csv are timestamped and opened with mode x (never overwritten).
usage: run_l2rw.py BIN OUTDIR FOOT1,FOOT2,.. [NCU=1]   env CUDA_VISIBLE_DEVICES / TMPDIR / LD_LIBRARY_PATH set by the caller."""
import subprocess, sys, time, json, os, io, csv
B, O, F = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(',')]; ncu = len(sys.argv) < 5 or sys.argv[4] == '1'
ts = time.strftime('%Y%m%dT%H%M%S'); os.makedirs(O, exist_ok=True); fo = open('%s/l2rw_%s.jsonl' % (O, ts), 'x')
M = 'gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum'
def idle():
    return subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip() == ''
for rep in range(3):
  for mb in F:
    for pat in (0, 1, 2):
        for wq in (0, 1, 2, 3, 4):
            if not idle(): sys.exit('GPU busy, stop')
            cmd = [B, str(mb), str(pat), str(wq), '4000', '5']
            r = subprocess.run(['timeout', '-k', '2', '60'] + cmd, capture_output=True, text=True); rec = dict(rep=rep, line=r.stdout.strip(), rc=r.returncode)
            if ncu and rep == 0:
                n = subprocess.run(['timeout', '-k', '2', '120', 'ncu', '--csv', '--clock-control', 'none', '--cache-control', 'none', '--metrics', M, '--kernel-name', 'regex:k', '--launch-skip', '3', '--launch-count', '1'] + cmd, capture_output=True, text=True)
                t = n.stdout; rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))) if '"ID"' in t else []
                rec['ncu'] = {x['Metric Name']: x['Metric Value'] for x in rows}
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            if r.returncode: sys.exit('run failed %s' % cmd)
print('done', fo.name)
