"""Run a command while sampling GPU 0 clocks, power and temperature with nvidia-smi every 50 ms (pc14; plain CUDA or TTU runs).
    sampled.py LOGBASE -- command ...      writes LOGBASE_<timestamp>.csv (mode x, never overwritten) and prints median, min, max of the SM clock
over the samples taken while the command ran (the process table is not sampled). Exit code = the command's."""
import os, statistics, subprocess, sys, time
i = sys.argv.index('--'); base = sys.argv[1]; cmd = sys.argv[i + 1:]
out = '%s_%s.csv' % (base, time.strftime('%Y%m%dT%H%M%S')); f = open(out, 'x')
sm = subprocess.Popen(['nvidia-smi', '-i', '0', '--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active', '--format=csv,noheader,nounits', '-lms', '50'], stdout=f, stderr=subprocess.DEVNULL)
time.sleep(0.3); t0 = time.time()
r = subprocess.run(cmd); t1 = time.time(); time.sleep(0.2); sm.terminate(); sm.wait(); f.close()
rows = [l.strip().split(', ') for l in open(out) if l.strip()]
def ts(x): return time.mktime(time.strptime(x[:19], '%Y/%m/%d %H:%M:%S')) + float(x[19:] or 0)
vals = [int(r_[1]) for r_ in rows if t0 - 1 <= ts(r_[0]) <= t1 + 1 and r_[1].isdigit()]
if vals: print('SM clock MHz over %d samples during the run (%.1f s): median %d min %d max %d; log %s' % (len(vals), t1 - t0, statistics.median(vals), min(vals), max(vals), out))
else: print('no clock samples; log', out)
sys.exit(r.returncode)
