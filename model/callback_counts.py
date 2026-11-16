"""Per program SM callback counts from ncu (queue 13). OptiX only, no TTU code, GPU 0.
    callback_counts.py profile OUTDIR LAUNCH_SKIP -- COMMAND...   run ncu on one optixLaunch (source counters, SASS), write OUTDIR/rep.csv (+ .ncu-rep)
    callback_counts.py table OUTDIR...                             per function table from the csv (functions are the SASS ranges that end in RET)
For every function of the OptiX module: calls (warp level, = executions of its first instruction), warp instructions per call, thread instructions per call,
global/local load and store instructions per call, L1 tag requests (global) per call, L2 theoretical sectors per call. Divergence is already in the warp call count.
Nothing here is fitted: these are counts. Cycle costs come from model/callback_cost.py (microbenchmarks)."""
import csv, json, os, subprocess, sys
NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')


def profile(out, skip, cmd):
    os.makedirs(out, exist_ok=True)
    rep = os.path.join(out, 'rep.ncu-rep')
    r = subprocess.run([NCU, '--target-processes', 'all', '--kernel-name', 'optixLaunch', '--launch-skip', str(skip), '--launch-count', '1', '--section', 'SourceCounters',
                        '--section', 'InstructionStats', '--section', 'MemoryWorkloadAnalysis', '--metrics', 'gpu__time_duration.sum,smsp__inst_executed.sum,smsp__thread_inst_executed.sum,'
                        'l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,'
                        'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sectors.sum,smsp__cycles_active.avg,sm__cycles_active.avg',
                        '--export', rep, '-f'] + cmd, capture_output=True, text=True)
    if r.returncode: sys.exit('ncu failed\n' + (r.stdout + r.stderr)[-2000:])
    for page, name in (('source', 'rep.csv'), ('raw', 'raw.csv')):
        a = [NCU, '--import', rep, '--page', page, '--csv'] + (['--print-source', 'sass'] if page == 'source' else [])
        open(os.path.join(out, name), 'w').write(subprocess.run(a, capture_output=True, text=True).stdout)


def functions(path):
    rows = list(csv.reader(open(path))); hi = next(i for i, r in enumerate(rows) if r and r[0] == 'Address'); h = rows[hi]; ix = {k: i for i, k in enumerate(h)}
    num = lambda r, k: float(r[ix[k]].replace(',', '') or 0)
    fs = []; cur = None
    for r in rows[hi + 1:]:
        if len(r) < len(h) or not r[0].startswith('0x'): continue
        if cur is None and (r[1].strip() == 'NOP' or r[1].strip().startswith('BRA 0x')): continue
        if cur is None: cur = dict(entry=num(r, 'Instructions Executed'), addr=r[0], ins=0, thr=0, gld=0, gst=0, lld=0, lst=0, tags=0, l2g=0, l2l=0, n=0, first=r[1].strip()); fs.append(cur)
        e = num(r, 'Instructions Executed'); cur['ins'] += e; cur['thr'] += num(r, 'Thread Instructions Executed'); cur['n'] += 1
        sp, op = r[ix['Address Space']], r[ix['Access Operation']]
        if sp == 'Global': cur['gld' if op == 'Load' else 'gst'] += e
        elif sp == 'Local': cur['lld' if op == 'Load' else 'lst'] += e
        cur['tags'] += num(r, 'L1 Tag Requests Global'); cur['l2g'] += num(r, 'L2 Theoretical Sectors Global'); cur['l2l'] += num(r, 'L2 Theoretical Sectors Local')
        if r[ix['Source']].strip().startswith('RET'): cur = None
    out = []
    for f in fs:
        c = f['entry']
        if c <= 0: continue
        out.append(dict(addr=f['addr'][-6:], calls=c, n_sass=f['n'], ins=f['ins'] / c, thr=f['thr'] / c, gld=f['gld'] / c, gst=f['gst'] / c, lld=f['lld'] / c, lst=f['lst'] / c,
                        tags=f['tags'] / c, l2g=f['l2g'] / c, l2l=f['l2l'] / c, ins_total=f['ins']))
    return out


def raw(path):
    rows = list(csv.reader(open(path)))
    h = rows[0] if rows and rows[0][0] == 'ID' else None
    for i, r in enumerate(rows):
        if r and r[0] == 'ID': h = r; v = rows[i + 1]; return {k: x for k, x in zip(h, v)}
    return {}


if __name__ == '__main__':
    if sys.argv[1] == 'profile':
        k = sys.argv.index('--'); profile(sys.argv[2], int(sys.argv[3]), sys.argv[k + 1:])
    else:
        for d in sys.argv[2:]:
            print('==', d)
            for f in functions(os.path.join(d, 'rep.csv')):
                print('%s calls %9.0f sass %4d warp ins/call %7.1f thr/call %7.1f gld %5.2f gst %5.2f lld %5.2f lst %5.2f tags %6.2f l2g %6.2f l2l %6.2f total %.3g' % (
                    f['addr'], f['calls'], f['n_sass'], f['ins'], f['thr'], f['gld'], f['gst'], f['lld'], f['lst'], f['tags'], f['l2g'], f['l2l'], f['ins_total']))
