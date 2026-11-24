"""Treelet_1 runner (predictions: model/predictions_treelet1.md). OptiX only, no TTU code. tethys GPU 1.
    run.py WORK sweep|main|ncu SET [WPS]
Needs /dev/shm/treelet1/{treelet1, treelet1_dev.ptx, data8m/<set>_rays.bin ...}. Raw files: WORK/<tag>_<TS>.json / .csv (the host opens with mode x).
Idle check before every run (retries for 30 s: the previous process may still be releasing the GPU)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(HERE, '..', 'Phase10f', 'tethys_gpucheck.sh')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID', LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.path.expanduser('~/opt/cuda-12.8/lib64') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
T = '/dev/shm/treelet1'; D = T + '/data8m'; BIN = T + '/treelet1'; PTX = T + '/treelet1_dev.ptx'; N = '8000000'
NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')
ARMS = ['nat_R', 'grp_R', 'pn_R', 'pg_R', 'pp_R', 'nat_T', 'grp_T', 'pn_T', 'pg_T', 'pp_T']


def idle():
    for _ in range(30):
        if subprocess.run(['bash', GC], capture_output=True).returncode == 0: return
        time.sleep(1)
    sys.exit('GPU busy')


def run(cmd, tmo):
    idle(); t = time.time(); p = subprocess.run(['timeout', '-k', '2', str(tmo)] + cmd, env=ENV, capture_output=True, text=True)
    print(os.path.basename(cmd[0]), 'rc', p.returncode, '%.1fs' % (time.time() - t), p.stdout.strip().splitlines()[-1:] or '', p.stderr[-300:], flush=True)
    if p.returncode: sys.exit('STOP')
    return p


if __name__ == '__main__':
    W, WHICH, SET = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(W, exist_ok=True); TS = time.strftime('%Y%m%dT%H%M%S')
    if WHICH == 'sweep':
        for w in (8, 16, 24, 32): run([BIN, 'run', PTX, D, N, '14', SET, str(w), '3', '%s/wpssweep_%s_w%d_%s.json' % (W, SET, w, TS), 'pg_R'], 400)
    elif WHICH == 'main':
        wps = sys.argv[4]; K = sys.argv[5] if len(sys.argv) > 5 else '14'
        run([BIN, 'run', PTX, D, N, K, SET, wps, '10', '%s/main_%s_k%s_w%s_%s.json' % (W, SET, K, wps, TS)] + ARMS, 900)
    elif WHICH == 'ncu':
        wps = sys.argv[4]; K = sys.argv[5] if len(sys.argv) > 5 else '14'; tmpd = os.environ.get('NCU_TMP', T + '/ncutmp'); os.makedirs(tmpd, exist_ok=True)
        GROUPS = {'A': 'l1tex__t_sectors.sum,l1tex__t_sectors_lookup_hit.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__t_sectors_pipe_lsu_lookup_hit.sum,gpu__time_duration.sum',
                  'B': 'l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,dram__bytes_read.sum,sm__cycles_elapsed.max'}
        for g, met in GROUPS.items():
            out = '%s/ncu_%s_k%s_w%s_%s_%s.csv' % (W, SET, K, wps, g, TS); assert not os.path.exists(out)
            env = dict(ENV, TMPDIR=tmpd); idle()
            cmd = ['timeout', '-k', '2', '1500', NCU, '--target-processes', 'all', '-k', 'regex:optixLaunch', '--cache-control', 'none', '--clock-control', 'none', '--metrics', met, '--csv', '--page', 'raw', '--log-file', out,
                   BIN, 'profile', PTX, D, N, K, SET, wps, '0', '%s/ncuprof_%s_k%s_w%s_%s_%s.json' % (W, SET, K, wps, g, TS)] + ARMS
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print('ncu group', g, 'rc', p.returncode, p.stdout.strip().splitlines()[-2:], p.stderr[-300:], flush=True)
            if p.returncode: sys.exit('STOP')
