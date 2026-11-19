"""Wait sweep (Phase 10j, predictions model/predictions_phase10j.md): node_run.py with arms single, twice, and own9_wait at 10, 70, 300 us.
    wait_run.py OUTDIR PHASE REPS     PHASE: build | stage0 | chk | arms   (same safety rules as node_run.py; builds in /dev/shm/t9m, delete after)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import node_run as N
N.ARMS = {'single': N.ARMS['single'], 'twice': N.ARMS['twice']}
for us in (10, 70, 300): N.ARMS['w%d' % us] = ('sib', '0:1024:1:0:9', N.NSM, '0,0', {'SIB_DNS': str(us * 1000)})
N.BUILDS = {k: N.BUILDS[k] for k in ('single', 'twice', 'sib')}
N.CAP_ARMS = []
N.main(*sys.argv[1:])
