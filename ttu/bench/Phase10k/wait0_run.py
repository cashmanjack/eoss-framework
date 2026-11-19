"""Zero wait arm (Phase 10k, predictions model/predictions_phase10k.md): node_run.py with arms single, twice, w0 (role block present, SIB_DNS 0) and w70 (control, 70 us).
    wait0_run.py OUTDIR PHASE REPS     PHASE: build | stage0 | chk | arms   (same safety rules as node_run.py; builds in /dev/shm/t9m, delete after)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Node_4090'))
import node_run as N
N.ARMS = {'single': N.ARMS['single'], 'twice': N.ARMS['twice']}
for us in (0, 70): N.ARMS['w%d' % us] = ('sib', '0:1024:1:0:9', N.NSM, '0,0', {'SIB_DNS': str(us * 1000)})
N.BUILDS = {k: N.BUILDS[k] for k in ('single', 'twice', 'sib')}
N.CAP_ARMS = []
N.main(*sys.argv[1:])
