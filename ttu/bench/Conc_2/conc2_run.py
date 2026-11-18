"""Conc_2 (Phase 10c): F229 mechanism and the ALU speedup, on top of Conc_1 (same checked lean TTU group, same checks, same stage 0 first).
    conc2_run.py WORK GROUP REPS      GROUP: smoke | cg | lvl | lvlsmoke | clk
cg : RT G = NSM/4 beside mem (plain load, L1 cached) and memcg (ld.global.cg, L1 bypass) on sib and high at 1, 2 and 4 compute blocks per SM, plus RT alone.
lvl: throttled levels (sleep per iteration) and memseq (coalesced stream), see model/predictions_phase10c.md 1b.
clk: alone, hi_alu, sib_alu with an NVML clock and power sampler (ttu/bench/Conc_2/clksamp, one row per ms) running for the whole group; launch stamps (CLOCK_MONOTONIC) are in the raw rows.
Raw files WORK/conc_<group>_<TS>.json (Conc_1 rows) and WORK/clk_<group>_<TS>.csv, never overwritten."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'Conc_1'))
import conc_run as C  # noqa: E402
Q, HALF = C.Q, C.HALF
def configs(group):
    c = [('alone%d' % Q, Q, None)]
    if group == 'clk': return c + [('hi_alu', Q, 'alu:high'), ('sib_alu', Q, 'alu:sib')]
    if group in ('lvl', 'lvlsmoke'):   # throttled levels at matched 128 blocks (sib 4 per SM, hi 2 per SM) and the coalesced stream
        for t in ('mem', 'memcg'):
            for pl, lim in (('sib', 4), ('high', 2)):
                for d in (150000, 800000): c.append(('%s_%s%d_d%d' % ('hi' if pl == 'high' else pl, t, lim, d // 1000), Q, '%s:%s:%d:%d' % (t, pl, lim, d)))
        for pl, lim in (('sib', 1), ('sib', 4), ('high', 1), ('high', 2)): c.append(('%s_memseq%d' % ('hi' if pl == 'high' else pl, lim), Q, 'memseq:%s:%d' % (pl, lim)))
        return [c[0], c[1], c[9]] if group == 'lvlsmoke' else c
    for t in ('mem', 'memcg'):
        for pl in ('sib', 'high'):
            for lim in (1, 2, 4): c.append(('%s_%s%d' % (pl[:2] if pl == 'high' else pl, t, lim), Q, '%s:%s:%d' % (t, pl, lim)))
    return [c[0], c[1], c[7]] if group == "smoke" else c   # smoke: alone, sib mem 1 block, sib memcg 1 block
C.configs = configs
if __name__ == '__main__':
    work, group, reps = sys.argv[1:4]; os.makedirs(work, exist_ok=True); sp = None
    gc = C.gpucfg.load()
    if group == 'clk':
        csv = os.path.join(work, 'clk_%s_%s.csv' % (group, time.strftime('%Y%m%dT%H%M%S')))
        sp = subprocess.Popen([os.path.join(HERE, 'clksamp'), str(gc.index), csv, '3000'])
        time.sleep(1); print('sampler pid', sp.pid, csv, flush=True)
    try: C.main(work, group, reps)
    finally:
        if sp: sp.terminate(); sp.wait()
