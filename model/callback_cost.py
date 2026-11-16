"""SM callback term of the performance model (queue 13): measured, not fitted. CPU only (reads ncu csv files made by model/callback_counts.py).
FORM FIXED BEFORE ANY COMPARISON WITH LAUNCH TIMES (committed first).
Counts (ncu source page, per SASS function, warp level): calls, warp instructions per call, L1 tag requests (global) per call, local load/store instructions per call
(one tag each: a local access of 32 lanes is 128 contiguous bytes), L2 theoretical sectors per call, and the kernel level L1 hit rate of global loads h.
Costs (plain CUDA microbenchmark model/cbcost/cbcost.cu, 4070 Ti, one block of 32 warps on every SM, results in model/results_callback_cost.md):
  c_i   = 0.25  SM cycles per warp instruction (independent FFMA, 4 schedulers)
  c_tag = 1.00  SM cycles per L1 tag request (32 lanes on 32 different lines, L1 resident: 32.08 cycles per instruction)
  c_x   = 1.90  extra SM cycles per sector that misses the L1 and is served by the L2 (32 lanes on 32 random lines of 8 MB: 92.9 cycles per instruction = 2.90 per sector, minus the tag)
Callback functions = SASS functions called more than 1.5 times per warp of the launch (closest hit, miss, any hit, intersection, and the per traversal glue); raygen (once per warp) is not included.
Cost of one warp call: max(ins c_i, (tags + local) c_tag + (1 - h) sectors c_x)   (issue and the load store pipe overlap)   [primary]
                       ins c_i + (tags + local) c_tag + (1 - h) sectors c_x     (no overlap)                          [sensitivity]
SM cycles of the launch: sum over callback functions of calls x cost / SM count.  cycles per query per SM (the unit of the model) = that x SM count / queries.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import callback_counts as CC
C_I, C_TAG, C_X = 0.25, 1.00, 1.90
NSM = 60


def load(d):
    f = CC.functions(os.path.join(d, 'rep.csv')); r = CC.raw(os.path.join(d, 'raw.csv'))
    h = float(r.get('l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct', '0').replace(',', '')) / 100
    t = float(r['gpu__time_duration.sum'].replace(',', '')) if 'gpu__time_duration.sum' in r else float('nan')
    return f, h, t


def term(d, warps, form='max'):
    """returns dict: SM cycles per launch (callbacks), parts per function"""
    f, h, t = load(d); tot = 0.0; parts = []
    for x in f:
        if x['calls'] <= 1.5 * warps: continue
        iss = x['ins'] * C_I; mem = (x['tags'] + x['lld'] + x['lst']) * C_TAG + (1 - h) * x['l2g'] * C_X
        c = max(iss, mem) if form == 'max' else iss + mem
        tot += x['calls'] * c / NSM; parts.append((x['addr'], x['calls'], x['ins'], iss, mem, c))
    return dict(cycles=tot, h=h, ncu_ns=t, parts=parts)


if __name__ == '__main__':
    for d in sys.argv[2:]:
        r = term(d, float(sys.argv[1])); print(d, 'SM cycles %.4g (%.3f ms at 2.5 GHz)  L1 hit rate %.3f  ncu time %.3f ms' % (r['cycles'], r['cycles'] / 2.5e6, r['h'], r['ncu_ns'] / 1e6))
        for p in r['parts']: print('   %s calls %9.0f ins %6.1f issue %7.1f mem %7.1f -> %7.1f cycles/call' % p)
