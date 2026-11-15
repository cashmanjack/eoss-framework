"""RAGRT case study (Phase C, 2026-11-15): why the RT core loses for RAGRT's stage 1 (CPU only, nothing is run on a GPU).

Workload (read from ~/RAGRT_david: README, src/optix_corr_device_3d.cu): one ray per (query token, 3D slice); the scene is E triangles (one per
codeword); optixTrace with tmax = 1/tau and an ANY HIT program that records the hit and calls optixIgnoreIntersection, so every codeword with d.c >= tau
is one program call on the SM. H = hits per ray = the number of codewords above the threshold.

Inputs, all from our own measurements (Finding numbers in MASTER_PLAN.md):
  C_Q    RT unit service per query on one SM, hot cache, one triangle ray: 7.4 cycles (F3: 0.134 rays per cycle steady state); 10.3 through the
         Throughput_9 harness at 32 warps (F90: 0.097 lanes per cycle). Random rays in a BVH too big for L1: about 100 cycles (F90: 0.010 to 0.013).
  L_Q    one query, unloaded latency: 1,640 / 1,910 / 2,315 cycles warm for 1 / 2 / 3 levels (F17, F20, F22); node 343, leaf 113, triangle 52 on top of 934 (F67).
  C_SM   one SM round trip for a program the RT core cannot run: about 850 to 900 cycles of latency per candidate (F49: custom primitives; the any hit
         on a triangle is the same kind of stop and resume but is NOT measured). Throughput side (SM issue slots of the program) not measured: bracketed.
  Q      queries per ray: closest hit rays 1.85 to 2.0 (moderate overlap) up to 3.8 mean (heavy overlap, max 9) (F81). All hit rays (any hit): H + 1.
  F90    about 160 lanes in service per SM at saturation; second SM of a TPC adds 1.38 to 1.53 on cheap queries.
Brute force on CUDA cores (engine ragrt_bf): per ray E codewords x (3 FMA for the dot product + compare + select) = 5 lane instructions each, 128 lanes per SM per cycle.
"""
import math

C_Q_HOT, C_Q_HARNESS, C_Q_RANDOM = 7.4, 10.3, 100.0
INSTR_PER_CODEWORD, LANES_PER_CYCLE = 5, 128
C_BF = INSTR_PER_CODEWORD / LANES_PER_CYCLE          # cycles per codeword per ray, one SM
SM_ISSUE_AH = (0.25, 7.5)                            # any hit issue cost per hit per ray: 30 instructions, full warp of hits (30/32/4 = 0.23) to one lane of 32 (30/4)


def rt_cost(h, c_q=C_Q_HOT, ah=0.0):
    """cycles per ray per SM, throughput bound (every query pays the RT unit, every hit also pays the program's issue slots)"""
    return (h + 1) * c_q + h * ah


def bf_cost(e):
    return e * C_BF


def breakeven_fraction(c_q=C_Q_HOT, ah=0.0):
    """fraction f = H / E of codewords that may pass the threshold for the RT core to still win (large E): E * C_BF = (f E + 1) c_q + f E ah"""
    return C_BF / (c_q + ah)


def report():
    print('C_BF = %.4f cycles per codeword per ray (5 instructions, 128 lanes per cycle)' % C_BF)
    print('break even hit fraction f = H / E (RT wins below it): hot %.2f%% (query only), with 0.25 / 7.5 issue cycles per hit %.2f%% / %.2f%%, harness 10.3: %.2f%%, random 100: %.3f%%'
          % (100 * breakeven_fraction(), 100 * breakeven_fraction(ah=SM_ISSUE_AH[0]), 100 * breakeven_fraction(ah=SM_ISSUE_AH[1]),
             100 * breakeven_fraction(C_Q_HARNESS), 100 * breakeven_fraction(C_Q_RANDOM)))
    print('\nRT / brute force cost ratio per ray, query service only (hot, 7.4 cycles) and with 7.5 issue cycles per hit (>1: RT loses)')
    print('%6s %6s | %9s %9s | %9s %9s' % ('E', 'H', 'RT hot', 'BF', 'RT/BF', 'RT+ah/BF'))
    for e in (64, 256, 1024, 4096, 16384):
        for h in (1, 4, 16, 64):
            if h > e: continue
            print('%6d %6d | %9.1f %9.1f | %9.2f %9.2f' % (e, h, rt_cost(h), bf_cost(e), rt_cost(h) / bf_cost(e), rt_cost(h, ah=SM_ISSUE_AH[1]) / bf_cost(e)))
    print('\nlatency of one ray (unloaded, no overlap with other rays): base 1,910 (F20, two levels) + 900 per hit (F49 round trip, any hit not measured)')
    for h in (1, 4, 16, 64):
        print('  H %3d: about %6d cycles = %.1f us at 2.4 GHz' % (h, 1910 + 900 * h, (1910 + 900 * h) / 2400.))
    print('\nlanes needed to hide one round trip at the hot query rate: 900 / 7.4 = %d lanes (the RT unit holds about 160, F90)' % round(900 / C_Q_HOT))
    print('E sweep prediction (selectivity fixed at 5%% of the codewords): RT/BF = (0.05 E + 1) 7.4 / (0.039 E) -> %.2f at E 256, %.2f at 16384 (limit %.2f)'
          % ((0.05 * 256 + 1) * 7.4 / bf_cost(256), (0.05 * 16384 + 1) * 7.4 / bf_cost(16384), 0.05 * 7.4 / C_BF))


if __name__ == '__main__':
    report()
