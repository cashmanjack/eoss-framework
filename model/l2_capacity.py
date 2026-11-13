"""RETTU model v1, L2 capacity term (CPU only).

Hit rate of the L2 for a working set of W bytes, L2 size C bytes:
  cyclic access (pointer chase, a loop over the same lines in the same order; worst case for LRU):
      h = 1 for W <= C(1-d), 0 for W >= C(1+d), linear between   (d = softness, 0 = LRU step)
  random access (each line equally likely, any order): h = min(1, C/W)
Latency per dependent load = h*L_L2 + (1-h)*L_DRAM.
Version 2 (2026-11-13, after the ramp points 40, 44, 48, 52, 56 MB on the 4070 Ti): the ramp is asymmetric and starts at the L2 size:
  h = 1 up to LO*C, 0 from HI*C, linear between, LO = 0.97, HI = 1.05 (4070 Ti: 46.5 to 50.5 MB).
Measured (4070 Ti, 3 reps each, notes/io/l2ramp): 40 MB 286.4, 44 MB 287.2 (h 0.995), 48 MB 423.8 to 424.0 (h 0.63), 52 MB 656.0 to 656.3 (h 0), 56 MB 656.1 to 656.5.
Constraints: h(44) about 1, h(48) = 0.63, h(52) = 0 and a straight ramp: start between 45.4 and 47.9 MB, end between 48.5 and 52 MB (0.95 to 1.0 C and 1.01 to 1.08 C);
the middle of that region is used. The curve between 44 and 48 MB is not known (no point at 46 MB), so a straight ramp is an assumption.
Version 1 (symmetric ramp C(1-d) to C(1+d), d = 1/6 from triton's bounds only) is kept as hit_cyclic_sym (it predicted 0.5 at 48 MB and 0.83 at 44 MB, 0.17 at 52).
Triton (RTX 6000 Ada, C = 96 MB): hits at 1..64 MB (292..298 cycles), DRAM at 512 MB (652): consistent with any ramp above 64 MB; a ramp scaled by C would sit at 93 to 101 MB there (not measured).
"""
LO, HI = 0.97, 1.05
D_SOFT = 1.0 / 6.0


def hit_cyclic(W, C, lo=LO, hi=HI):
    if W <= C * lo:
        return 1.0
    if W >= C * hi:
        return 0.0
    return (C * hi - W) / (C * (hi - lo))


def hit_cyclic_sym(W, C, d=D_SOFT):   # version 1
    if W <= C * (1 - d):
        return 1.0
    if W >= C * (1 + d):
        return 0.0
    return (C * (1 + d) - W) / (2 * d * C)


def hit_random(W, C):
    return min(1.0, C / W)


def latency(h, l2=285.0, dram=658.0):
    return h * l2 + (1 - h) * dram


if __name__ == "__main__":
    import json, sys
    # triton chase (MB, cycles): from notes/setup3/runs/chase and the 2026-11-09 log (64 MB: 298)
    tri = [(1, 291.9), (16, 293.2), (64, 298.0), (512, 652.2)]
    print("triton (C=96 MB, L2 292, DRAM 652): MB, measured, predicted")
    for mb, m in tri:
        print(mb, m, round(latency(hit_cyclic(mb, 96), 292, 652), 1))
    ti = [(24, 284.4), (32, 284.4), (40, 286.4), (44, 287.2), (48, 423.9), (52, 656.2), (56, 656.3), (64, 658.2), (96, 660.0), (128, 661.1)]
    print("4070 Ti (C=48 MB, L2 285, DRAM 658): MB, measured, v1 symmetric d=1/6, v2 asymmetric, step d=0")
    for mb, m in ti:
        print(mb, m, round(latency(hit_cyclic_sym(mb, 48)), 1), round(latency(hit_cyclic(mb, 48)), 1), round(latency(hit_cyclic_sym(mb, 48, 0.0)), 1))
