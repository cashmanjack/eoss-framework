"""RETTU model v1, L2 capacity term (CPU only).

Hit rate of the L2 for a working set of W bytes, L2 size C bytes:
  cyclic access (pointer chase, a loop over the same lines in the same order; worst case for LRU):
      h = 1 for W <= C(1-d), 0 for W >= C(1+d), linear between   (d = softness, 0 = LRU step)
  random access (each line equally likely, any order): h = min(1, C/W)
Latency per dependent load = h*L_L2 + (1-h)*L_DRAM.
d is fitted on triton's chase data (RTX 6000 Ada, C = 96 MB): hits at 1..64 MB (292..298 cycles),
DRAM at 512 MB (652). That only fixes d in [0, 1/3] (64 <= 96(1-d)). The middle, d = 1/6, is used.
The 4070 Ti (C = 48 MB) is then a check, not a fit.
"""
D_SOFT = 1.0 / 6.0


def hit_cyclic(W, C, d=D_SOFT):
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
    ti = [(24, 284.4), (32, 284.4), (40, 286.3), (48, 424.0), (56, 658.1), (64, 658.2), (96, 660.0), (128, 661.1)]
    print("4070 Ti (C=48 MB, L2 285, DRAM 658): MB, measured, v1 before (L2 285 up to 64 MB), step d=0, d=1/6, random")
    for mb, m in ti:
        print(mb, m, 285, round(latency(hit_cyclic(mb, 48, 0.0)), 1), round(latency(hit_cyclic(mb, 48)), 1),
              round(latency(hit_random(mb, 48)), 1))
