"""RETTU model v1: performance model of the RT core traversal unit (Ada, SM89).

Inputs are measured workload descriptors per ray (queries per ray, RT L1 sector
requests, L1 fills = L2 reads, DRAM bytes). Terms (all cycles per ray, one SM):
  saturated part   B = a_q*q + a_s*S1 + a_f*F + a_d*D       (unit / L1 / L2 / DRAM work, W independent)
  latency part     L = b_q*q + b_f*F + b_d*D, hidden by W warps: L / W
  contention       C = c_qW * q * W / 16  (unit queueing: more warps, slower per query)
  c(W) = B + L / W + C,   rays per cycle per SM = 1 / c(W)
Whole GPU: TPC scaling (sibling SM adds 4%), L2 sector ceiling, per GPC limit.
Fitted numbers are written to notes/model_v1.md by run_model.py, never hard coded here.
"""
import numpy as np
from scipy.optimize import nnls

FEATS = ["q", "S1", "F", "D"]


def design(rows, W_key="W"):
    """Columns: saturated [q,S1,F,D], latency [q,F,D]/W, contention q*W/16."""
    X = []
    for r in rows:
        w = r[W_key]
        X.append([r["q"], r["S1"], r["F"], r["D"] / 32.0,
                  r["q"] / w, r["F"] / w, r["D"] / 32.0 / w, r["q"] * w / 16.0])
    return np.array(X, float)


PARAMS = ["a_q", "a_S1", "a_F", "a_D32", "b_q", "b_F", "b_D32", "c_qW"]


def fit(rows):
    X = design(rows)
    y = np.array([1.0 / r["rpc"] for r in rows])
    # relative error weighting: divide each row by y
    coef, _ = nnls(X / y[:, None], np.ones(len(y)))
    return dict(zip(PARAMS, coef))


def predict_sm(p, r):
    c = (p["a_q"] * r["q"] + p["a_S1"] * r["S1"] + p["a_F"] * r["F"] + p["a_D32"] * r["D"] / 32.0
         + (p["b_q"] * r["q"] + p["b_F"] * r["F"] + p["b_D32"] * r["D"] / 32.0) / r["W"]
         + p["c_qW"] * r["q"] * r["W"] / 16.0)
    return 1.0 / c


# whole GPU terms (measured, not fitted here)
N_TPC = 71
SIBLING_GAIN = 0.04       # second SM of a TPC adds 4%
L2_CEIL = 38.2            # sectors per SM cycle, plain load test (Plateau a)
GPC_CEIL = 6.0           # sectors per SM cycle per GPC (Plateau b)
SM_GPC = lambda sm: (sm // 2) % 12


def predict_gpu(x1, F, G):
    """Aggregate rays per cycle for G blocks, one per SM on SMs 0,2,4,... then the odd SMs.
    x1 = one SM rays per cycle, F = L2 sectors per ray."""
    per_gpc = np.zeros(12)
    for b in range(G):
        if b < N_TPC:
            per_gpc[SM_GPC(2 * b)] += x1                    # first SM of a TPC
        else:
            per_gpc[SM_GPC(2 * (b - N_TPC) + 1)] += SIBLING_GAIN * x1  # sibling adds 4%
    per_gpc = np.minimum(per_gpc, GPC_CEIL / F)            # per GPC sector limit
    return min(per_gpc.sum(), L2_CEIL / F)                 # whole L2 sector limit


# L2 capacity term (2026-11-13): hit rate falls as the working set passes the L2 size.
# See l2_capacity.py: cyclic access h = 1 up to C(1-d), 0 from C(1+d); d = 1/6 from triton's chase interval.
# Use it to split a ray's L2 reads F into L2 hits and DRAM misses when the touched bytes exceed the L2.
from l2_capacity import hit_cyclic, hit_random  # noqa: E402,F401
