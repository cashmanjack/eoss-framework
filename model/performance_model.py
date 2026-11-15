"""RETTU the performance model: performance model of the RT core traversal unit (Ada, SM89).

Inputs are measured workload descriptors per ray (queries per ray, RT L1 sector
requests, L1 fills = L2 reads, DRAM bytes). Terms (all cycles per ray, one SM):
  saturated part   B = a_q*q + a_s*S1 + a_f*F + a_d*D       (unit / L1 / L2 / DRAM work, W independent)
  latency part     L = b_q*q + b_f*F + b_d*D, hidden by W warps: L / W
  contention       C = c_qW * q * W / 16  (unit queueing: more warps, slower per query)
  c(W) = B + L / W + C,   rays per cycle per SM = 1 / c(W)
Whole GPU: TPC scaling (sibling SM adds 4%), L2 sector ceiling, per GPC limit.
Fitted numbers are written to notes/performance_model.md by run_model.py, never hard coded here.
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
# See l2_capacity.py: cyclic access h = 1 up to 0.97 C, 0 from 1.05 C (4070 Ti ramp points); the symmetric d = 1/6 version is hit_cyclic_sym.
# Use it to split a ray's L2 reads F into L2 hits and DRAM misses when the touched bytes exceed the L2.
from l2_capacity import hit_cyclic, hit_random  # noqa: E402,F401


# ---------------------------------------------------------------------------------------------------------------------------------
# Terms added 2026-11-15 (the performance model is one model; each term below was measured, then checked on data it was not set from)
# ---------------------------------------------------------------------------------------------------------------------------------

# (1) short query term (Finding 105): the traversal unit of a TPC serves both SMs. Per TPC rate = min(sum of what its SMs can issue, U).
#     U = unit capacity in rays per cycle per TPC: 4070 Ti 0.14 (short queries), RTX 6000 Ada 0.153 (from tri1, 142 SMs). For long queries one SM already
#     fills the unit (U = 1.04 x one SM), which is the 4 percent sibling gain of the original term.
def tpc_aggregate(x1, n_sms, n_tpc, U, long_query=False):
    """Aggregate rays per cycle with blocks on SMs 0, 2, 4, ... (first SM of each TPC) and then the odd SMs. x1 = rays per cycle of one SM running alone."""
    if long_query:
        U = (1 + SIBLING_GAIN) * x1
    first = min(n_sms, n_tpc); second = max(0, n_sms - n_tpc)
    both = min(second, n_tpc)
    return (first - both) * min(x1, U) + both * min(2 * x1, U)


# (2) payload and attribute term (Finding 112): registers per thread -> resident blocks -> warps, then a slowdown with an exponent per regime.
REG_FILE = 65536
BLOCK_THREADS = 128          # OptiX launches 128 thread blocks
ACTIVE_FRACTION = 13.6 / 16  # active warps per SM / resident warp slots at full occupancy (ncu, 4070 Ti)


def optix_registers(payload_values, attribute_values, regs0=101.0, per_payload=1.1, per_attr=1.7):
    """Registers per thread of an OptiX closest hit kernel (4070 Ti, driver 580): about 1 per payload value and 1.7 per attribute value."""
    return regs0 + per_payload * payload_values + per_attr * attribute_values


def resident_warps(regs):
    alloc = -(-int(round(regs)) // 8) * 8          # register allocation granularity 8
    blocks = REG_FILE // (alloc * BLOCK_THREADS)
    return 4 * min(blocks, 4) * ACTIVE_FRACTION


def payload_factor(regs_a, regs_b, k):
    """Predicted time(regs_b) / time(regs_a) = (warps_a / warps_b)^k. k = share of the kernel that scales with resident warps: about 0.1 for random rays on triangles (the RT core
    and memory dominate), 0.7 to 0.85 when the SM's own instruction issue dominates (coherent rays, custom primitive intersection programs)."""
    return (resident_warps(regs_a) / resident_warps(regs_b)) ** k


# (3) L2 capacity term: l2_capacity.hit_cyclic(W, C) for a cyclic walk, hit_random for random access; F (fill sectors per ray) splits into L2 hits and DRAM with it.
def split_fills(F, working_set, l2_bytes, cyclic=False):
    h = hit_cyclic(working_set, l2_bytes) if cyclic else hit_random(working_set, l2_bytes)
    return F * h, F * (1 - h)     # sectors served by L2, sectors that go to DRAM


# (4) inputs from the scene (no measured descriptors): queries per ray q from the CPU replay (Finding 81, model/replay), S1, F, D from the replay reuse histograms
#     (Finding 100, reuse_inputs.py). The replay reads the decoded GAS (BVH_5/gas.py) and the rays; nothing is measured on the GPU.
