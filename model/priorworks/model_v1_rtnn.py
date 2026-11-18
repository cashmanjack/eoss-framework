"""Model v1, RTNN part, as a function (2026-11-18). Frozen form: model/priorworks/FORM_combined_v1.md (committed eb22f98). Nothing is refitted here.
T(launch) = max(T_throughput, T_chain).
 T_throughput = (110 + (4.9 + cb - cbref) q_mean) N / (SM x CLK), q = 1.10 Ce + 1, Ce = K (C + 1)/(S + 1) if S > K else C  (v1 was written with K = 50; K is a parameter here, the only generalization)
   C = points in the query's Chebyshev ball of radius r (the AABB hits, self included), S = points in its Euclidean ball (self included): the replay counts, exact on the CPU with a kd tree.
   cb - cbref = the measured callback cost of the program on this input minus that of the uniform reference. In a prediction made BEFORE any GPU run on the new data it is not known: here cb = cbref (difference 0), stated wherever the result is used.
 T_chain = (largest q) x L / CLK, L = 575 cycles (RTNN).
Hardware: SM = 142, CLK = 2.5e9 (RTX 6000 Ada, triton). Returns ms for the search launch only (no query sort, no geometry build, no result copy: v1 does not model those)."""
import numpy as np
from scipy.spatial import cKDTree
SM, CLK, L_RT = 142, 2.5e9, 575.
def replay_counts(pts, r, tree=None, workers=-1):
    t = tree or cKDTree(pts)
    C = t.query_ball_point(pts, r, p=np.inf, return_length=True, workers=workers).astype(float)
    S = t.query_ball_point(pts, r, return_length=True, workers=workers).astype(float)
    return C, S
def predict(N, K, C, S, cb_minus_cbref=0.0):
    Ce = np.where(S > K, K * (C + 1) / (S + 1), C); q = 1.10 * Ce + 1
    thr = (110 + (4.9 + cb_minus_cbref) * q.mean()) * N / (SM * CLK) * 1e3; ch = q.max() * L_RT / CLK * 1e3
    return dict(thr_ms=float(thr), chain_ms=float(ch), ms=float(max(thr, ch)), mean_q=float(q.mean()), max_q=float(q.max()), mean_C=float(C.mean()), mean_S=float(S.mean()))
