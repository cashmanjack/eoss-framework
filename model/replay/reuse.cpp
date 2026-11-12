// Reuse distance histograms of the node fetch stream of one SM (or one TPC) from a CPU replay. No GPU.
//   reuse TABLE.bin RAYS.bin NRAYS ORDER.bin OUT.txt [WRES=13] [J=52] [WARM=13]
// TABLE, RAYS: as replay.cpp / wset.cpp (nearest first depth first replay over the decoded OptiX BVH, every ray from the root).
// ORDER.bin: int32 ray index of each launch position (launch order = grouped order; warp w holds positions 32w..32w+31).
// Residency model (as depth_wset.py): warp w goes to SM w mod 142, SM s processes warps P + s + 142 j, j = 0..J-1, WRES resident at a time, windows P at 1/3 and 2/3 of the launch.
// Stream: resident warps are served round robin, one traversal step per turn: at step t a warp issues the t-th fetch of each of its lanes whose ray has one (lane order). A warp leaves
// after its longest lane's last fetch and the next warp of the SM takes its slot at the next step. A TPC stream merges SM 2t and 2t+1 (2 x WRES slots).
// Fetch ids: internal node (128 B) = node index; leaf slot entered (one line each, stand in for the triangle block) = nnodes + slot. Variants: nodes only / nodes and leaf slots, and dedupe
// (same id requested by several lanes of one warp in the same step counts once) / raw.
// Measured accesses: those of warps with j >= WARM (the first WRES warps only fill the cache). Distance = distinct ids touched since the previous access of the same id (LRU stack distance), cold = first touch.
// OUT lines: "unit variant dedupe total cold overflow h0 h1 ... h(MAXD-1)" (overflow = distance >= MAXD).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstring>
struct Slot { int32_t type, child; float b[6]; };
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct E { float t; int32_t node; int32_t leaf; };
static const int MAXD = 8192;
static bool slab(const float *b, const double *o, const double *id, double tmin, double tmax, double &t0)
{
    double lo = tmin, hi = tmax;
    for (int a = 0; a < 3; ++a) { double u = (b[a] - o[a]) * id[a], v = (b[3 + a] - o[a]) * id[a]; if (u > v) std::swap(u, v); lo = std::max(lo, u); hi = std::min(hi, v); }
    t0 = lo; return lo <= hi;
}
static int NN, NT, NL;
static std::vector<Slot> sl; static std::vector<int32_t> ls, lt; static std::vector<float> tr;
// fetch sequence of one ray: ids (node index, or NN + leaf slot)
static void replay(const Ray8 &r, std::vector<int32_t> &seq)
{
    seq.clear(); double o[3], d[3], id[3];
    for (int a = 0; a < 3; ++a) { o[a] = r.o[a]; d[a] = r.d[a]; id[a] = d[a] != 0 ? 1.0 / d[a] : 1e300; }
    double best = r.tmax, t0; std::vector<E> st; st.push_back({0.f, 0, -1});
    while (!st.empty()) {
        E e = st.back(); st.pop_back();
        if (e.t > best) continue;
        if (e.leaf >= 0) {
            seq.push_back(NN + e.leaf);
            for (int k = ls[e.leaf]; k < ls[e.leaf + 1]; ++k) {
                const int p = lt[k]; const float *v = &tr[(size_t)p * 9]; double A[3], B[3], C[3];
                for (int a = 0; a < 3; ++a) { A[a] = v[a]; B[a] = v[3 + a]; C[a] = v[6 + a]; }
                double e1[3], e2[3], pv[3], s[3], qv[3];
                for (int a = 0; a < 3; ++a) { e1[a] = B[a] - A[a]; e2[a] = C[a] - A[a]; s[a] = o[a] - A[a]; }
                pv[0] = d[1] * e2[2] - d[2] * e2[1]; pv[1] = d[2] * e2[0] - d[0] * e2[2]; pv[2] = d[0] * e2[1] - d[1] * e2[0];
                const double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
                if (std::fabs(det) < 1e-12) continue;
                const double inv = 1.0 / det, u = (s[0] * pv[0] + s[1] * pv[1] + s[2] * pv[2]) * inv;
                if (u < 0 || u > 1) continue;
                qv[0] = s[1] * e1[2] - s[2] * e1[1]; qv[1] = s[2] * e1[0] - s[0] * e1[2]; qv[2] = s[0] * e1[1] - s[1] * e1[0];
                const double vv = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
                if (vv < 0 || u + vv > 1) continue;
                const double t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
                if (t < r.tmin || t >= best) continue;
                best = t;
            }
            continue;
        }
        seq.push_back(e.node);
        E kids[12]; int nk = 0;
        for (int s = 0; s < 12; ++s) {
            const Slot &S = sl[(size_t)e.node * 12 + s]; if (!S.type) continue;
            if (!slab(S.b, o, id, r.tmin, best, t0)) continue;
            kids[nk++] = {(float)t0, S.type == 1 ? S.child : -1, S.type == 2 ? S.child : -1};
        }
        std::sort(kids, kids + nk, [](const E &a, const E &b) { return a.t > b.t; });
        for (int k = 0; k < nk; ++k) st.push_back(kids[k]);
    }
}
struct Acc { int32_t id; uint8_t meas; };
struct Hist { long total = 0, cold = 0, over = 0; std::vector<long> h = std::vector<long>(MAXD, 0); };
// LRU stack distances of a stream (BIT over positions)
static void stackdist(const std::vector<Acc> &s, std::vector<int32_t> &last, Hist &H)
{
    const int n = s.size(); std::vector<int32_t> bit(n + 2, 0), touched;
    auto add = [&](int i, int v) { for (++i; i <= n + 1; i += i & -i) bit[i] += v; };
    auto sum = [&](int i) { int r = 0; for (++i; i > 0; i -= i & -i) r += bit[i]; return r; };   // sum of positions 0..i
    for (int i = 0; i < n; ++i) {
        const int id = s[i].id; const int p = last[id];
        if (s[i].meas) {
            ++H.total;
            if (p < 0) ++H.cold; else { const int dist = sum(i - 1) - sum(p); if (dist >= MAXD) ++H.over; else ++H.h[dist]; }
        }
        if (p >= 0) add(p, -1); else touched.push_back(id);
        add(i, 1); last[id] = i;
    }
    for (int id : touched) last[id] = -1;
}
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb"); int32_t hdr[3]; fread(hdr, 4, 3, f); NN = hdr[0]; NT = hdr[1]; NL = hdr[2];
    sl.resize((size_t)NN * 12); fread(sl.data(), sizeof(Slot), sl.size(), f);
    ls.resize(NL + 1); fread(ls.data(), 4, NL + 1, f); lt.resize(ls[NL]); fread(lt.data(), 4, lt.size(), f); tr.resize((size_t)NT * 9); fread(tr.data(), 4, tr.size(), f); fclose(f);
    const long nr = atol(argv[3]); std::vector<Ray8> rays(nr); f = fopen(argv[2], "rb"); fread(rays.data(), sizeof(Ray8), nr, f); fclose(f);
    std::vector<int32_t> order(nr); f = fopen(argv[4], "rb"); fread(order.data(), 4, nr, f); fclose(f);
    const int WRES = argc > 6 ? atoi(argv[6]) : 13, J = argc > 7 ? atoi(argv[7]) : 52, WARM = argc > 8 ? atoi(argv[8]) : 13, NSM = 142;
    const long nw = nr / 32; const long Ps[2] = {nw / 3, 2 * nw / 3};
    // histograms: [unit 0 sm, 1 tpc][variant 0 nodes, 1 nodes+leaves][dedupe 0 raw, 1 dedupe]
    Hist H[2][2][2]; std::vector<int32_t> last(NN + NL + 1, -1);
    std::vector<int32_t> seq;
    for (int wi = 0; wi < 2; ++wi) {
        const long P = Ps[wi];
        // per SM: per warp (j) per lane: fetch sequences
        std::vector<std::vector<std::vector<std::vector<int32_t>>>> seqs(NSM);   // [sm][j][lane] -> ids
        for (int s = 0; s < NSM; ++s) {
            seqs[s].resize(J);
            for (int j = 0; j < J; ++j) {
                const long w = P + s + (long)NSM * j; seqs[s][j].resize(32);
                if ((w + 1) * 32 > nr) continue;
                for (int l = 0; l < 32; ++l) { replay(rays[order[w * 32 + l]], seq); seqs[s][j][l] = seq; }
            }
        }
        // stream builder for a set of SMs (1 for SM unit, 2 for TPC)
        auto build = [&](const std::vector<int> &sms, int variant, int dedupe, std::vector<Acc> &out) {
            out.clear(); const int ns = sms.size();
            struct Slotx { int sm, j, step, len; };
            std::vector<Slotx> sl_; std::vector<int> nextj(ns, 0);
            auto maxlen = [&](int sm, int j) { int m = 0; for (auto &q : seqs[sm][j]) m = std::max(m, (int)q.size()); return m; };
            for (int k = 0; k < ns; ++k) for (int r = 0; r < WRES; ++r) { sl_.push_back({sms[k], nextj[k], 0, 0}); sl_.back().len = maxlen(sms[k], nextj[k]); ++nextj[k]; }
            std::vector<int32_t> seen; bool any = true;
            while (any) {
                any = false;
                for (size_t q = 0; q < sl_.size(); ++q) {
                    Slotx &S = sl_[q]; const int k = q / WRES;
                    if (S.j >= J) continue;
                    any = true; seen.clear();
                    for (int l = 0; l < 32; ++l) {
                        const auto &sq = seqs[S.sm][S.j][l]; if (S.step >= (int)sq.size()) continue;
                        const int id = sq[S.step]; if (variant == 0 && id >= NN) continue;   // nodes only: leaf slot fetches are not part of the stream
                        if (dedupe) { bool dup = false; for (int x : seen) if (x == id) { dup = true; break; } if (dup) continue; seen.push_back(id); }
                        out.push_back({id, (uint8_t)(S.j >= WARM)});
                    }
                    if (++S.step >= S.len) { S.j = nextj[k]++; S.step = 0; if (S.j < J) S.len = maxlen(S.sm, S.j); }
                }
            }
        };
        std::vector<Acc> stream;
        for (int variant = 0; variant < 2; ++variant) for (int dedupe = 0; dedupe < 2; ++dedupe) {
            for (int s = 0; s < NSM; ++s) { build({s}, variant, dedupe, stream); stackdist(stream, last, H[0][variant][dedupe]); }
            for (int t = 0; t < NSM / 2; ++t) { build({2 * t, 2 * t + 1}, variant, dedupe, stream); stackdist(stream, last, H[1][variant][dedupe]); }
        }
    }
    f = fopen(argv[5], "w"); fprintf(f, "# nnodes %d nleafslots %d wres %d J %d warm %d maxd %d\n", NN, NL, WRES, J, WARM, MAXD);
    for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) for (int d = 0; d < 2; ++d) {
        const Hist &h = H[u][v][d]; fprintf(f, "%s %s %s %ld %ld %ld", u ? "tpc" : "sm", v ? "nodes+leaf" : "nodes", d ? "dedupe" : "raw", h.total, h.cold, h.over);
        for (int k = 0; k < MAXD; ++k) fprintf(f, " %ld", h.h[k]); fprintf(f, "\n");
    }
    fclose(f); return 0;
}
