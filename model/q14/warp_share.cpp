// CPU, queue 14: request merging across the lanes of a warp for RT-BarnesHut, from the replay of the real ray sequence (first hit termination).
//   warp_share TABLE.bin RAYS.bin CNT.bin NBODIES [W0 W1]    (warps W0..W1-1 only; default all)
// For every lane (body) the sequence of traversals (CNT) and for every traversal the ordered list of fetched ids: step 0 = instance node (id -1, shared by all rays),
// then internal nodes (id = node index) and entered leaf slots (id = NN + slot). Variants (counts per raw lane traversal):
//   raw: every fetch of every lane. lock: ids identical across lanes at the same (iteration, step) count once. iter: identical ids among the lanes of the same iteration (any step) count once.
//   warp: identical ids among all fetches of the warp (any iteration, any step; upper bound of what merging in one request could ever remove; an L1 hit would still be a request).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <cstring>
struct Slot { int32_t type, child; float b[6]; };
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct E { float t; int32_t node; int32_t leaf; };
static bool slab(const float *b, const double *o, const double *id, double tmin, double tmax, double &t0)
{ double lo = tmin, hi = tmax; for (int a = 0; a < 3; ++a) { double u = (b[a] - o[a]) * id[a], v = (b[3 + a] - o[a]) * id[a]; if (u > v) std::swap(u, v); lo = std::max(lo, u); hi = std::min(hi, v); } t0 = lo; return lo <= hi; }
int NN, NT, NL; std::vector<Slot> sl; std::vector<int32_t> ls, lt; std::vector<float> tr;
static void fetches(const Ray8 &r, std::vector<int> &ids, int &leaves)
{
    double o[3], d[3], id[3]; for (int a = 0; a < 3; ++a) { o[a] = r.o[a]; d[a] = r.d[a]; id[a] = d[a] != 0 ? 1.0 / d[a] : 1e300; }
    double best = r.tmax, t0; ids.clear(); ids.push_back(-1); leaves = 0; std::vector<E> st; st.push_back({0.f, 0, -1});
    while (!st.empty()) {
        E e = st.back(); st.pop_back(); if (e.t > best) continue;
        if (e.leaf >= 0) {
            ids.push_back(NN + e.leaf); ++leaves; bool improved = false;
            for (int k = ls[e.leaf]; k < ls[e.leaf + 1]; ++k) {
                const int p = lt[k]; const float *v = &tr[(size_t)p * 9]; double A[3], B[3], C[3], e1[3], e2[3], pv[3], s[3], qv[3];
                for (int a = 0; a < 3; ++a) { A[a] = v[a]; B[a] = v[3 + a]; C[a] = v[6 + a]; e1[a] = B[a] - A[a]; e2[a] = C[a] - A[a]; s[a] = o[a] - A[a]; }
                pv[0] = d[1] * e2[2] - d[2] * e2[1]; pv[1] = d[2] * e2[0] - d[0] * e2[2]; pv[2] = d[0] * e2[1] - d[1] * e2[0];
                const double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2]; if (std::fabs(det) < 1e-12) continue;
                const double inv = 1.0 / det, u = (s[0] * pv[0] + s[1] * pv[1] + s[2] * pv[2]) * inv; if (u < 0 || u > 1) continue;
                qv[0] = s[1] * e1[2] - s[2] * e1[1]; qv[1] = s[2] * e1[0] - s[0] * e1[2]; qv[2] = s[0] * e1[1] - s[1] * e1[0];
                const double vv = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv; if (vv < 0 || u + vv > 1) continue;
                const double t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv; if (t < r.tmin || t >= best) continue; best = t; improved = true;
            }
            if (improved) return; continue;   // first hit: terminate
        }
        ids.push_back(e.node); E kids[12]; int nk = 0;
        for (int s = 0; s < 12; ++s) { const Slot &S = sl[(size_t)e.node * 12 + s]; if (!S.type) continue; if (!slab(S.b, o, id, r.tmin, best, t0)) continue; kids[nk++] = {(float)t0, S.type == 1 ? S.child : -1, S.type == 2 ? S.child : -1}; }
        std::sort(kids, kids + nk, [](const E &a, const E &b) { return a.t > b.t; }); for (int k = 0; k < nk; ++k) st.push_back(kids[k]);
    }
}
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb"); int32_t hdr[3]; fread(hdr, 4, 3, f); NN = hdr[0]; NT = hdr[1]; NL = hdr[2];
    sl.resize((size_t)NN * 12); fread(sl.data(), sizeof(Slot), sl.size(), f); ls.resize(NL + 1); fread(ls.data(), 4, NL + 1, f); lt.resize(ls[NL]); fread(lt.data(), 4, lt.size(), f); tr.resize((size_t)NT * 9); fread(tr.data(), 4, tr.size(), f); fclose(f);
    const int nb = atoi(argv[4]); std::vector<int32_t> cnt(nb); f = fopen(argv[3], "rb"); fread(cnt.data(), 4, nb, f); fclose(f);
    std::vector<long> off(nb + 1, 0); for (int i = 0; i < nb; ++i) off[i + 1] = off[i] + cnt[i];
    const int nw = nb / 32; int W0 = argc > 5 ? atoi(argv[5]) : 0, W1 = argc > 6 ? atoi(argv[6]) : nw;
    const long base = off[(long)W0 * 32]; FILE *fr = fopen(argv[2], "rb");   // RAYS holds the bodies from warp W0 on (BH_DUMP_W0 window)
    double raw_n = 0, raw_l = 0, lock_n = 0, lock_l = 0, iter_n = 0, iter_l = 0, warp_n = 0, warp_l = 0, trav = 0, raw_i = 0, lock_i = 0, iter_i = 0;
    // instance node counted separately (id -1): raw_i = one per traversal; lock/iter/warp = merged
    for (int w = W0; w < W1; ++w) {
        std::vector<std::vector<std::vector<int>>> L(32);   // lane -> traversal -> ids
        int maxit = 0;
        for (int l = 0; l < 32; ++l) { int b = w * 32 + l; fseek(fr, (off[b] - base) * 32L, SEEK_SET); std::vector<Ray8> rr(cnt[b]); fread(rr.data(), sizeof(Ray8), cnt[b], fr); L[l].resize(cnt[b]); for (int i = 0; i < cnt[b]; ++i) { int lv; fetches(rr[i], L[l][i], lv); } maxit = std::max(maxit, (int)cnt[b]); }
        std::set<int> wset_n, wset_l;
        for (int i = 0; i < maxit; ++i) {
            std::set<int> itn, itl; size_t maxs = 0; for (int l = 0; l < 32; ++l) if (i < (int)L[l].size()) maxs = std::max(maxs, L[l][i].size());
            for (size_t s = 0; s < maxs; ++s) { std::set<int> ln, ll;
                for (int l = 0; l < 32; ++l) { if (i >= (int)L[l].size() || s >= L[l][i].size()) continue; int id = L[l][i][s];
                    if (id == -1) { raw_i += 1; ln.insert(-1); itn.insert(-1); wset_n.insert(-1); continue; }
                    if (id >= NN) { raw_l += 1; ll.insert(id); itl.insert(id); wset_l.insert(id); } else { raw_n += 1; ln.insert(id); itn.insert(id); wset_n.insert(id); } }
                lock_n += std::count_if(ln.begin(), ln.end(), [](int x) { return x >= 0; }); lock_i += ln.count(-1); lock_l += ll.size(); }
            iter_n += std::count_if(itn.begin(), itn.end(), [](int x) { return x >= 0; }); iter_i += itn.count(-1); iter_l += itl.size();
            for (int l = 0; l < 32; ++l) if (i < (int)L[l].size()) trav += 1;
        }
        // warp-wide unique ids over all iterations (counted once per warp)
        for (int x : wset_n) { if (x >= 0) warp_n += 1; } warp_l += wset_l.size();
    }
    printf("traversals %.0f\nper traversal: nodes raw %.3f lock %.3f iter %.3f warp-unique %.4f | instance raw %.3f lock %.3f iter %.3f | leaf slots raw %.3f lock %.3f iter %.3f warp-unique %.4f\n", trav,
           raw_n / trav, lock_n / trav, iter_n / trav, warp_n / trav, raw_i / trav, lock_i / trav, iter_i / trav, raw_l / trav, lock_l / trav, iter_l / trav, warp_l / trav);
    return 0;
}
