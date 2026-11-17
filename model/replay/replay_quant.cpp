// Struct_1 variant (2026-11-17): the exact candidate test is the slab test of the ray segment against the primitive box (works for points and segments); counts of nodes, leaf slots, prims in entered slots (calls) and exact candidates (hits, column 7).
// CPU replay of a hit-only-return nearest-first traversal over the decoded OptiX BVH (notes/bvh_format.md). No GPU.
//   replay TABLE.bin RAYS.bin NRAYS ORDER OUT.bin
// TABLE: int32 nnodes, int32 ntris, int32 nleafslots; nodes[nnodes] = 12 slots x {int32 type(0 empty,1 node,2 leaf), int32 child, float box[6] lo xyz hi xyz};
//        leaf slot child = leaf slot index; leafstart[nleafslots+1] int32; leaftri[...] int32; tris[ntris] 9 float.
// RAYS: Ray8 (o[3], tmin, d[3], tmax). ORDER: 0 = depth first, children sorted by entry distance (far ones pushed); 1 = global priority queue by entry distance.
// argv[6] (optional): per query element counts of variant A: int32 [nr][8][3] (nodes, leaves, tris of each query, queries beyond 8 are merged into the last).
// OUT per ray (int32 x 8): nodes fetched, leaf slots entered, triangles tested, queries A (any pending entry resumes), queries B (only unculled pending resumes), hit prim, hit t (float bits), hit events.
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
struct E { float t; int32_t node; int32_t leaf; bool operator<(const E &o) const { return t > o.t; } };
static bool slab(const float *b, const double *o, const double *id, double tmin, double tmax, double &t0)
{
    double lo = tmin, hi = tmax;
    for (int a = 0; a < 3; ++a) {
        double u = (b[a] - o[a]) * id[a], v = (b[3 + a] - o[a]) * id[a];
        if (u > v) std::swap(u, v);
        lo = std::max(lo, u); hi = std::min(hi, v);
    }
    t0 = lo; return lo <= hi;
}
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb"); int32_t hdr[3]; fread(hdr, 4, 3, f);
    const int NN = hdr[0], NT = hdr[1], NL = hdr[2];
    std::vector<Slot> sl((size_t)NN * 12); fread(sl.data(), sizeof(Slot), sl.size(), f);
    std::vector<int32_t> ls(NL + 1); fread(ls.data(), 4, NL + 1, f);
    std::vector<int32_t> lt(ls[NL]); fread(lt.data(), 4, lt.size(), f);
    std::vector<float> tr((size_t)NT * 9); fread(tr.data(), 4, tr.size(), f); fclose(f);
    const long nr = atol(argv[3]); const int order = atoi(argv[4]);
    std::vector<Ray8> rays(nr); f = fopen(argv[2], "rb"); fread(rays.data(), sizeof(Ray8), nr, f); fclose(f);
    std::vector<int32_t> out(nr * 8); std::vector<int32_t> pq3; if (argc > 6) pq3.assign((size_t)nr * 24, 0);
    for (long i = 0; i < nr; ++i) {
        const Ray8 &r = rays[i]; double o[3], d[3], id[3];
        for (int a = 0; a < 3; ++a) { o[a] = r.o[a]; d[a] = r.d[a]; id[a] = d[a] != 0 ? 1.0 / d[a] : 1e300; }
        int cq = 0, c0 = 0, c1 = 0, c2 = 0; double best = r.tmax; int bestp = -1, nodes = 0, leaves = 0, tris = 0, hits = 0, qa = 1, qb = 1, leafc = 0;
        std::vector<E> st; std::priority_queue<E> pq;
        auto push = [&](const E &e) { if (order) pq.push(e); else st.push_back(e); };
        auto empty = [&]() { return order ? pq.empty() : st.empty(); };
        auto pop = [&]() { E e; if (order) { e = pq.top(); pq.pop(); } else { e = st.back(); st.pop_back(); } return e; };
        double t0; push({0.f, 0, -1});   // the root is always fetched
        while (!empty()) {
            E e = pop();
            if (e.t > best) continue;
            if (e.leaf >= 0) {   // leaf slot: test its triangles
                ++leaves; bool improved = false; const int t_before = tris;
                bool anyc = false;
                for (int k = ls[e.leaf]; k < ls[e.leaf + 1]; ++k) {   // custom primitives: the "triangle" holds the primitive's box (lo xyz, hi xyz, 3 unused floats): a candidate if the query point is inside it
                    const int p = lt[k]; ++tris; const float *v = &tr[(size_t)p * 9];
                    double tt0; if (slab(v, o, id, r.tmin, best, tt0)) { ++hits; anyc = true; }   // exact candidate: the ray segment meets the primitive's own box (a point query is a segment of length 1e-16)
                }
                if (anyc) ++leafc;
                continue;
            }
            if (0) {
                bool improved = false; const int t_before = tris; (void)t_before;
                if (improved) {   // hardware returns after a leaf that improved the hit: does the traversal have to be resumed?
                    ++hits; bool anyA = !empty(), anyB = false;
                    if (anyA) { if (order) { anyB = pq.top().t <= best; } else { for (auto &x : st) if (x.t <= best) { anyB = true; break; } } }
                    if (anyA) { if (!pq3.empty()) { int k = std::min(cq, 7); pq3[(size_t)i * 24 + k * 3] += nodes - c0; pq3[(size_t)i * 24 + k * 3 + 1] += leaves - c1; pq3[(size_t)i * 24 + k * 3 + 2] += tris - c2; } c0 = nodes; c1 = leaves; c2 = tris; ++cq; } if (anyA) ++qa; if (anyB) ++qb;
                }
                continue;
            }
            ++nodes;   // internal node fetch (128 bytes)
            E kids[12]; int nk = 0;
            for (int s = 0; s < 12; ++s) {
                const Slot &S = sl[(size_t)e.node * 12 + s]; if (!S.type) continue;
                if (!slab(S.b, o, id, r.tmin, best, t0)) continue;
                kids[nk++] = {(float)t0, S.type == 1 ? S.child : -1, S.type == 2 ? S.child : -1};
            }
            if (order) { for (int k = 0; k < nk; ++k) push(kids[k]); }
            else { std::sort(kids, kids + nk, [](const E &a, const E &b) { return a.t > b.t; }); for (int k = 0; k < nk; ++k) push(kids[k]); }   // far first, so the nearest is popped first
        }
        if (!pq3.empty()) { int k = std::min(cq, 7); pq3[(size_t)i * 24 + k * 3] += nodes - c0; pq3[(size_t)i * 24 + k * 3 + 1] += leaves - c1; pq3[(size_t)i * 24 + k * 3 + 2] += tris - c2; }
        int32_t *q = &out[i * 8]; q[0] = nodes; q[1] = leaves; q[2] = tris; q[3] = hits + 1; q[4] = leafc + 1; q[5] = bestp;
        float bt = (float)best; memcpy(&q[6], &bt, 4); q[7] = hits;
    }
    f = fopen(argv[5], "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
    if (!pq3.empty()) { f = fopen(argv[6], "wb"); fwrite(pq3.data(), 4, pq3.size(), f); fclose(f); }
    return 0;
}
