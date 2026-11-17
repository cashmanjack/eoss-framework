// assign TABLE0.bin(nodes only, as make_table pre-file) OUT_LEAF.bin : for every triangle, the leaf slot with the smallest box that contains the triangle's box,
// found by depth first search through every node slot whose box contains it (sibling boxes overlap, so one path is not enough). -1 if none.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
// exact triangle / box overlap (separating axis test, double precision; Akenine-Moeller): used by mode 1 (argv[4] = 1) for triangles that no leaf slot contains
static bool tri_box(const float *v, const float *lo, const float *hi, double tol)
{
    double c[3], h[3], a[3][3];
    for (int i = 0; i < 3; ++i) { c[i] = 0.5 * (lo[i] + hi[i]); h[i] = 0.5 * (hi[i] - lo[i]) + tol; }
    for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) a[k][i] = (double)v[k * 3 + i] - c[i];
    for (int i = 0; i < 3; ++i) { double mn = std::min(a[0][i], std::min(a[1][i], a[2][i])), mx = std::max(a[0][i], std::max(a[1][i], a[2][i])); if (mn > h[i] || mx < -h[i]) return false; }
    double e[3][3]; for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) e[k][i] = a[(k + 1) % 3][i] - a[k][i];
    for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) {   // 9 edge x axis tests
        const int j = (i + 1) % 3, l = (i + 2) % 3;   // axis = unit_i x edge: (0, -e_l, e_j) style
        double ax[3] = {0, 0, 0}; ax[j] = e[k][l]; ax[l] = -e[k][j]; (void)ax;
        double p0 = a[0][j] * e[k][l] - a[0][l] * e[k][j], p1 = a[1][j] * e[k][l] - a[1][l] * e[k][j], p2 = a[2][j] * e[k][l] - a[2][l] * e[k][j];
        double r = h[j] * std::fabs(e[k][l]) + h[l] * std::fabs(e[k][j]);
        if (std::min(p0, std::min(p1, p2)) > r || std::max(p0, std::max(p1, p2)) < -r) return false;
    }
    double n[3] = {e[0][1] * e[1][2] - e[0][2] * e[1][1], e[0][2] * e[1][0] - e[0][0] * e[1][2], e[0][0] * e[1][1] - e[0][1] * e[1][0]};
    double d = n[0] * a[0][0] + n[1] * a[0][1] + n[2] * a[0][2], r = h[0] * std::fabs(n[0]) + h[1] * std::fabs(n[1]) + h[2] * std::fabs(n[2]);
    return std::fabs(d) <= r;
}
struct Slot { int32_t type, child; float b[6]; };
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb"); int32_t h[3]; fread(h, 4, 3, f);
    std::vector<Slot> sl((size_t)h[0] * 12); fread(sl.data(), sizeof(Slot), sl.size(), f);
    std::vector<float> tr((size_t)h[1] * 9); fread(tr.data(), 4, tr.size(), f); fclose(f);
    // pass 1: smallest containing leaf slot. pass 2 (triangles with none: split by the builder, each piece box lies inside the triangle's box): every leaf slot
    // left empty by pass 1 whose box lies inside the triangle's box. Output: pairs (triangle, leaf slot).
    const float tol = argc > 3 ? (float)atof(argv[3]) : 1e-6f; long multi = 0, nun = 0, nsplit = 0, n3 = 0; std::vector<int32_t> pairs; std::vector<char> used;
    int nleaf = 0; for (auto &S : sl) if (S.type == 2 && S.child + 1 > nleaf) nleaf = S.child + 1;
    used.assign(nleaf, 0); std::vector<int> un;
    auto bounds = [&](int p, float *lo, float *hi) { for (int a = 0; a < 3; ++a) { lo[a] = 1e30f; hi[a] = -1e30f; } for (int v = 0; v < 3; ++v) for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], tr[p * 9 + v * 3 + a]); hi[a] = std::max(hi[a], tr[p * 9 + v * 3 + a]); } };
    for (int p = 0; p < h[1]; ++p) {
        float lo[3], hi[3]; bounds(p, lo, hi);
        std::vector<int> st{0}; double bestv = 1e300; int bl = -1, nc = 0;
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            for (int s = 0; s < 12; ++s) {
                const Slot &S = sl[(size_t)n * 12 + s]; if (!S.type) continue;
                bool ok = true; for (int a = 0; a < 3; ++a) ok &= S.b[a] - tol <= lo[a] && hi[a] <= S.b[3 + a] + tol;
                if (!ok) continue;
                if (S.type == 1) st.push_back(S.child);
                else { ++nc; double v = 0; for (int a = 0; a < 3; ++a) v += S.b[3 + a] - S.b[a]; if (v < bestv) { bestv = v; bl = S.child; } }
            }
        }
        multi += nc > 1;
        if (bl >= 0) { pairs.push_back(p); pairs.push_back(bl); used[bl] = 1; } else un.push_back(p);
    }
    const int mode = argc > 4 ? atoi(argv[4]) : 0;   // mode 1: a triangle no leaf slot contains goes into EVERY leaf slot it overlaps (exact test): superset, the nearest hit is then always reachable (F189 limitation fix)
    if (mode == 1) {
        for (int p : un) {
            float lo[3], hi[3]; bounds(p, lo, hi); std::vector<int> st{0}; int got = 0;
            while (!st.empty()) {
                int n = st.back(); st.pop_back();
                for (int s = 0; s < 12; ++s) {
                    const Slot &S = sl[(size_t)n * 12 + s]; if (!S.type) continue;
                    bool ov = true; for (int a = 0; a < 3; ++a) ov &= S.b[a] <= hi[a] + tol && lo[a] <= S.b[3 + a] + tol;
                    if (!ov || !tri_box(&tr[(size_t)p * 9], S.b, S.b + 3, tol)) continue;
                    if (S.type == 1) st.push_back(S.child); else { pairs.push_back(p); pairs.push_back(S.child); ++got; }
                }
            }
            nun += 1; nsplit += got > 0; n3 += got == 0;
        }
        un.clear();
    }
    for (int p : un) {
        float lo[3], hi[3]; bounds(p, lo, hi); int got = 0;
        std::vector<int> st{0};
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            for (int s = 0; s < 12; ++s) {
                const Slot &S = sl[(size_t)n * 12 + s]; if (!S.type) continue;
                bool ov = true; for (int a = 0; a < 3; ++a) ov &= S.b[a] <= hi[a] + tol && lo[a] <= S.b[3 + a] + tol;
                if (!ov) continue;
                if (S.type == 1) st.push_back(S.child);
                else { bool in = true; for (int a = 0; a < 3; ++a) in &= lo[a] - tol <= S.b[a] && S.b[3 + a] <= hi[a] + tol;
                       if (in && !used[S.child]) { pairs.push_back(p); pairs.push_back(S.child); ++got; } }
            }
        }
        nun += 1; nsplit += got > 0;
        if (!got) {   // pass 3: the overlapping leaf slot whose box is closest to the triangle's box (sum of corner differences)
            std::vector<int> st3{0}; double bd = 1e300; int bs = -1;
            while (!st3.empty()) {
                int n = st3.back(); st3.pop_back();
                for (int s = 0; s < 12; ++s) {
                    const Slot &S = sl[(size_t)n * 12 + s]; if (!S.type) continue;
                    bool ov = true; for (int a = 0; a < 3; ++a) ov &= S.b[a] <= hi[a] + tol && lo[a] <= S.b[3 + a] + tol;
                    if (!ov) continue;
                    if (S.type == 1) st3.push_back(S.child);
                    else { double dd = 0; for (int a = 0; a < 3; ++a) dd += std::abs(S.b[a] - lo[a]) + std::abs(S.b[3 + a] - hi[a]); if (dd < bd) { bd = dd; bs = S.child; } }
                }
            }
            if (bs >= 0) { pairs.push_back(p); pairs.push_back(bs); ++n3; }
        }
    }
    f = fopen(argv[2], "wb"); fwrite(pairs.data(), 4, pairs.size(), f); fclose(f);
    fprintf(stderr, "triangles with more than one containing leaf slot: %ld; none containing: %ld, of which given pieces: %ld, closest leaf: %ld\n", multi, nun, nsplit, n3);
}
