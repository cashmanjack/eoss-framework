// assign TABLE0.bin(nodes only, as make_table pre-file) OUT_LEAF.bin : for every triangle, the leaf slot with the smallest box that contains the triangle's box,
// found by depth first search through every node slot whose box contains it (sibling boxes overlap, so one path is not enough). -1 if none.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
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
