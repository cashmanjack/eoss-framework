// CLOVER correctness check, second version (2026-11-17): rules out the checker's own failure modes (ties, duplicate points, tolerance) and classifies every wrong result.
// Our code; includes CLOVER's linear-scans.cu (third party, not copied) through -DCLOVER_SRC="dir" with its main renamed.
// Build: nvcc -O3 -DNDEBUG -arch=sm_89 -std=c++20 -w -I<clover>/include -DCLOVER_SRC=\"<clover>/src\" clover_check2.cu -o clover_check2
// Usage: ./clover_check2 ALG N K FILE|uniform [dumpprefix]    (queries = every point; the dump holds the returned indices (int32) and distances (float32), N x K each, for the independent python check)
#include <set>
#include <cmath>
#include <map>
#define main clover_unused_main
#define STR2(x) #x
#define STR(x) STR2(x)
#include STR(CLOVER_SRC/linear-scans.cu)
#undef main
int main(int argc, char **argv) {
    int alg = atoi(argv[1]); std::size_t N = atol(argv[2]), K = atol(argv[3]); std::string src = argv[4];
    std::vector<float> d(N * 3);
    if (src == "uniform") { std::mt19937 g(7); std::uniform_real_distribution<float> u(0, 1); for (auto &x : d) x = u(g); }
    else { std::ifstream f(src); std::string l; std::size_t i = 0; while (i < N && std::getline(f, l)) { for (auto &c : l) if (c == ',') c = ' '; std::istringstream s(l); s >> d[3*i] >> d[3*i+1] >> d[3*i+2]; ++i; } if (i < N) { std::cerr << "file too short\n"; return 2; } }
    std::set<std::array<float, 3>> uniq; for (std::size_t i = 0; i < N; ++i) uniq.insert({d[3*i], d[3*i+1], d[3*i+2]});
    std::size_t dupPoints = N - uniq.size();
    std::vector<idx_t> Q(N); std::iota(Q.begin(), Q.end(), 0);
    auto res = dispatch_knn<float, 3>(d, Q, Algorithm(alg), K, N, N);
    const auto &knn = res.first; const auto &dist = res.second;
    if (argc > 5) { std::string p = argv[5]; FILE *fa = fopen((p + "_idx.bin").c_str(), "wbx"), *fb = fopen((p + "_dist.bin").c_str(), "wbx"); if (!fa || !fb) { std::cerr << "dump file exists or cannot be created\n"; return 3; } fwrite(knn.data(), sizeof(idx_t), N * K, fa); fwrite(dist.data(), sizeof(float), N * K, fb); fclose(fa); fclose(fb); }
    std::size_t wrongQ = 0, qDup = 0, qOob = 0, qFar = 0, qSelfMissing = 0, qTieBoundary = 0, qWrongAndTie = 0, qWrongNoTie = 0, listsOk = 0; double sumRecall = 0, worstExcess = 0, sumExcess = 0; std::size_t nExcess = 0;
    std::vector<double> bd(N);
    for (std::size_t q = 0; q < N; ++q) {
        for (std::size_t j = 0; j < N; ++j) { double a = (double)d[3*q]-d[3*j], b = (double)d[3*q+1]-d[3*j+1], c = (double)d[3*q+2]-d[3*j+2]; bd[j] = a*a + b*b + c*c; }
        std::vector<double> s(bd); std::sort(s.begin(), s.end());
        double kth = s[K-1], next = s[K];                                             // true K-th and (K+1)-th squared distance (self counts, as in CLOVER)
        bool tie = (next - kth) <= 1e-6 * kth + 1e-12;                                 // a tie at the boundary within about 8 float ulps: the K-th neighbour is not unique
        std::vector<char> inTop(N, 0); { std::vector<std::pair<double, std::size_t>> v(N); for (std::size_t j = 0; j < N; ++j) v[j] = {bd[j], j}; std::partial_sort(v.begin(), v.begin() + K, v.end()); for (std::size_t j = 0; j < K; ++j) inTop[v[j].second] = 1; }
        bool ok = true, dup = false, oob = false, far = false; std::set<idx_t> seen; std::size_t hit = 0;
        for (std::size_t j = 0; j < K; ++j) {
            idx_t id = knn[q*K + j];
            if (id < 0 || (std::size_t)id >= N) { oob = true; ok = false; continue; }
            if (!seen.insert(id).second) { dup = true; ok = false; continue; }
            if (inTop[id] || bd[id] <= kth * (1 + 1e-6)) ++hit;                         // in the true top K, or tied with the K-th within float precision
            if (bd[id] > kth * (1 + 1e-6) + 1e-12) { far = true; ok = false; double ex = (bd[id] - kth) / (kth + 1e-30); worstExcess = std::max(worstExcess, ex); sumExcess += ex; ++nExcess; }
        }
        sumRecall += (double)hit / K; if (dup) ++qDup; if (oob) ++qOob; if (far) ++qFar; if (!seen.count((idx_t)q)) ++qSelfMissing; if (tie) ++qTieBoundary;
        if (!ok) { ++wrongQ; if (tie) ++qWrongAndTie; else ++qWrongNoTie; }
    }
    std::cout << "alg " << alg << " N " << N << " K " << K << " " << src << ": duplicate points in the data " << dupPoints << "; queries with a tie at the K-th boundary " << qTieBoundary
              << "; WRONG queries " << wrongQ << " (with boundary tie " << qWrongAndTie << ", without " << qWrongNoTie << "); of the wrong: repeated index " << qDup << ", index out of range " << qOob << ", a returned point farther than the true K-th " << qFar
              << "; self missing " << qSelfMissing << "; mean recall " << sumRecall / N << "; worst relative excess of a returned squared distance over the K-th " << worstExcess << ", mean " << (nExcess ? sumExcess / nExcess : 0.0) << std::endl;
    return wrongQ ? 1 : 0;
}
