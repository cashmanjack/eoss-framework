// CLOVER timing and correctness harness (2026-11-17). Our code; includes CLOVER's linear-scans.cu (third party, not copied) with its main renamed.
// Build: nvcc -O3 -DNDEBUG -arch=sm_89 -std=c++20 -w -I<clover>/include -DCLOVER_SRC=<clover>/src clover_bench.cu -o clover_bench
// Usage: ./clover_bench ALG FILE N Q K REPS CHECK [seed [DUMPFILE]]   DUMPFILE (new, 2026-11-17): writes the Q x K result indices (idx_t) of the last rep, never overwrites (full correctness check in python, check_clover_full.py)
//   ALG 0 bitonic, 1 warpwise, 2 hubs. FILE x,y,z csv (first N lines). Q queries: Q == N means every point, else a random subset (seed).
//   Prints CLOVER's own span for each rep (ns, device synchronized, hubs include the index build) and, for CHECK sampled queries, wrong count against a CPU double brute force (tie tolerance 1e-6 relative).
#include <set>
#include <cmath>
#include <random>
#define main clover_unused_main
#define STR2(x) #x
#define STR(x) STR2(x)
#include STR(CLOVER_SRC/linear-scans.cu)
#undef main
int main(int argc, char **argv) {
    if (argc < 8) { std::cerr << "usage\n"; return 2; }
    int alg = atoi(argv[1]); std::string src = argv[2]; std::size_t N = atol(argv[3]), Q = atol(argv[4]), K = atol(argv[5]); int reps = atoi(argv[6]); std::size_t CH = atol(argv[7]); unsigned seed = argc > 8 ? atoi(argv[8]) : 7;
    std::vector<float> d(N * 3);
    { std::ifstream f(src); std::string l; std::size_t i = 0; while (i < N && std::getline(f, l)) { for (auto &c : l) if (c == ',') c = ' '; std::istringstream s(l); s >> d[3*i] >> d[3*i+1] >> d[3*i+2]; ++i; } if (i < N) { std::cerr << "file too short\n"; return 2; } }
    std::vector<idx_t> Qv(Q);
    if (Q == N) std::iota(Qv.begin(), Qv.end(), 0);
    else { std::vector<idx_t> all(N); std::iota(all.begin(), all.end(), 0); std::mt19937 g(seed); std::shuffle(all.begin(), all.end(), g); std::copy(all.begin(), all.begin() + Q, Qv.begin()); }
    std::pair<thrust::host_vector<idx_t>, thrust::host_vector<float>> res;
    for (int r = 0; r < reps; ++r) { res = dispatch_knn<float, 3>(d, Qv, Algorithm(alg), K, N, Q); std::cout << "rep " << r << std::endl; }
    const auto &knn = res.first;
    std::size_t wrong = 0, far = 0, bad = 0; double recall = 0;
    std::mt19937 g2(1234); std::vector<double> bd(N);
    for (std::size_t c = 0; c < CH; ++c) {
        std::size_t qi = g2() % Q; idx_t q = Qv[qi];
        for (std::size_t j = 0; j < N; ++j) { double a = (double)d[3*q]-d[3*j], b = (double)d[3*q+1]-d[3*j+1], cc = (double)d[3*q+2]-d[3*j+2]; bd[j] = a*a + b*b + cc*cc; }
        std::vector<double> s(bd); std::nth_element(s.begin(), s.begin() + (K-1), s.end()); double kth = s[K-1];
        bool ok = true; std::set<idx_t> seen; std::size_t hit = 0;
        for (std::size_t j = 0; j < K; ++j) { idx_t id = knn[qi*K + j];
            if (id < 0 || (std::size_t)id >= N || !seen.insert(id).second) { ok = false; ++bad; continue; }
            if (bd[id] <= kth * (1 + 1e-6) + 1e-12) ++hit; else { ok = false; ++far; } }
        recall += (double)hit / K; if (!ok) ++wrong;
    }
    std::cout << "CHECK alg " << alg << " N " << N << " Q " << Q << " K " << K << " sampled " << CH << " wrong " << wrong << " far " << far << " invalid " << bad << " recall " << (CH ? recall / CH : 1) << std::endl;
    if (argc > 9) { FILE *fo = fopen(argv[9], "wbx"); if (fo) { fwrite(&knn[0], sizeof(idx_t), Q * K, fo); fclose(fo); } else std::cerr << "dump file exists or cannot open\n"; }
    return 0;
}
