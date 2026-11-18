// Minimal repro (2026-11-18): CLOVER bitonic and hubs answer all N points whatever the query list says; with Q < N the result buffers (sized Q x K) are written beyond their end.
// Build: nvcc -O3 -DNDEBUG -arch=sm_89 -std=c++20 -w -I<clover>/include "-DCLOVER_SRC=<clover>/src" clover_qlt_n.cu $(find <clover>/include/treelogy -name "*.cu") -o clover_qlt_n      (clover = github.com/ampslab/clover-knn, commit 03cfaa2; includes its src/linear-scans.cu, main renamed)
// Run:   ./clover_qlt_n ALG MODE        ALG 0 bitonic, 1 warpwise, 2 hubs.   MODE sub: N = 2000 points, Q = 1000 queries (the first 1000 ids), K = 8
//                                       MODE all: Q = N, query list in order (works for every ALG)
//                                       MODE perm: Q = N, query list REVERSED; prints whether result row 0 is the answer for query id queries[0] (= N - 1) or for point 0
//        compute-sanitizer ./clover_qlt_n 0 sub     shows the invalid global writes of the bitonic kernel (hubs the same); warpwise is clean.
#include <random>
#define main clover_unused_main
#define STR2(x) #x
#define STR(x) STR2(x)
#include STR(CLOVER_SRC/linear-scans.cu)
#undef main
int main(int argc, char **argv) {
    if (argc < 3) { std::cerr << "usage: ALG MODE\n"; return 2; }
    int alg = atoi(argv[1]); std::string mode = argv[2]; const std::size_t N = 2000, K = 8; std::size_t Q = mode == "sub" ? 1000 : N;
    std::mt19937 g(1); std::uniform_real_distribution<float> u(0.f, 1.f); std::vector<float> d(N * 3); for (auto &x : d) x = u(g);
    std::vector<idx_t> qv(Q); for (std::size_t i = 0; i < Q; ++i) qv[i] = mode == "perm" ? (idx_t)(N - 1 - i) : (idx_t)i;
    auto res = dispatch_knn<float, 3>(d, qv, Algorithm(alg), K, N, Q);
    cudaError_t e = cudaDeviceSynchronize(); std::cout << "alg " << alg << " mode " << mode << " N " << N << " Q " << Q << " K " << K << " result rows " << res.first.size() / K << " cuda status: " << cudaGetErrorString(e) << std::endl;
    if (e == cudaSuccess && mode == "perm") {   // is row 0 the answer for point queries[0] = N-1 (its nearest is itself, distance 0) or for point 0?
        const idx_t q0 = qv[0]; bool selfq = false, self0 = false; for (std::size_t j = 0; j < K; ++j) { selfq |= res.first[j] == q0; self0 |= res.first[j] == 0; }
        std::cout << "row 0 contains query id " << q0 << " (queries[0]): " << selfq << ", contains point 0: " << self0 << std::endl; }
    return e == cudaSuccess ? 0 : 1;
}
