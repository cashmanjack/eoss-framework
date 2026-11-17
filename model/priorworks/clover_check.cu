// CLOVER correctness check (2026-11-17): brute force on the CPU against CLOVER's own kNN results. Our code; includes CLOVER's linear-scans.cu (third party, not copied) with its main renamed.
// Build: nvcc -O3 -DNDEBUG -arch=sm_89 -std=c++20 -I<clover>/include -Dmain=clover_unused_main -include <clover>/src/linear-scans.cu ... see run line in the log; usage: ./clover_check ALG N K FILE|uniform  (queries = every point, as CLOVER's own runs)
#include <set>
#include <cmath>
#define main clover_unused_main
#include "/local/scratch/a/cashman3/baselines/clover-knn/src/linear-scans.cu"
#undef main
int main(int argc, char **argv) {
    int alg = atoi(argv[1]); std::size_t N = atol(argv[2]), K = atol(argv[3]); std::string src = argv[4];
    std::vector<float> d(N * 3);
    if (src == "uniform") { std::mt19937 g(7); std::uniform_real_distribution<float> u(0, 1); for (auto &x : d) x = u(g); }
    else { std::ifstream f(src); std::string l; std::size_t i = 0; while (i < N && std::getline(f, l)) { for (auto &c : l) if (c == ',') c = ' '; std::istringstream s(l); s >> d[3*i] >> d[3*i+1] >> d[3*i+2]; ++i; } if (i < N) { std::cerr << "file too short\n"; return 2; } }
    std::vector<idx_t> Q(N); std::iota(Q.begin(), Q.end(), 0);
    auto res = dispatch_knn<float, 3>(d, Q, Algorithm(alg), K, N, N);
    const auto &knn = res.first; const auto &dist = res.second;
    std::size_t badset = 0, badorder = 0, badself = 0, nsq = 0, nroot = 0, tie_ok = 0; double maxrel = 0;
    std::vector<double> bd(N);
    for (std::size_t q = 0; q < N; ++q) {
        for (std::size_t j = 0; j < N; ++j) { double a = d[3*q]-d[3*j], b = d[3*q+1]-d[3*j+1], c = d[3*q+2]-d[3*j+2]; bd[j] = a*a + b*b + c*c; }
        std::vector<double> s(bd); std::nth_element(s.begin(), s.begin() + K - 1, s.end()); double kth = *std::max_element(s.begin(), s.begin() + K);   // K-th smallest squared distance
        std::vector<double> sk(s.begin(), s.begin() + K); std::sort(sk.begin(), sk.end());
        // reported distances: squared or not?
        double r0 = dist[q*K + K-1]; if (std::fabs(r0 - kth) <= 1e-4 * kth + 1e-9) ++nsq; else if (std::fabs(r0 * r0 - kth) <= 1e-4 * kth + 1e-9) ++nroot;
        bool ok = true, ord = true; std::set<idx_t> seen;
        for (std::size_t j = 0; j < K; ++j) {
            idx_t id = knn[q*K + j]; if (id < 0 || (std::size_t)id >= N || !seen.insert(id).second) { ok = false; continue; }
            if (bd[id] > kth * (1 + 1e-5) + 1e-9) ok = false;                       // a neighbour farther than the true K-th (ties allowed)
            double rep = dist[q*K + j]; double rel = std::fabs(rep - sk[j]) / (sk[j] + 1e-9); if (rel > maxrel && sk[j] > 1e-9) maxrel = rel; if (std::fabs(rep - sk[j]) > 1e-4 * sk[j] + 1e-6 && std::fabs(rep*rep - sk[j]) > 1e-4 * sk[j] + 1e-6) ord = false;
        }
        if (!ok) ++badset; if (!ord) ++badorder; if (!seen.count((idx_t)q)) ++badself;
    }
    std::cout << "alg " << alg << " N " << N << " K " << K << " " << src << ": queries " << N << " wrong neighbour sets " << badset << " distance list mismatches " << badorder << " self missing " << badself << " (last dist squared in " << nsq << ", sqrt in " << nroot << ") max rel dist err " << maxrel << std::endl;
    return badset ? 1 : 0;
}
