// CPU only: writes the triangles of scene soup:N exactly as ttu/bench/Morton/morton.cu builds them (float3 a,b,c per triangle).  soupgen N OUT.bin
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
struct F3 { float x, y, z; };
struct Tri { F3 a, b, c; };
int main(int argc, char **argv)
{
    const int n = atoi(argv[1]); const float e = 3.f / std::cbrt((float)n);
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; v.reserve(n);
    for (int i = 0; i < n; ++i) {
        const F3 c{U(g), U(g), U(g)};
        auto pt = [&]() { return F3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; };
        v.push_back({pt(), pt(), pt()});
    }
    FILE *f = fopen(argv[2], "wb"); fwrite(v.data(), sizeof(Tri), v.size(), f); fclose(f);
}
