// CPU brute force (double precision Moller-Trumbore, OpenMP): exact nearest hit distance of every ray over all triangles.   brute TRIS.bin RAYS.bin OUT_tfirst.f64
// RAYS: Ray8 (o[3], tmin, d[3], tmax) float32; TRIS: 9 float32 per triangle. Hit if det != 0, u, v >= 0, u + v <= 1, t >= 0 (tmin 0). Output per ray: t (double), inf if none; second array: primitive (int32).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); size_t nt = ftell(f) / 36; fseek(f, 0, SEEK_SET); std::vector<float> tf(nt * 9); if (fread(tf.data(), 4, nt * 9, f) != nt * 9) return 1; fclose(f);
    f = fopen(argv[2], "rb"); fseek(f, 0, SEEK_END); size_t nr = ftell(f) / 32; fseek(f, 0, SEEK_SET); std::vector<float> rf(nr * 8); if (fread(rf.data(), 4, nr * 8, f) != nr * 8) return 1; fclose(f);
    std::vector<double> A(nt * 9), E1(nt * 3), E2(nt * 3);
    for (size_t i = 0; i < nt; ++i) { for (int k = 0; k < 9; ++k) A[i * 9 + k] = tf[i * 9 + k]; for (int k = 0; k < 3; ++k) { E1[i * 3 + k] = A[i * 9 + 3 + k] - A[i * 9 + k]; E2[i * 3 + k] = A[i * 9 + 6 + k] - A[i * 9 + k]; } }
    std::vector<double> T(nr); std::vector<int32_t> P(nr);
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < (long)nr; ++r) {
        const double o[3] = {rf[r * 8], rf[r * 8 + 1], rf[r * 8 + 2]}, d[3] = {rf[r * 8 + 4], rf[r * 8 + 5], rf[r * 8 + 6]};
        double best = INFINITY; int bp = -1;
        for (size_t i = 0; i < nt; ++i) {
            const double *e1 = &E1[i * 3], *e2 = &E2[i * 3];
            const double p0 = d[1] * e2[2] - d[2] * e2[1], p1 = d[2] * e2[0] - d[0] * e2[2], p2 = d[0] * e2[1] - d[1] * e2[0];
            const double det = e1[0] * p0 + e1[1] * p1 + e1[2] * p2; if (std::fabs(det) <= 1e-14) continue; const double inv = 1.0 / det;
            const double s0 = o[0] - A[i * 9], s1 = o[1] - A[i * 9 + 1], s2 = o[2] - A[i * 9 + 2];
            const double u = (s0 * p0 + s1 * p1 + s2 * p2) * inv; if (u < 0 || u > 1) continue;
            const double q0 = s1 * e1[2] - s2 * e1[1], q1 = s2 * e1[0] - s0 * e1[2], q2 = s0 * e1[1] - s1 * e1[0];
            const double v = (q0 * d[0] + q1 * d[1] + q2 * d[2]) * inv; if (v < 0 || u + v > 1) continue;
            const double t = (e2[0] * q0 + e2[1] * q1 + e2[2] * q2) * inv; if (t < 0 || t >= best) continue; best = t; bp = (int)i;
        }
        T[r] = best; P[r] = bp;
    }
    FILE *g = fopen(argv[3], "wbx"); if (!g) return 2; fwrite(T.data(), 8, nr, g); fwrite(P.data(), 4, nr, g); fclose(g);
    return 0;
}
