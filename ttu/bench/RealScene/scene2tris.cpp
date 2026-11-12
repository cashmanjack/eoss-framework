// OBJ / PLY loader: scene2tris [--ground] OUT.tris IN1 [IN2 ...]   (our own code; input files are third party scene data, kept in /dev/shm, never in git)
// OUT.tris: float3 a, b, c per triangle (36 bytes), degenerate (zero area or non finite) triangles dropped.
// OBJ: v lines, f lines (v, v/vt, v//vn, v/vt/vn, negative indices), polygons fanned. PLY: ascii or binary (either endianness), vertex x y z (float or double), face list.
// --ground: add two triangles of a ground plane (4x the scene's extent, at the minimum of the largest extent axis; for single object scenes such as Lucy).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
struct V { float x, y, z; };
static std::vector<V> tris;   // 3 per triangle
static void add(const V &a, const V &b, const V &c)
{
    if (!std::isfinite(a.x + a.y + a.z + b.x + b.y + b.z + c.x + c.y + c.z)) return;
    const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
    const float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (nx * nx + ny * ny + nz * nz == 0.f) return;
    tris.push_back(a); tris.push_back(b); tris.push_back(c);
}
static std::vector<char> slurp(const char *p)
{
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); std::vector<char> b(n + 1);
    if (fread(b.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(2); } b[n] = 0; fclose(f); return b;
}
static void load_obj(const char *p)
{
    std::vector<char> b = slurp(p); std::vector<V> v; v.reserve(1 << 20); std::vector<int> idx; char *s = b.data();
    while (*s) {
        char *e = strchr(s, '\n'); if (e) *e = 0;
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) { char *q = s + 1; V x; x.x = strtof(q, &q); x.y = strtof(q, &q); x.z = strtof(q, &q); v.push_back(x); }
        else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            idx.clear(); char *q = s + 1;
            while (*q) {
                while (*q == ' ' || *q == '\t' || *q == '\r') ++q; if (!*q) break;
                char *r; long k = strtol(q, &r, 10); if (r == q) break;
                idx.push_back((int)(k < 0 ? (long)v.size() + k : k - 1));
                while (*r && *r != ' ' && *r != '\t' && *r != '\r') ++r; q = r;
            }
            for (size_t i = 2; i < idx.size(); ++i) {
                if (idx[0] < 0 || idx[i - 1] < 0 || idx[i] < 0 || (size_t)idx[0] >= v.size() || (size_t)idx[i - 1] >= v.size() || (size_t)idx[i] >= v.size()) continue;
                add(v[idx[0]], v[idx[i - 1]], v[idx[i]]);
            }
        }
        if (!e) break; s = e + 1;
    }
}
template <class T> static T rd(const unsigned char *&p, bool swap) { unsigned char t[sizeof(T)]; memcpy(t, p, sizeof(T)); p += sizeof(T); if (swap) std::reverse(t, t + sizeof(T)); T x; memcpy(&x, t, sizeof(T)); return x; }
static void load_ply(const char *p)
{
    std::vector<char> b = slurp(p); const char *h = b.data(); const char *he = strstr(h, "end_header"); if (!he) { fprintf(stderr, "no ply header\n"); exit(2); }
    const char *body = strchr(he, '\n') + 1; std::string hd(h, he); bool ascii = hd.find("format ascii") != std::string::npos, be = hd.find("binary_big_endian") != std::string::npos;
    long nv = 0, nf = 0; char vt[3][16]; int xt[3] = {0, 0, 0}; int vprops = 0; std::vector<std::string> vptype; bool inv = false, inf = false; std::string ctype, itype; int xi = -1, yi = -1, zi = -1;
    size_t pos = 0; while (pos < hd.size()) {
        size_t e = hd.find('\n', pos); if (e == std::string::npos) e = hd.size(); std::string ln = hd.substr(pos, e - pos); pos = e + 1; char a[64], t1[64], t2[64], t3[64], nm[64];
        if (sscanf(ln.c_str(), "element %63s %ld", a, &nv) == 2) { inv = !strcmp(a, "vertex"); inf = !strcmp(a, "face"); if (inv) { vprops = 0; vptype.clear(); } if (inf) nf = nv; if (inv) { } continue; }
        if (inv && sscanf(ln.c_str(), "property %63s %63s", t1, nm) == 2) { if (!strcmp(nm, "x")) xi = vprops; if (!strcmp(nm, "y")) yi = vprops; if (!strcmp(nm, "z")) zi = vprops; vptype.push_back(t1); ++vprops; }
        if (inf && sscanf(ln.c_str(), "property list %63s %63s %63s", t1, t2, t3) == 3) { ctype = t1; itype = t2; }
    }
    (void)vt; (void)xt;
    // re-read the vertex count (the last "element" line seen was face)
    { const char *q = strstr(hd.c_str(), "element vertex"); nv = atol(q + 14); const char *f = strstr(hd.c_str(), "element face"); nf = atol(f + 12); }
    std::vector<V> v(nv); auto size_of = [](const std::string &t) { if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1; if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2; if (t == "double" || t == "float64") return 8; return 4; };
    auto rdnum = [&](const std::string &t, const unsigned char *&q) -> double {
        if (t == "float" || t == "float32") return rd<float>(q, be); if (t == "double" || t == "float64") return rd<double>(q, be);
        if (t == "uchar" || t == "uint8") return rd<uint8_t>(q, be); if (t == "char" || t == "int8") return rd<int8_t>(q, be);
        if (t == "ushort" || t == "uint16") return rd<uint16_t>(q, be); if (t == "short" || t == "int16") return rd<int16_t>(q, be);
        if (t == "uint" || t == "uint32") return rd<uint32_t>(q, be); return rd<int32_t>(q, be); };
    if (ascii) {
        char *s = (char *)body;
        for (long i = 0; i < nv; ++i) { double val[16]; for (int k = 0; k < vprops; ++k) val[k] = strtod(s, &s); v[i] = {(float)val[xi], (float)val[yi], (float)val[zi]}; }
        for (long f = 0; f < nf; ++f) { long c = strtol(s, &s, 10); std::vector<long> ix(c); for (long k = 0; k < c; ++k) ix[k] = strtol(s, &s, 10); for (long k = 2; k < c; ++k) add(v[ix[0]], v[ix[k - 1]], v[ix[k]]); }
    } else {
        const unsigned char *q = (const unsigned char *)body;
        for (long i = 0; i < nv; ++i) { double val[16]; for (int k = 0; k < vprops; ++k) val[k] = rdnum(vptype[k], q); v[i] = {(float)val[xi], (float)val[yi], (float)val[zi]}; }
        (void)size_of; std::vector<long> ix;
        for (long f = 0; f < nf; ++f) { long c = (long)rdnum(ctype, q); ix.resize(c); for (long k = 0; k < c; ++k) ix[k] = (long)rdnum(itype, q); for (long k = 2; k < c; ++k) add(v[ix[0]], v[ix[k - 1]], v[ix[k]]); }
    }
}
int main(int argc, char **argv)
{
    bool ground = false; int a = 1; if (a < argc && !strcmp(argv[a], "--ground")) { ground = true; ++a; }
    if (argc - a < 2) { fprintf(stderr, "usage: scene2tris [--ground] OUT.tris IN...\n"); return 2; }
    const char *out = argv[a++];
    for (; a < argc; ++a) { size_t n = strlen(argv[a]); if (n > 4 && !strcmp(argv[a] + n - 4, ".ply")) load_ply(argv[a]); else load_obj(argv[a]); fprintf(stderr, "%s: %zu triangles so far\n", argv[a], tris.size() / 3); }
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (auto &p : tris) { const float c[3] = {p.x, p.y, p.z}; for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], c[k]); hi[k] = std::max(hi[k], c[k]); } }
    if (ground) {
        int up = 0; for (int k = 1; k < 3; ++k) if (hi[k] - lo[k] > hi[up] - lo[up]) up = k;
        const int p1 = (up + 1) % 3, p2 = (up + 2) % 3; const float cx = 0.5f * (lo[p1] + hi[p1]), cz = 0.5f * (lo[p2] + hi[p2]), r = 2.f * std::max(hi[p1] - lo[p1], hi[p2] - lo[p2]) + (hi[up] - lo[up]);
        auto pt = [&](float s1, float s2) { float c[3]; c[up] = lo[up] - 1e-3f * (hi[up] - lo[up]); c[p1] = cx + s1 * r; c[p2] = cz + s2 * r; return V{c[0], c[1], c[2]}; };
        add(pt(-1, -1), pt(1, -1), pt(1, 1)); add(pt(-1, -1), pt(1, 1), pt(-1, 1));
        fprintf(stderr, "ground plane added below axis %d\n", up);
    }
    FILE *f = fopen(out, "wb"); fwrite(tris.data(), sizeof(V), tris.size(), f); fclose(f);
    printf("{\"triangles\": %zu, \"lo\": [%g, %g, %g], \"hi\": [%g, %g, %g]}\n", tris.size() / 3, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    return 0;
}
