// LD_PRELOAD hook that saves every code image an application (e.g. OptiX)
// hands to the CUDA driver. It only copies host memory; it does not change
// what is loaded.
//
// Build: ttu/optix_dump/build.sh
// Use:   OPTIX_DUMP_DIR=/dev/shm/optix_dump LD_PRELOAD=.../liboptixdump.so ./app
//
// Interception paths:
//   1. direct calls to the exported cu* symbols
//   2. cuGetProcAddress / cuGetProcAddress_v2 (returns our wrappers)
//   3. dlsym (returns our wrappers for the names above)
// Every lookup of an interesting name, and every cuGetExportTable call,
// is logged to <dir>/log.txt so we can see which path the app uses.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef int CUresult;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_count = 0;
static FILE *g_log = NULL;
static char g_dir[512];

static void *(*real_dlsym)(void *, const char *) = NULL;

static void init_once(void) {
    if (g_log) return;
    const char *d = getenv("OPTIX_DUMP_DIR");
    snprintf(g_dir, sizeof g_dir, "%s", d ? d : "/dev/shm/optix_dump");
    mkdir(g_dir, 0755);
    char p[600];
    snprintf(p, sizeof p, "%s/log.txt", g_dir);
    g_log = fopen(p, "a");
    if (!g_log) g_log = stderr;
    setvbuf(g_log, NULL, _IOLBF, 0);
}

static void logf_(const char *fmt, const char *a, const void *b) {
    pthread_mutex_lock(&g_lock);
    init_once();
    fprintf(g_log, fmt, a, b);
    pthread_mutex_unlock(&g_lock);
}

static void *get_real_dlsym(void) {
    if (!real_dlsym) {
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (!real_dlsym)
            real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    return (void *)real_dlsym;
}

// Size of an image from its header. kind gets a short tag.
static size_t image_size(const void *img, const char **kind) {
    const uint8_t *p = (const uint8_t *)img;
    uint32_t m;
    memcpy(&m, p, 4);
    if (m == 0x464c457f) {  // ELF (cubin)
        uint64_t shoff; uint16_t shentsize, shnum; uint64_t phoff; uint16_t phentsize, phnum;
        memcpy(&phoff, p + 0x20, 8); memcpy(&shoff, p + 0x28, 8);
        memcpy(&phentsize, p + 0x36, 2); memcpy(&phnum, p + 0x38, 2);
        memcpy(&shentsize, p + 0x3a, 2); memcpy(&shnum, p + 0x3c, 2);
        size_t end = shoff + (size_t)shentsize * shnum;
        // Also cover the end of every section's data.
        for (uint16_t i = 0; i < shnum; i++) {
            const uint8_t *sh = p + shoff + (size_t)i * shentsize;
            uint32_t type; uint64_t off, sz;
            memcpy(&type, sh + 4, 4); memcpy(&off, sh + 0x18, 8); memcpy(&sz, sh + 0x20, 8);
            if (type != 8 /*NOBITS*/ && off + sz > end) end = off + sz;
        }
        size_t pend = phoff + (size_t)phentsize * phnum;
        if (pend > end) end = pend;
        *kind = "cubin";
        return end;
    }
    if (m == 0xba55ed50) {  // fatbin
        uint16_t hs; uint64_t fs;
        memcpy(&hs, p + 6, 2); memcpy(&fs, p + 8, 8);
        *kind = "fatbin";
        return hs + fs;
    }
    if (m == 0x466243b1) {  // fatbin wrapper struct {magic, version, data*, ...}
        const void *data;
        memcpy(&data, p + 8, 8);
        *kind = "fatbin";
        size_t n = image_size(data, kind);
        return n;  // caller must use the inner pointer; handled in save()
    }
    *kind = "text";  // PTX or other NUL terminated text
    return strlen((const char *)p) + 1;
}

static void save(const char *fn, const void *img, size_t known_size) {
    if (!img) return;
    const void *src = img;
    uint32_t m;
    memcpy(&m, img, 4);
    if (m == 0x466243b1) memcpy(&src, (const uint8_t *)img + 8, 8);
    const char *kind = "raw";
    size_t n = known_size ? known_size : image_size(src, &kind);
    pthread_mutex_lock(&g_lock);
    init_once();
    int id = g_count++;
    char p[700];
    snprintf(p, sizeof p, "%s/%03d_%s.%s", g_dir, id, fn, kind);
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(src, 1, n, f); fclose(f); }
    fprintf(g_log, "SAVE %s %s %zu bytes -> %s\n", fn, kind, n, p);
    pthread_mutex_unlock(&g_lock);
}

// ---- wrappers -------------------------------------------------------------
#define REAL(name, type) \
    static type real_##name = NULL; \
    static type get_##name(void) { \
        if (!real_##name) { \
            void *(*ds)(void *, const char *) = get_real_dlsym(); \
            void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD); \
            if (!h) h = dlopen("libcuda.so.1", RTLD_NOW); \
            real_##name = (type)ds(h, #name); \
        } \
        return real_##name; \
    }

typedef CUresult (*t_ModuleLoadData)(void **, const void *);
typedef CUresult (*t_ModuleLoadDataEx)(void **, const void *, unsigned, void *, void **);
typedef CUresult (*t_ModuleLoadFatBinary)(void **, const void *);
typedef CUresult (*t_LibraryLoadData)(void **, const void *, void *, void **, unsigned, void *, void **, unsigned);
typedef CUresult (*t_LinkComplete)(void *, void **, size_t *);
typedef CUresult (*t_GetProcAddress)(const char *, void **, int, uint64_t);
typedef CUresult (*t_GetProcAddress_v2)(const char *, void **, int, uint64_t, void *);
typedef CUresult (*t_GetExportTable)(const void **, const void *);

REAL(cuModuleLoadData, t_ModuleLoadData)
REAL(cuModuleLoadDataEx, t_ModuleLoadDataEx)
REAL(cuModuleLoadFatBinary, t_ModuleLoadFatBinary)
REAL(cuLibraryLoadData, t_LibraryLoadData)
REAL(cuLinkComplete, t_LinkComplete)
REAL(cuGetProcAddress, t_GetProcAddress)
REAL(cuGetProcAddress_v2, t_GetProcAddress_v2)
REAL(cuGetExportTable, t_GetExportTable)

CUresult cuModuleLoadData(void **m, const void *img) {
    save("cuModuleLoadData", img, 0);
    return get_cuModuleLoadData()(m, img);
}
CUresult cuModuleLoadDataEx(void **m, const void *img, unsigned n, void *o, void **v) {
    save("cuModuleLoadDataEx", img, 0);
    return get_cuModuleLoadDataEx()(m, img, n, o, v);
}
CUresult cuModuleLoadFatBinary(void **m, const void *img) {
    save("cuModuleLoadFatBinary", img, 0);
    return get_cuModuleLoadFatBinary()(m, img);
}
CUresult cuLibraryLoadData(void **l, const void *code, void *jo, void **jv, unsigned nj,
                           void *lo, void **lv, unsigned nl) {
    save("cuLibraryLoadData", code, 0);
    return get_cuLibraryLoadData()(l, code, jo, jv, nj, lo, lv, nl);
}
CUresult cuLinkComplete(void *st, void **out, size_t *sz) {
    CUresult r = get_cuLinkComplete()(st, out, sz);
    if (r == 0 && out && *out && sz) save("cuLinkComplete", *out, *sz);
    return r;
}
CUresult cuGetExportTable(const void **tbl, const void *id) {
    logf_("cuGetExportTable %s%p\n", "", id);
    return get_cuGetExportTable()(tbl, id);
}

// Map a name to our wrapper, or NULL.
static void *wrapper_for(const char *s) {
    if (!s) return NULL;
    if (!strcmp(s, "cuModuleLoadData")) return (void *)cuModuleLoadData;
    if (!strcmp(s, "cuModuleLoadDataEx")) return (void *)cuModuleLoadDataEx;
    if (!strcmp(s, "cuModuleLoadFatBinary")) return (void *)cuModuleLoadFatBinary;
    if (!strcmp(s, "cuLibraryLoadData")) return (void *)cuLibraryLoadData;
    if (!strcmp(s, "cuLinkComplete")) return (void *)cuLinkComplete;
    if (!strcmp(s, "cuLinkComplete_v2")) return (void *)cuLinkComplete;
    if (!strcmp(s, "cuGetExportTable")) return (void *)cuGetExportTable;
    return NULL;
}

static int interesting(const char *s) {
    return s && (strstr(s, "Load") || strstr(s, "Link") || strstr(s, "ProcAddress") ||
                 strstr(s, "ExportTable"));
}

CUresult cuGetProcAddress(const char *s, void **pfn, int ver, uint64_t fl) {
    CUresult r = get_cuGetProcAddress()(s, pfn, ver, fl);
    if (interesting(s)) logf_("cuGetProcAddress %s %p\n", s, pfn ? *pfn : NULL);
    void *w = wrapper_for(s);
    if (r == 0 && w && pfn && *pfn) *pfn = w;
    return r;
}
CUresult cuGetProcAddress_v2(const char *s, void **pfn, int ver, uint64_t fl, void *st) {
    CUresult r = get_cuGetProcAddress_v2()(s, pfn, ver, fl, st);
    if (interesting(s)) logf_("cuGetProcAddress_v2 %s %p\n", s, pfn ? *pfn : NULL);
    void *w = wrapper_for(s);
    if (r == 0 && w && pfn && *pfn) *pfn = w;
    return r;
}

void *dlsym(void *h, const char *s) {
    void *(*ds)(void *, const char *) = get_real_dlsym();
    void *r = ds(h, s);
    if (s && !strcmp(s, "cuGetProcAddress") && r) r = (void *)cuGetProcAddress;
    else if (s && !strcmp(s, "cuGetProcAddress_v2") && r) r = (void *)cuGetProcAddress_v2;
    else if (r && wrapper_for(s)) r = wrapper_for(s);
    if (s && !strncmp(s, "cu", 2) && interesting(s)) logf_("dlsym %s %p\n", s, r);
    return r;
}
