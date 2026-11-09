// CUPTI injection library that saves the cubin of every module the CUDA
// driver loads, including modules loaded through private driver interfaces
// (as OptiX does). It only copies what CUPTI hands it.
//
// Build: ttu/optix_dump/build.sh
// Use:   CUPTI_DUMP_DIR=/dev/shm/cupti_dump
//        CUDA_INJECTION64_PATH=.../libcuptidump.so
//        LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH ./app
//        (all three on one command line)
#include <cupti.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_log = NULL;
static char g_dir[512];
static int g_count = 0;

static void open_log(void) {
    const char *d = getenv("CUPTI_DUMP_DIR");
    snprintf(g_dir, sizeof g_dir, "%s", d ? d : "/dev/shm/cupti_dump");
    mkdir(g_dir, 0755);
    char p[600];
    snprintf(p, sizeof p, "%s/log.txt", g_dir);
    g_log = fopen(p, "a");
    if (!g_log) g_log = stderr;
    setvbuf(g_log, NULL, _IOLBF, 0);
}

static void CUPTIAPI callback(void *ud, CUpti_CallbackDomain dom, CUpti_CallbackId cbid,
                              const void *cbdata) {
    (void)ud;
    if (dom != CUPTI_CB_DOMAIN_RESOURCE) return;
    if (cbid != CUPTI_CBID_RESOURCE_MODULE_LOADED &&
        cbid != CUPTI_CBID_RESOURCE_MODULE_PROFILED)
        return;
    const CUpti_ResourceData *rd = (const CUpti_ResourceData *)cbdata;
    const CUpti_ModuleResourceData *md = (const CUpti_ModuleResourceData *)rd->resourceDescriptor;
    if (!md) return;
    pthread_mutex_lock(&g_lock);
    int id = g_count++;
    const char *ev = cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED ? "loaded" : "profiled";
    if (md->pCubin && md->cubinSize) {
        char p[700];
        snprintf(p, sizeof p, "%s/%03d_mod%u_%s.cubin", g_dir, id, md->moduleId, ev);
        FILE *f = fopen(p, "wb");
        if (f) { fwrite(md->pCubin, 1, md->cubinSize, f); fclose(f); }
        fprintf(g_log, "MODULE %s id=%u size=%zu -> %s\n", ev, md->moduleId, md->cubinSize, p);
    } else {
        fprintf(g_log, "MODULE %s id=%u no cubin\n", ev, md->moduleId);
    }
    pthread_mutex_unlock(&g_lock);
}

int InitializeInjection(void) {
    open_log();
    CUpti_SubscriberHandle sub;
    CUptiResult r = cuptiSubscribe(&sub, (CUpti_CallbackFunc)callback, NULL);
    if (r != CUPTI_SUCCESS) {
        const char *s = NULL;
        cuptiGetResultString(r, &s);
        fprintf(g_log, "cuptiSubscribe failed: %s\n", s ? s : "?");
        return 0;
    }
    r = cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE);
    fprintf(g_log, "InitializeInjection ok, enableDomain=%d\n", (int)r);
    return 1;
}
