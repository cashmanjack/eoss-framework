#pragma once
#include <optix_types.h>
#define KMAX 32
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned front, gas_index, pad; };
struct Cur { float t; unsigned id; unsigned k; unsigned pad; };   // per ray cursor of the top level walk: entry parameter (pulled back origin) and id of the last treelet queued, visits so far
struct Params {
    // as Treelet_2 (same order)
    const Ray8 *rays; RefOut *out;
    OptixTraversableHandle handle;               // full GAS (root start)
    const OptixTraversableHandle *tl_handle;     // one GAS per treelet
    const unsigned *tl;                          // treelet of every ray (old arms)
    const unsigned *n_dev; unsigned use_tl;
    unsigned *head; const unsigned *qend; unsigned nq, pinned;   // old persistent kernel
    unsigned *stats;                             // old: 0 own chunks, 1 stolen, 2 lane mismatches, 3 chunks
    unsigned *smhist; unsigned *warp_sm;
    // Treelet_3
    OptixTraversableHandle ias_shell, ias_real;  // IAS whose instances are the treelets: shell proxy (box) children for the inspector, the real treelet GASes for the natural IAS baseline
    const float *boxes;                          // 6 floats per treelet (tight box plus margin)
    float pull; unsigned C, record, steal_tries;
    Cur *cur;
    unsigned *qhead, *qtail; const unsigned *qbase, *qcap; unsigned *qbuf;
    unsigned *tcur; const unsigned *tlo, *thi;
    unsigned *outstanding, *hist;
    const unsigned *toff, *tri_sorted;
    unsigned short *vis;
    unsigned long long *cyc;                     // 0 pop/probe, 1 load + treelet trace, 2 top level step, 3 requeue, 4 idle, 5 warps
    unsigned *cnt;                               // 0 chunks, 1 stolen chunks, 2 requeues (tail atomics), 3 lanes visited, 4 overflow, 5 spin timeouts, 6 max visits, 7 top level steps
    // replay (one level)
    unsigned *qpub;                              // per treelet: entries published (stored); consumers claim only up to the published prefix
    const unsigned *lvl_idx; const Ray8 *lvl_ray; const unsigned *lvl_key; unsigned lvl_level; unsigned *lvl_head; const unsigned *lvl_end;
};
