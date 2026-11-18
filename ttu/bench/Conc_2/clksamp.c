// Conc_2/Conc_3: NVML clock and power sampler, launch window version (Phase 10d fix; the first version wrote every sample, 3 GB).
// clksamp INDEX OUTFILE : samples SM MHz, memory MHz and power (mW) about every 1 ms into a ring (last 400 s) and writes NOTHING until a window arrives on stdin:
//   line "T0 T1 KEY LAUNCH" (CLOCK_MONOTONIC seconds) -> the rows with T0-0.01 <= t <= T1+0.01 are appended to OUTFILE (created with mode x, never overwritten). EOF on stdin ends it.
// Output: t,sm,mem,mw,key,launch
#include <nvml.h>
#include <poll.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CAP 400000
typedef struct { double t; unsigned sm, mem, mw; } S;
static S ring[CAP]; static long n = 0;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv)
{
    if (argc < 3) return 2; nvmlDevice_t d; if (nvmlInit() || nvmlDeviceGetHandleByIndex(atoi(argv[1]), &d)) return 3;
    FILE *f = fopen(argv[2], "wx"); if (!f) return 4; fprintf(f, "t,sm,mem,mw,key,launch\n"); fflush(f);
    char line[512]; size_t ll = 0; struct pollfd p = {0, POLLIN, 0}; int eof = 0;
    while (!eof) {
        S *s = &ring[n % CAP]; s->t = now(); s->sm = s->mem = s->mw = 0; nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &s->sm); nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &s->mem); nvmlDeviceGetPowerUsage(d, &s->mw); ++n;
        if (poll(&p, 1, 1) > 0) {
            char c; ssize_t r;
            while (p.revents & (POLLIN | POLLHUP) && (r = read(0, &c, 1)) == 1) {
                if (c != '\n') { if (ll < sizeof line - 1) line[ll++] = c; continue; }
                line[ll] = 0; ll = 0; double t0, t1; char key[256]; int la = 0;
                if (sscanf(line, "%lf %lf %255s %d", &t0, &t1, key, &la) >= 3) {
                    long lo = n > CAP ? n - CAP : 0;
                    for (long i = lo; i < n; ++i) { S *q = &ring[i % CAP]; if (q->t >= t0 - 0.01 && q->t <= t1 + 0.01) fprintf(f, "%.6f,%u,%u,%u,%s,%d\n", q->t, q->sm, q->mem, q->mw, key, la); }
                    fflush(f);
                }
                { struct pollfd q2 = {0, POLLIN, 0}; if (poll(&q2, 1, 0) <= 0) break; }
            }
            if (p.revents & POLLHUP && !(p.revents & POLLIN)) eof = 1; else if (p.revents & POLLHUP) { struct pollfd q3 = {0, POLLIN, 0}; if (poll(&q3, 1, 0) <= 0) eof = 1; }
        }
    }
    fclose(f); nvmlShutdown(); return 0;
}
