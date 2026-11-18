// Conc_2: NVML clock and power sampler. clksamp INDEX OUTFILE SECONDS : one row per sample: CLOCK_MONOTONIC s, SM MHz, memory MHz, power mW. File created with mode x (never overwritten).
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(int argc, char **argv)
{
    if (argc < 4) return 2; nvmlDevice_t d; if (nvmlInit() || nvmlDeviceGetHandleByIndex(atoi(argv[1]), &d)) return 3;
    FILE *f = fopen(argv[2], "wx"); if (!f) return 4; const double T = atof(argv[3]); struct timespec t, t0; clock_gettime(CLOCK_MONOTONIC, &t0); fprintf(f, "t,sm,mem,mw\n");
    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &t); const double s = t.tv_sec + t.tv_nsec * 1e-9; if (s - (t0.tv_sec + t0.tv_nsec * 1e-9) > T) break;
        unsigned sm = 0, mem = 0, mw = 0; nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &sm); nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &mem); nvmlDeviceGetPowerUsage(d, &mw);
        fprintf(f, "%.6f,%u,%u,%u\n", s, sm, mem, mw);
    }
    fclose(f); nvmlShutdown(); return 0;
}
