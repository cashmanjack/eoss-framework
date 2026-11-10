// BITTU kill tests (NVBit 1.8.1). Inject with CUDA_INJECTION64_PATH into a setup3 run.
//   BITTU_MODE=list   test 1 and 4: print every launched function; for ttu_one, every instruction with
//                     NVBit's opcode name and our classification from the raw bytes (TTU opcodes by low 12 bits)
//   BITTU_MODE=probe  test 2: empty calls after the instruction right before TTUOPEN and before the
//                     instruction right after the fused group (no TTU instruction is moved)
//   BITTU_MODE=regs   test 3: the "before" call reads the TTUST source registers (warp 0, every lane)
//   BITTU_OUT=file    where the report goes (default stderr)
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <set>
#include <vector>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#define NREG 12
static const int REGS[NREG] = {12, 13, 14, 15, 16, 17, 82, 83, 70, 71, 34, 35};   // 0x40, 0x50 sources, root and instance words
__managed__ unsigned long long hits_before = 0, hits_after = 0;
__managed__ uint32_t regbuf[32][NREG];

static int mode = 0;
static FILE *out = stderr;
static std::set<CUfunction> done;
static int launches = 0;

static void collect(uint8_t b, void *u) { ((std::vector<uint8_t> *)u)->push_back(b); }
static unsigned low12(Instr *i) {
    std::vector<uint8_t> v; i->getSassBinary(collect, &v);
    return v.size() >= 2 ? (v[0] | (v[1] << 8)) & 0xfff : 0xffff;
}
static const char *ttu_name(unsigned op) {
    switch (op) { case 0x3d0: return "TTUOPEN"; case 0x9d4: return "TTUMACROFUSE"; case 0x3d1: return "TTUST";
                  case 0x3d3: return "TTUGO"; case 0x3d2: return "TTULD"; default: return nullptr; }
}

void nvbit_at_init() {
    const char *m = getenv("BITTU_MODE");
    mode = !m || !strcmp(m, "list") ? 0 : !strcmp(m, "probe") ? 1 : !strcmp(m, "regs") ? 2 : 0;
    const char *o = getenv("BITTU_OUT"); if (o) out = fopen(o, "w");
    fprintf(out, "bittu kill: mode %d\n", mode);
}

static void instrument(CUcontext ctx, CUfunction f) {
    std::vector<CUfunction> fs = nvbit_get_related_functions(ctx, f); fs.push_back(f);
    for (CUfunction g : fs) {
        if (!done.insert(g).second) continue;
        const char *name = nvbit_get_func_name(ctx, g);
        const std::vector<Instr *> &ins = nvbit_get_instrs(ctx, g);
        int nttu = 0, named = 0; for (Instr *i : ins) if (ttu_name(low12(i))) { ++nttu; named += !strncmp(i->getOpcode(), "TTU", 3); }
        fprintf(out, "function %s: %zu instructions, %d TTU by raw opcode, %d of them named TTU* by NVBit\n", name, ins.size(), nttu, named);
        if (!strstr(name, "ttu_one")) continue;
        // NVBit's instruction list may hide TTU instructions: find the group as a gap in the offsets, then
        // read the function's code from GPU memory and check the gap holds exactly one complete fused group.
        int gk = -1;
        for (size_t k = 0; k + 1 < ins.size(); ++k)
            if (ins[k + 1]->getOffset() - ins[k]->getOffset() > 16) {
                fprintf(out, "  offset gap after [%zu] +0x%04x %s: %u bytes hidden (%u instructions)\n", k, ins[k]->getOffset(), ins[k]->getSass(),
                        ins[k + 1]->getOffset() - ins[k]->getOffset() - 16, (ins[k + 1]->getOffset() - ins[k]->getOffset()) / 16 - 1);
                if (gk < 0) gk = (int)k; else gk = -2;
            }
        if (gk < 0) { fprintf(out, "  %s\n", gk == -1 ? "no hidden instructions" : "more than one gap: refusing to instrument"); continue; }
        Instr *before = ins[gk], *after = ins[gk + 1];
        const uint32_t g0 = before->getOffset() + 16, g1 = after->getOffset();
        std::vector<uint8_t> code(g1 - g0);
        const uint64_t fa = nvbit_get_func_addr(ctx, g);
        if (cuMemcpyDtoH(code.data(), (CUdeviceptr)(fa + g0), code.size()) != CUDA_SUCCESS) {
            // Fallback: the group's place in the checked binary (check_seq layout: BITTU_GROUP_OFF, BITTU_GROUP_LEN).
            const char *eo = getenv("BITTU_GROUP_OFF"), *el = getenv("BITTU_GROUP_LEN");
            const bool match = eo && el && strtoul(eo, 0, 0) == g0 && strtoul(el, 0, 0) * 16 == code.size();
            fprintf(out, "  code not readable through NVBit's address; checked layout says group at %s, %s instructions: %s\n",
                    eo ? eo : "?", el ? el : "?", match ? "gap matches" : "MISMATCH");
            if (!match) { fprintf(out, "  refusing to instrument\n"); continue; }
            fprintf(out, "  probe points: after +0x%04x %s, before +0x%04x %s\n", before->getOffset(), before->getSass(), after->getOffset(), after->getSass());
            goto insert;
        }
        {   // verification from the code bytes (skipped by the layout fallback above)
        auto wd = [&](size_t k, int hi) { uint64_t w = 0; for (int b = 0; b < 8; ++b) w |= (uint64_t)code[16 * k + 8 * hi + b] << (8 * b); return w; };
        const size_t n = code.size() / 16; bool good = n >= 3 && (wd(0, 0) & 0xfff) == 0x3d0 && (wd(1, 0) & 0xfff) == 0x9d4;
        const int fuse = good ? (int)((wd(1, 0) >> 40) & 0x1f) : -1;
        good = good && (size_t)fuse + 2 == n;
        for (size_t k = 2; good && k < n; ++k) { const unsigned op = wd(k, 0) & 0xfff; good = op == 0x3d1 || op == 0x3d3 || op == 0x3d2; }
        const bool close_last = good && (wd(n - 1, 0) & 0xfff) == 0x3d2 && ((wd(n - 1, 1) >> 10) & 1);   // bit 74
        fprintf(out, "  hidden code at +0x%04x: %zu instructions, TTUOPEN/TTUMACROFUSE %d/all TTU: %s, last is TTULD with bit 74: %s\n",
                g0, n, fuse, good ? "yes" : "NO", close_last ? "yes" : "NO");
        for (size_t k = 0; k < n; ++k) { const char *t = ttu_name(wd(k, 0) & 0xfff); fprintf(out, "    +0x%04zx %s\n", g0 + 16 * k, t ? t : "not TTU"); }
        if (!good || !close_last) { fprintf(out, "  not one complete fused group: refusing to instrument\n"); continue; }
        fprintf(out, "  probe points: after +0x%04x %s, before +0x%04x %s\n", before->getOffset(), before->getSass(), after->getOffset(), after->getSass());
        }
    insert:
        if (mode == 1) {
            nvbit_insert_call(before, "bittu_before", IPOINT_AFTER);
            nvbit_add_call_arg_const_val64(before, (uint64_t)&hits_before);
            nvbit_insert_call(after, "bittu_after", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val64(after, (uint64_t)&hits_after);
        } else if (mode == 2) {
            nvbit_insert_call(before, "bittu_regs", IPOINT_AFTER);
            nvbit_add_call_arg_guard_pred_val(before);
            nvbit_add_call_arg_const_val64(before, (uint64_t)&hits_before);
            nvbit_add_call_arg_const_val64(before, (uint64_t)&regbuf[0][0]);
            for (int r = 0; r < NREG; ++r) nvbit_add_call_arg_reg_val(before, REGS[r], true);
            nvbit_insert_call(after, "bittu_after", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val64(after, (uint64_t)&hits_after);
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params, CUresult *pStatus) {
    if (cbid != API_CUDA_cuLaunchKernel && cbid != API_CUDA_cuLaunchKernel_ptsz && cbid != API_CUDA_cuLaunchKernelEx && cbid != API_CUDA_cuLaunchKernelEx_ptsz) return;
    CUfunction f = (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) ? ((cuLaunchKernelEx_params *)params)->f : ((cuLaunchKernel_params *)params)->f;
    if (!is_exit) {
        ++launches;
        fprintf(out, "launch %d: %s\n", launches, nvbit_get_func_name(ctx, f));
        instrument(ctx, f);
        nvbit_enable_instrumented(ctx, f, mode != 0 && strstr(nvbit_get_func_name(ctx, f), "ttu_one") != nullptr);
    } else if (strstr(nvbit_get_func_name(ctx, f), "ttu_one")) {
        cudaDeviceSynchronize();
        fprintf(out, "after launch %d: probe calls before %llu, after %llu\n", launches, hits_before, hits_after);
        if (mode == 2) for (int l = 0; l < 32; ++l) {
            fprintf(out, "  lane %2d regs", l); for (int r = 0; r < NREG; ++r) fprintf(out, " R%d=%08x", REGS[r], regbuf[l][r]); fprintf(out, "\n");
        }
        fflush(out);
    }
}

void nvbit_at_term() { fprintf(out, "bittu kill: %d launches seen\n", launches); if (out != stderr) fclose(out); }
