# Mochi last arm: OptiX setup flips, results (predictions: model/FORM_options_arm.md, committed a934137 before the run)

2026-11-16, triton GPU 1, OptiX only, no TTU code. Replica (Mochi program, instance level, real dataset in Mochi's Morton order, 100,000 queries, 200 warmups, 400 timed launches per arm, two rounds, rotated order, every run exit 0). Raw private: notes/q32/options_20261116T193158.json.

| arm | span median us (round 0 / 1) | warp cycles median | p90 over median | calls per query |
| base (old PTX) | 11.26 / 11.26 | 19,200 | 1.15 | 1.075 |
| refac (my edited source, same options; control) | 11.26 / 11.26 | 19,200 | 1.15 | 1.075 |
| F1 attribute values 2 | 11.26 / 11.26 | 19,240 | 1.15 | 1.075 |
| F2 sphere primitive flag | 11.26 / 11.26 | 19,200 | 1.15 | 1.075 |
| F3 maxTraceDepth 2 | 11.26 / 11.26 | 19,230 | 1.15 | 1.075 |
| F4 stack sizes as OWL (2048 x 3, depth 2) | 11.26 / 11.26 | 19,470 | 1.15 | 1.075 |
| F5 program data from SBT records | 12.29 / 11.26 | 18,000 | 1.13 | 1.075 |
| F6 PTX as Mochi (.target sm_52, -O3 -lineinfo) | 11.26 / 11.26 | 19,200 | 1.15 | 1.075 |
| ALL (F1 to F6) | 12.29 / 12.29 | 18,070 | 1.13 | 1.075 |
(Timer step is 1.02 us.) Mochi frame 0: 100 us, median warp 40,000 cycles, p90 189,000, p90 over median 4.7.

Result. No flip produces it (rule: span at least 34 us and p90 over median at least 2.0). Prediction held: every span inside 9.6 to 13.0 us, p90 over median below 1.5. F5 and ALL are one timer step (9 percent) slower in span but 6 to 7 percent faster in warp cycles: not an effect of the size sought. Both stop rules end here: none of the six differences, alone or together, gives the warp tail or the 9 times.
What this rules out: OptiX pipeline and module options as OWL sets them, OWL's fixed stack sizes (the stack default of the replica is not readable, so "stack size as OWL" is tested but the replica's own value is unknown), SBT record layout and reads, PTX target and flags. Same traversal work and same count (F172), same program (F171).
Not tested, left unexplained: anything else specific to the OWL process or Mochi's run (allocation layout of its buffers, the atomics on real accumulated forces after earlier frames, its launch parameter copy before each launch, other kernels sharing the context). The gap (100 against 11.3 microseconds at frames 0 to 4, with a tail of slow warps) is recorded as unexplained.
