"""Phase 15 item 1 scoring: BITTU probes on one warp (predictions model/priorworks/predictions_phase15.md, e57d74e).
Usage: /usr/bin/python3 model/priorworks/analyze_bittu_w.py notes/q45/bittu/summary_w_TS.json > model/priorworks/results_bittu_oneprobe.md"""
import json, sys
d = json.load(open(sys.argv[1]))
print('# BITTU timing with probes on one warp only (2026-11-19, triton GPU 1, RTX 6000 Ada, checked GAS group, 1 block, scene tri1, ray hit; file %s)\n' % sys.argv[1].split('/')[-1])
print('Corrected = window(stage 1 under the tool) - window(stage 0 twin under the tool), both with the probed warp only recording; harness = stage 1 minus stage 0 group time of the same warp from the plain run (cycles). "5 launches" = median over launches 2 to 5 of the 5 launch run; "1 launch" = the 1 launch run. Non probed: median over the other warps of the group time (stage 1 minus stage 0 harness clocks) in the plain run and in the instrumented run (both stubs and probes of the probed warp present).\n')
print('| case (warps, probed warp) | launches | window with group | window twin | probe cost | harness group | corrected | corrected / harness | non probed warps, plain | non probed warps, under the tool |\n|---|---|---|---|---|---|---|---|---|---|')
ok = 0; within10 = 0; n = 0; ratios = {}
for k, v in d.items():
    for r in ('1', '5'):
        x = v[r]; print('| %s | %s | %.0f | %.0f | %.0f | %.0f | %.0f | %.2f | %.0f | %.0f |' % (k, r, x['window_s1'], x['window_twin'], x['probe_cost'], x['harness_group_probed_warp'], x['corrected'], x['corrected_over_harness'], x['nonprobed_group_plain'], x['nonprobed_group_instrumented']))
    x = v['5']; n += 1; ok += 0.95 <= x['corrected_over_harness'] <= 1.05; within10 += 0.90 <= x['corrected_over_harness'] <= 1.10; ratios[k] = x['corrected_over_harness']
print('\nCases passing the rule (median ratio within 0.95 to 1.05): %d of %d; within 0.90 to 1.10: %d of %d. All counts and decodes held: %s.' % (ok, n, within10, n, all(v[r]['counts_ok'] and v[r]['decoded'] for v in d.values() for r in ('1', '5'))))
print('Ratios (5 launch run): ' + ', '.join('%s %.2f' % kv for kv in ratios.items()))
