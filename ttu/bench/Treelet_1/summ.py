"""summ.py FILE.json ...  per arm medians, spread, speeds, checks"""
import json, sys
for f in sys.argv[1:]:
    j = json.load(open(f)); n = j['rays']; print(f.split('/')[-1], 'rays', n, 'wps', j['wps'], 'K', j['K'])
    for a in j['arms']:
        ms = a['ms']; print('  %-6s med %.3f min %.3f max %.3f  Mray/ms %.0f  bad %d tbad %d clk %s-%s %s' % (a['arm'], a['median_ms'], min(ms), max(ms), n / a['median_ms'] / 1e3, a['bad'], a['tbad'], min(a['clk_mhz']), max(a['clk_mhz']),
              ('own %d stolen %d (%.1f%%) warps/SM %d-%d lanemis %d' % (a['own_chunks'], a['stolen_chunks'], 100. * a['stolen_chunks'] / a['chunks'], a['warps_per_sm_min'], a['warps_per_sm_max'], a['lane_mismatch'])) if 'own_chunks' in a else ''))
