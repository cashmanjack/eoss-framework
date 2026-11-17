"""Sched_2 analysis (predictions model/predictions_sched.md). CPU only.   /usr/bin/python3 model/analyze_sched2.py [DIR=notes/sched]
One TTU warp latency (warp_clk end - start, first launch of every run dropped), arms s1_alone, s1_alu, s0_alone, s0_alu, 5 repetitions of 30 launches; medians of the run medians, and
extra delay of a returning warp = (L1_alu - L1_alone) - (L0_alu - L0_alone)."""
import glob, json, re, statistics as st, sys, collections
D = sys.argv[1] if len(sys.argv) > 1 else 'notes/sched'; R = collections.defaultdict(list); allv = collections.defaultdict(list)
for f in sorted(glob.glob(D + '/s2_s[01]_*_r[0-9]_*.json')):
    arm = re.search(r's2_(s[01]_(?:alone|alu))_r', f).group(1); j = json.load(open(f)); v = [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches'][1:]]
    R[arm].append(st.median(v)); allv[arm] += v
for a in ('s0_alone', 's0_alu', 's1_alone', 's1_alu'):
    print('%-9s runs %d  run medians %s  median %.1f  (all launches: min %d max %d)' % (a, len(R[a]), [round(x, 1) for x in R[a]], st.median(R[a]), min(allv[a]), max(allv[a])))
m = {a: st.median(R[a]) for a in R}
print('stage 0 change with compute: %+.1f cycles; stage 1 change with compute: %+.1f cycles; extra delay of the returning warp (stage 1 change minus stage 0 change): %+.1f cycles (%.1f percent of the %.0f cycle stage 1 latency alone)' %
      (m['s0_alu'] - m['s0_alone'], m['s1_alu'] - m['s1_alone'], (m['s1_alu'] - m['s1_alone']) - (m['s0_alu'] - m['s0_alone']), 100 * ((m['s1_alu'] - m['s1_alone']) - (m['s0_alu'] - m['s0_alone'])) / m['s1_alone'], m['s1_alone']))
