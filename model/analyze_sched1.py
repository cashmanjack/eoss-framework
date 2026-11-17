"""Sched_1 analysis (predictions model/predictions_sched.md, committed before the run). CPU only.   /usr/bin/python3 model/analyze_sched1.py [DIR=notes/sched]
Per file (mask, segment length, dependent or not; 40 launches, the first dropped as warm up): per warp the median over launches of (stamp[64] - stamp[0]) / 64 = cycles per segment;
for each scheduler group (warp id mod 4) with at least 2 active warps: the mean run length of the stamp sequence (all stamps of the group sorted by time, runs of the same warp; 1 = strict alternation at segment grain,
64 = one warp runs to the end before the next) and, per launch, which warp has the earliest stamp[0] (release order) and by how many cycles the stamps[0] of the group differ."""
import glob, json, sys, statistics as st, collections
D = sys.argv[1] if len(sys.argv) > 1 else 'notes/sched'
for f in sorted(glob.glob(D + '/s1_m*.json')):
    j = json.load(open(f)); L = j['launches'][1:]; ws = sorted(int(w) for w in L[0]['warps'])
    seg = {w: st.median([(l['warps'][str(w)][64] - l['warps'][str(w)][0]) / 64 for l in L]) for w in ws}
    print('%s: mask %x seglen %d dep %d warps %s' % (f.split('/')[-1][:28], j['mask'], j['seglen'], j['dep'], ws))
    print('   cycles per segment by warp (median over launches):', {w: round(v, 1) for w, v in seg.items()} if len(ws) <= 8 else 'min %.1f median %.1f max %.1f' % (min(seg.values()), st.median(seg.values()), max(seg.values())))
    groups = collections.defaultdict(list)
    for w in ws: groups[w % 4].append(w)
    for g, members in sorted(groups.items()):
        if len(members) < 2 or (len(ws) > 8 and g > 0): continue
        runs = []; first = collections.Counter(); spread = []
        for l in L:
            ev = sorted((l['warps'][str(w)][s], w) for w in members for s in range(1, 65)); r = 1
            for a, b in zip(ev, ev[1:]): 
                if a[1] == b[1]: r += 1
                else: runs.append(r); r = 1
            runs.append(r)
            f0 = sorted((l['warps'][str(w)][0], w) for w in members); first[f0[0][1]] += 1; spread.append(f0[-1][0] - f0[0][0])
        print('   scheduler %d (warps %s): mean run length %.2f, max %d; earliest release warp counts %s; release spread median %d cycles' % (g, members, st.mean(runs), max(runs), dict(first), st.median(spread)))
