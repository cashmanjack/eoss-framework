"""Score the committed held out predictions (model/predictions_heldout.md) against the measured speedups. /usr/bin/python3 model/score_heldout.py   (reads notes/hs/f123_predictions_sponza_conference_rungholt.json, notes/hs/measure_*.json)"""
import json, numpy as np
P = json.load(open('notes/hs/f123_predictions_sponza_conference_rungholt.json')); allx = {'f123': [], 'memory4': []}
for tag in ('sponza', 'conference', 'rungholt'):
    M = {r['order']: r['ms'] for r in json.load(open('notes/hs/measure_%s.json' % tag))['results']}; wrong = sum(r['wrong'] for r in json.load(open('notes/hs/measure_%s.json' % tag))['results'])
    print('%s (wrong results: %d)' % (tag, wrong)); print('   order         measured   F123  err      Memory_4  err')
    e = {'f123': [], 'memory4': []}; ml = []
    for k in P[tag]['f123']:
        m = M['natural'] / M[k]; ml.append(np.log(m))
        for f in ('f123', 'memory4'): e[f].append(np.log(P[tag][f][k] / m))
        print('   %-12s  %.3f      %.3f  %+.3f   %.3f  %+.3f' % (k, m, P[tag]['f123'][k], e['f123'][-1], P[tag]['memory4'][k], e['memory4'][-1]))
    r = lambda x: float(np.sqrt(np.mean(np.square(x))))
    print('   rms log error F123 %.3f  Memory_4 form %.3f  spread of measured log speedups %.3f  (F123 mean signed %+.3f)' % (r(e['f123']), r(e['memory4']), r(ml), np.mean(e['f123'])))
    for f in allx: allx[f] += e[f]
print('pooled over %d points: F123 %.3f  Memory_4 form %.3f' % (len(allx['f123']), float(np.sqrt(np.mean(np.square(allx['f123'])))), float(np.sqrt(np.mean(np.square(allx['memory4']))))))
