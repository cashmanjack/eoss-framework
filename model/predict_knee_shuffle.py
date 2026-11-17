"""Knee term vs the F208 shuffle penalty (2026-11-17, tethys). CPU only, existing data only, nothing refit.
    /usr/bin/python3 model/predict_knee_shuffle.py predict OUT.json    prediction table (run and committed BEFORE the comparison)
    /usr/bin/python3 model/predict_knee_shuffle.py compare PRED.json   compare with the F208 measured shuffled speedups (notes/hs2 measure files)
Form (written before the numbers were computed, model/predictions_knee_shuffle.md):
  h(order)  = 32 / share_ratio = 32 * dedupe_total / raw_total  (share_ratio = raw / dedupe node lines of the replay reuse files, F123 json): mean distinct node lines
              per warp step if every lane is active (an upper bound: inactive lanes lower it)
  K(h)      = max(1, h / CAP), CAP = 10 distinct lines per step (F201)
  speedup of the shuffled order over the natural order = F123 predicted speedup x K(h_natural) / K(h_shuffled)
"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); HS2 = os.path.join(ROOT, 'notes', 'hs2'); CAP = 10.0
SETS = ['conference_camera', 'sponza_camera', 'rungholt_camera', 'conference_bounce', 'sponza_bounce', 'rungholt_bounce']
F123 = os.path.join(HS2, 'f123_scenes_conference_camera_conference_bounce_sponza_camera_sponza_bounce_rungholt_camera_rungholt_bounce.json')


def K(h): return max(1.0, h / CAP)


def predict():
    j = json.load(open(F123)); out = {}
    for t in SETS:
        sr = j[t]['share_ratio']; hn, hs = 32.0 / sr['natural'], 32.0 / sr['shuffled']
        f = j[t]['f123']['shuffled']
        out[t] = dict(h_natural=hn, h_shuffled=hs, K_natural=K(hn), K_shuffled=K(hs), f123=f, knee=f * K(hn) / K(hs))
    return out


def measured(t):
    m = json.load(open(sorted(glob.glob(os.path.join(HS2, 'measure_%s_2*.json' % t)))[-1]))['results']; d = {r['order']: r['ms'] for r in m}
    return d['natural'] / d['shuffled']


if __name__ == '__main__':
    mode, fn = sys.argv[1], sys.argv[2]
    if mode == 'predict':
        P = predict(); json.dump(P, open(fn, 'x'), indent=1)
        print('%-18s %7s %7s %6s %6s %8s %8s' % ('set', 'h_nat', 'h_shuf', 'K_nat', 'K_shuf', 'F123', 'knee'))
        for t in SETS: p = P[t]; print('%-18s %7.2f %7.2f %6.2f %6.2f %8.3f %8.3f' % (t, p['h_natural'], p['h_shuffled'], p['K_natural'], p['K_shuffled'], p['f123'], p['knee']))
    else:
        P = json.load(open(fn)); print('%-18s %8s %8s %8s %9s %9s' % ('set', 'measured', 'F123', 'knee', 'logerr F123', 'logerr knee'))
        e = {'f123': [], 'knee': []}
        for t in SETS:
            m = measured(t); p = P[t]; a, b = np.log(p['f123'] / m), np.log(p['knee'] / m); e['f123'].append(a); e['knee'].append(b)
            print('%-18s %8.3f %8.3f %8.3f %9.3f %9.3f' % (t, m, p['f123'], p['knee'], a, b))
        for nm, sl in (('camera', slice(0, 3)), ('bounce', slice(3, 6)), ('all six', slice(0, 6))):
            print('%-8s mean |log err|  F123 %.3f  knee %.3f   rms  F123 %.3f  knee %.3f' % (nm, np.mean(np.abs(e['f123'][sl])), np.mean(np.abs(e['knee'][sl])), np.sqrt(np.mean(np.square(e['f123'][sl]))), np.sqrt(np.mean(np.square(e['knee'][sl])))))
