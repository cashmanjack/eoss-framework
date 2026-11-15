"""Figures 6 and 7 (CPU only): payload size vs registers and speed (Finding 112), scene only prediction on the 4070 Ti (Finding 113). Output notes/figures."""
import json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), '..'); O = os.path.join(R, 'notes', 'figures')
rows = json.load(open(os.path.join(R, 'notes/io/payload/payload.json')))
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for rs, c, lab in (('rand', '#1b6ca8', 'random rays'), ('coh', '#d98c1f', 'coherent rays')):
    for a, ls in ((2, '-'), (8, '--')):
        r = sorted([x for x in rows if x['custom'] == 0 and x['nattr'] == a and x['rays'] == rs], key=lambda x: x['npay'])
        ax[0].plot([x['npay'] for x in r], [x['ncu']['launch__registers_per_thread'] for x in r], 'o' + ls, color=c, ms=3, label='%s, attr %d' % (lab, a)) if rs == 'rand' else None
        ax[1].plot([x['npay'] for x in r], [1048576 / x['ms'] / 1000 for x in r], 'o' + ls, color=c, ms=3, label='%s, attr %d' % (lab, a))
        ax[2].plot([x['npay'] for x in r], [x['ncu']['sm__warps_active.avg.per_cycle_active'] for x in r], 'o' + ls, color=c, ms=3)
ax[0].axhline(128, color='gray', lw=0.7, ls=':'); ax[0].set_xlabel('payload values'); ax[0].set_ylabel('registers per thread'); ax[0].set_title('registers (128 = 4 blocks of 128 threads)')
ax[1].set_xlabel('payload values'); ax[1].set_ylabel('million rays per second'); ax[1].set_yscale('log'); ax[1].set_title('speed (triangles)'); ax[1].legend(fontsize=6)
ax[2].set_xlabel('payload values'); ax[2].set_ylabel('active warps per SM'); ax[2].set_title('occupancy')
fig.tight_layout(); fig.savefig(os.path.join(O, 'fig6_payload.png'), dpi=140)
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for s, c in ((100000, '#1b6ca8'), (3000000, '#b03a2e')):
    d = json.load(open(os.path.join(R, 'notes/io/predict/predict_bounce_%d.json' % s)))
    ax.scatter([m for _, p, m in d['rows']], [p for _, p, m in d['rows']], color=c, s=18, label='%s triangles' % ('1e5' if s == 100000 else '3e6'))
ax.plot([0.5, 1.4], [0.5, 1.4], 'k-', lw=0.8); ax.plot([0.5, 1.4], [0.55, 1.54], 'k:', lw=0.6); ax.plot([0.5, 1.4], [0.45, 1.26], 'k:', lw=0.6)
ax.set_xlabel('measured speedup over pixel order (4070 Ti)'); ax.set_ylabel('predicted from the scene (replay + model fitted on the 6000 Ada)'); ax.legend(fontsize=7); ax.set_title('Nothing fitted on this chip', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(O, 'fig7_scene_prediction.png'), dpi=140)
print('wrote fig6, fig7')
