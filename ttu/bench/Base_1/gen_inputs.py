"""Regenerate the baseline inputs from the recorded seeds (original recipes), into ~/thirdparty/base_data. Refuses to overwrite.
- u200000.csv: numpy default_rng(5).random((200000,3)), '%.9g' (the kNN input of Phase 10f, log 2026-11-24 'u2e5 seed 5')
- ark2e5.txt: default_rng(11).random((200000,2)), '%.7f' (Arkade 100,000 points + 100,000 queries; seed 11 as ark_a1e5.txt, whose first 110,000 lines match this stream: checked)
- bh_g100k.txt, bh_g1M.txt: Gaussian disc (Jack's generate_synthetic1M.py recipe, seed 42), header N,1,0.025,0.05,0.5 as printed by the earlier runs (q12/bh/g100k_1.log)."""
import os, numpy as np
T = os.path.expanduser('~/thirdparty/base_data'); os.makedirs(T, exist_ok=True)
def sv(name, arr, fmt):
    with open(T + '/' + name, 'x') as f: np.savetxt(f, arr, fmt=fmt, delimiter=',')
sv('u200000.csv', np.random.default_rng(5).random((200000, 3)), '%.9g')
sv('ark2e5.txt', np.random.default_rng(11).random((200000, 2)), '%.7f')
for n, nm in ((100000, 'bh_g100k.txt'), (1000000, 'bh_g1M.txt')):
    np.random.seed(42)
    x = np.random.normal(0, 100, n).astype(np.float32); y = np.random.normal(0, 100, n).astype(np.float32); z = np.random.normal(0, 20, n).astype(np.float32)
    m = np.random.uniform(1.0, 100.0, n).astype(np.float32)
    with open(T + '/' + nm.replace('bh_', 'bh2_'), 'x') as f:
        f.write('%d\n1\n0.025\n0.05\n0.5\n' % n)
        for i in range(n): f.write('%f %f %f %f 0.000000 0.000000 0.000000\n' % (m[i], x[i], y[i], z[i]))
