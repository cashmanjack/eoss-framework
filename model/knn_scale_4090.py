"""Why the 4090 kNN launch times looked 1.8 to 2.1x shorter than 'triton scaled' (F255). CPU only, existing data.
The F138 kNN times (model/results_mva3.md header: 4070 Ti, io, 60 SMs, 48 MB L2) were NOT triton's; the Phase 10f prediction scaled them by 142/128.
Right basis: x 60/128. Then a two parameter check T4090 = a + b * T_io * 60/128 over the four rounds of each build."""
import math
io  = {'orig': [0.661, 0.979, 3.502, 11.016], 'reg': [0.165, 0.225, 0.692, 2.937]}   # F138 / results_mva3, 4070 Ti, ms
g   = {'orig': [0.4270, 0.6103, 2.161, 6.648], 'reg': [0.1147, 0.1331, 0.3891, 1.576]}  # 4090, results_phase10f, ms
out = ['# kNN on the 4090: the scaling basis (model/knn_scale_4090.py, existing data only)', '',
       'F138 kNN launch times were measured on the 4070 Ti (io, 60 SMs, 48 MB L2: model/results_mva3.md header, log queue 15 on io), not on triton. Phase 10f scaled them by 142/128 (wrong chip).', '',
       '| build round | 4070 Ti ms | x142/128 (used) | measured/that | x60/128 | measured/that |', '|---|---|---|---|---|---|']
for k in io:
    for r in range(4):
        a = io[k][r]; m = g[k][r]
        out.append('| %s %d | %.3f | %.3f | %.2f | %.3f | %.2f |' % (k, r + 1, a, a * 142 / 128, m / (a * 142 / 128), a * 60 / 128, m / (a * 60 / 128)))
out += ['', 'Two parameter fit over rounds 1 to 4 (least squares, T4090 = a + b x T_io x 60/128; b = 1 means perfect scaling with SM count):']
for k in io:
    x = [t * 60 / 128 for t in io[k]]; y = g[k]; n = 4; mx = sum(x) / n; my = sum(y) / n
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sum((xi - mx) ** 2 for xi in x); a = my - b * mx
    res = math.sqrt(sum(math.log((a + b * xi) / yi) ** 2 for xi, yi in zip(x, y)) / n)
    out.append('- %s: a = %.3f ms, b = %.2f (efficiency %.0f percent of SM count scaling), rms log residual %.3f' % (k, a, b, 100 / b, res))
lr = [math.log(g[k][r] / (io[k][r] * 60 / 128)) for k in io for r in range(4)]
out += ['', 'rms log error of the corrected basis (x60/128, no fit): %.2f (the 142/128 basis: %.2f). Not tested: the clocks on the 4070 Ti runs (not sampled in q15), so a clock difference stays open; DRAM bandwidth (504 vs 1,008 GB/s) and L2 (48 vs 72 MB) do not explain slower than scaled.' %
        (math.sqrt(sum(v * v for v in lr) / 8), math.sqrt(sum(math.log(g[k][r] / (io[k][r] * 142 / 128)) ** 2 for k in io for r in range(4)) / 8))]
open('model/results_knn_scale_4090.md', 'x').write('\n'.join(out) + '\n'); print('\n'.join(out))
