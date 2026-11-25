"""Baseline 6: Mochi (Mandarapu, James, Kulkarni; DCD on RT cores), UNPATCHED build ~/Mochi-DCD-on-RT/build/sample23-3Dparticlesim (RENDER 0, COUNT_COLLISIONS 0, NUM_PARTICLES 100000, sm_89).
usage: python base_mochi.py [REPS]   -> notes/base1/mochi_<ts>.json. Change no code. (The div1 copy carries our lane histogram patches; the home build does not.)"""
import re, sys, base_common as B
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BD = B.HOME + '/Mochi-DCD-on-RT/build'; BIN = BD + '/sample23-3Dparticlesim'; DATA = B.HOME + '/Mochi-DCD-on-RT/datasets/100k_particles.txt'
def parse(t):
    L = [l.strip() for l in t.splitlines() if l.strip()]
    p = {}
    try:
        i = L.index('100000'); v = [float(x) for x in L[i + 5:i + 9]]   # after: particles, frames, gravity, restitution, bounce coef -> build, cd, update, total (ms)
        p = dict(build_ms=v[0], collision_ms=v[1], update_ms=v[2], total_ms=v[3])
    except (ValueError, IndexError): pass
    p['frames_done'] = len(re.findall(r'frame \d+ done', t))
    return p
doc = B.header('Mochi', BIN, [DATA], '100,000 particles (100k_particles.txt), 10 frames. Timers in the program: build (BVH build and refit, summed over frames, the first build included), collision detection plus response (the launches, host timer around a synchronised launch), update (summed). Totals are the program\'s own host timers in milliseconds.')
cmd = [BIN, DATA, '10']; runs = []
for i in range(REPS):
    x = B.run(cmd, BD, B.env(), 300); x['parsed'] = parse(x['stdout']); runs.append(x)
doc['configs'] = [dict(name='100k_10frames', cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summarize(runs, ['build_ms', 'collision_ms', 'update_ms', 'total_ms', 'frames_done']) | {'wall_s': B.stats([x['wall_s'] for x in runs if x['rc'] == 0])})]
print(B.write('mochi', doc))
