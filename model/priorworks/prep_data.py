"""Data preparation for the prior works comparison (2026-11-17). Writes text files to /dev/shm/pw/data (rebuildable, outside git).
Sources are in /local/scratch/a/cashman3/rtdata. Usage: python3 prep_data.py NAME  (dragon bunny road porto uni1m gowalla manuscript glove3d)"""
import sys, os, gzip, io, zipfile, struct, numpy as np
R = '/local/scratch/a/cashman3/rtdata/'; O = '/dev/shm/pw/data/'; os.makedirs(O, exist_ok=True)
def ply_be(path, nvert_prop_bytes):
    with gzip.open(path, 'rb') as f:
        h = b''
        while not h.endswith(b'end_header\n'): h += f.readline()
        n = int([l for l in h.decode().split('\n') if l.startswith('element vertex')][0].split()[2])
        a = np.frombuffer(f.read(n * nvert_prop_bytes), dtype=np.uint8).reshape(n, nvert_prop_bytes)
    return np.ascontiguousarray(a[:, :12]).view('>f4').reshape(n, 3).astype(np.float32)
def save(name, p, sep):
    np.savetxt(O + name, p, fmt='%.9g', delimiter=sep); print(name, p.shape, p.min(0), p.max(0))
def quat(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
n = sys.argv[1]
if n == 'dragon': save('dragon.csv', ply_be(R + 'stanford/xyzrgb_dragon.ply.gz', 12), ',')
elif n == 'manuscript':
    p = ply_be(R + 'manuscript/xyzrgb_manuscript.ply.gz', 15); save('manuscript.txt', p, ' ')
elif n == 'bunny':
    import tarfile, re; d = '/dev/shm/pw/bunny/bunny/data/'; pts = []
    for l in open(d + 'bun.conf'):
        t = l.split()
        if not t or t[0] != 'bmesh': continue
        v = []; fp = open(d + t[1]); st = False; nv = 0
        for ln in fp:
            if ln.startswith('element vertex'): nv = int(ln.split()[2])
            if ln.startswith('end_header'): break
        for _ in range(nv): v.append([float(x) for x in fp.readline().split()[:3]])
        v = np.array(v); tr = np.array(t[2:5], float); q = [float(x) for x in t[5:9]]
        pts.append(v @ quat(q).T + tr)
    save('bunny.csv', np.vstack(pts).astype(np.float32), ',')
elif n == 'road': os.system('cp %s3droad/3droad_full.csv %s3droad.csv' % (R, O)); print('road copied')
elif n == 'uni1m': save('uni1m.csv', np.random.default_rng(11).random((1000000, 3)).astype(np.float32), ',')
elif n == 'porto':
    import json, csv
    z = zipfile.ZipFile(R + 'porto/porto.zip'); inner = zipfile.ZipFile(io.BytesIO(z.read('train.csv.zip'))); f = io.TextIOWrapper(inner.open(inner.namelist()[0]))
    rd = csv.reader(f); next(rd); pts = []; tot = 0
    for row in rd:
        for lon, lat in json.loads(row[-1]): pts.append((lon, lat, 0.0)); tot += 1
        if tot >= 1000000: break
    save('porto1m.csv', np.array(pts[:1000000], np.float64), ',')
elif n == 'gowalla':
    s = set()
    for l in gzip.open(R + 'gowalla/loc-gowalla_totalCheckins.txt.gz', 'rt'):
        t = l.split('\t'); s.add((t[2], t[3]))   # lat, lon as text; location id is t[4]
    p = np.array([(float(a), float(b), 0.0) for a, b in s], np.float64); print('unique (lat,lon):', len(p)); save('gowalla.txt', p, ' ')
elif n == 'glove3d':
    z = zipfile.ZipFile(R + 'glove/glove.twitter.27B.zip'); rows = []
    for l in io.TextIOWrapper(z.open('glove.twitter.27B.25d.txt'), encoding='utf8'):
        t = l.rstrip('\n').rsplit(' ', 25)
        if len(t) == 26: rows.append([float(x) for x in t[1:]])
    a = np.array(rows, np.float64); print('vectors', a.shape); a -= a.mean(0); u, s, vt = np.linalg.svd(a, full_matrices=False) if a.shape[0] < 5000 else (None, None, np.linalg.svd(a.T @ a)[2])
    save('glove3d.txt', (a @ vt[:3].T).astype(np.float32), ' ')
