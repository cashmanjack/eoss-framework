"""CPU reference: Moller Trumbore ray/triangle in float64, same triangle as setup3.cu."""
TRI = ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0))


def sub(a, b): return [a[i] - b[i] for i in range(3)]
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def intersect(o, d, tmin=0.0, tmax=1e16, tri=TRI):
    """Returns (hit, t, u, v); u, v are the barycentrics of vertices 1 and 2 (OptiX convention)."""
    e1, e2 = sub(tri[1], tri[0]), sub(tri[2], tri[0])
    p = cross(d, e2); det = dot(e1, p)
    if abs(det) < 1e-12: return (False, None, None, None)
    s = sub(o, tri[0]); u = dot(s, p) / det
    q = cross(s, e1); v = dot(d, q) / det; t = dot(e2, q) / det
    hit = u >= 0 and v >= 0 and u + v <= 1 and tmin <= t <= tmax
    return (hit, t, u, v) if hit else (False, None, None, None)


f32 = lambda x: __import__('struct').unpack('<f', __import__('struct').pack('<f', x))[0]   # setup3.cu uses float
RAYS = {k: (tuple(map(f32, o)), tuple(map(f32, d))) for k, (o, d) in {   # keep in sync with setup3.cu
    'hit': ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), 'miss': ((2.0, 2.0, -1.0), (0.0, 0.0, 1.0)),
    't2': ((0.0, 0.0, -2.0), (0.0, 0.0, 1.0)), 'back': ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
    'moved': ((0.2, -0.3, -1.0), (0.0, 0.0, 1.0)), 'moved2': ((-0.3, -0.4, -1.0), (0.0, 0.0, 1.0))}.items()}

if __name__ == '__main__':
    for k, (o, d) in RAYS.items():
        print(k, intersect(o, d))
