"""OptiX SM89 GAS reader (hypothesis of the format, see private notes): walk the tree and decode boxes.

Node = 128 bytes. +0x00: byte 0 flags, bytes 1..3 exponents ex, ey, ez; +0x04..+0x0c: float origin (box min);
+0x1c: offset in 128 byte units to the first internal child (internal children contiguous, slot order);
+0x20: 12 slots of 8 bytes: x lo, x hi, y lo, y hi, z lo, z hi, ?, type (0 internal, bit 7 leaf);
An empty slot is exactly z lo 0xff, z hi 0, byte 6 0, type 0 (a real child at the top of the z range
has z lo = z hi = 0xff; a flat child has z hi = z lo - 1). Child box = origin + [lo, hi + 1] * 2^(e - 135) per axis.
"""
import struct


def node_hdr(d, off):
    e = d[off + 1:off + 4]
    org = struct.unpack_from('<3f', d, off + 4)
    unit = [2.0 ** (x - 135) for x in e]
    return org, unit


def slots(d, off):
    org, unit = node_hdr(d, off)
    rel = struct.unpack_from('<i', d, off + 0x1c)[0]
    k = 0
    for s in range(12):
        b = d[off + 0x20 + 8 * s: off + 0x28 + 8 * s]
        if b[4] == 0xff and b[5] == 0 and b[6] == 0 and b[7] == 0:
            continue
        box = tuple((org[a] + b[2 * a] * unit[a], org[a] + (b[2 * a + 1] + 1) * unit[a]) for a in range(3))
        if b[7] == 0:
            yield s, 'node', off + 0x80 * (rel + k), box
            k += 1
        else:
            yield s, 'leaf', b[7], box


def walk(d):
    """All internal nodes: {off: dict(depth, path (slot numbers from the root), box, parent)}."""
    root = struct.unpack_from('<Q', d, 0x20)[0]
    out = {root: dict(depth=0, path=(), box=None, parent=None)}
    st = [root]
    while st:
        off = st.pop()
        for s, kind, child, box in slots(d, off):
            if kind == 'node':
                assert child not in out
                out[child] = dict(depth=out[off]['depth'] + 1, path=out[off]['path'] + (s,), box=box, parent=off)
                st.append(child)
    assert len(out) == struct.unpack_from('<I', d, 0x58)[0], 'walk does not reach the header node count'
    return out


def by_path(d, path):
    for off, n in walk(d).items():
        if n['path'] == tuple(path):
            return off
    raise KeyError(path)
