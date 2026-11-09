# Scans files for embedded SM cubins and counts TTU opcodes (low 12 bits) in .text sections.
# Usage: python3 -I ttuscan.py FILE...
import json, struct, sys, collections
d = json.load(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'encodings', 'ttu_sm89.json')))
ops = {}
for ins in d['sequences']['canonical_traversal']['instructions']:
    ops[int(ins['lo'], 16) & 0xfff] = ins['class']
print('opcodes', {hex(k): v for k, v in ops.items()})
OPEN = [k for k, v in ops.items() if v == 'ttuopen_'][0]
FUSE = [k for k, v in ops.items() if v == 'ttumacrofuse_'][0]

def cubins(blob):
    i = 0
    while True:
        i = blob.find(b'\x7fELF\x02\x01\x01', i)
        if i < 0: return
        try:
            mach, = struct.unpack_from('<H', blob, i + 0x12)
            if mach == 190:
                shoff, = struct.unpack_from('<Q', blob, i + 0x28)
                shes, shn, shstr = struct.unpack_from('<HHH', blob, i + 0x3a)
                yield i, shoff, shes, shn, shstr
        except struct.error: pass
        i += 4

def sections(blob, base, shoff, shes, shn, shstr):
    hdr = lambda k: struct.unpack_from('<IIQQQQ', blob, base + shoff + k * shes)
    sn = hdr(shstr); stroff = base + sn[4]
    for k in range(shn):
        name, typ, fl, addr, off, sz = hdr(k)
        e = blob.find(b'\0', stroff + name)
        yield blob[stroff + name:e].decode('latin1'), typ, base + off, sz

for path in sys.argv[1:]:
    blob = open(path, 'rb').read()
    n_cubin = 0; arch = collections.Counter(); hits = collections.Counter(); seqs = []
    for base, shoff, shes, shn, shstr in cubins(blob):
        n_cubin += 1
        flags, = struct.unpack_from('<I', blob, base + 0x30)
        # ELF ABI v8 (OSABI 0x41, CUDA 12.x+) keeps the SM in e_flags bits 15:8; older cubins in bits 7:0
        sm = (flags >> 8) & 0xff if blob[base + 8] >= 8 else flags & 0xff
        arch[sm] += 1
        try:
            for name, typ, off, sz in sections(blob, base, shoff, shes, shn, shstr):
                if not name.startswith('.text'): continue
                words = [struct.unpack_from('<QQ', blob, off + j) for j in range(0, sz - 15, 16)]
                for j, (lo, hi) in enumerate(words):
                    op = lo & 0xfff
                    if op in ops: hits[(sm, ops[op])] += 1
                    if op == OPEN and any((w[0] & 0xfff) == FUSE for w in words[j+1:j+8]):
                        seqs.append((base, sm, name, off + 16 * j))
        except Exception as ex:
            pass
    print(f'\n== {path}: {n_cubin} cubins, sm counts {dict(arch)}')
    print('TTU opcode hits by (sm, class):', dict(hits))
    print('TTUOPEN followed by TTUMACROFUSE within 8:', len(seqs))
    for s in seqs[:10]: print('   cubin@%#x sm%d %s word@%#x' % s)
