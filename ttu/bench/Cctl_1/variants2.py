"""Build the edit variants of catalogue items 1 to 7 from the checked base builds in /dev/shm/cctl1 (cctl_s1, twice_s1) and register their sha256 in the
private notes/cctl_variant_sha.json (key 'edits'). The edits themselves are defined in check_seq.py (variant_edits), the single source.
    variants2.py build      CPU only
"""
import hashlib, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..')); S3 = os.path.join(ROOT, 'ttu', 'harness', 'setup3')
sys.path.insert(0, ROOT); sys.path.insert(0, S3)
import check_seq as cs
from eoss.binary.patcher import BinaryPatcher
from eoss.binary import elf
D = '/dev/shm/cctl1'; PJ = os.path.join(ROOT, 'notes', 'cctl_variant_sha.json')
V = {'F0_fuse0': 'cctl', 'P1': 'cctl', 'P2': 'cctl', 'P3': 'cctl', 'O1': 'cctl', 'O2': 'cctl', 'O3': 'cctl', 'O4': 'cctl', 'W1': 'cctl', 'W2': 'cctl',
     'S1': 'twice', 'S2': 'twice', 'X1': 'twice', 'D1': 'twice'}
def build():
    j = json.load(open(PJ)); j.setdefault('edits', {})
    for n, b in V.items():
        base = '%s/%s_s1' % (D, b); bd = open(base, 'rb').read(); meta = json.load(open(base + '.layout.json'))
        p = BinaryPatcher(base); sec = [s for s in p.sections if s[2] == meta['section']][0]; ins = list(elf.iter_instructions(p.data, sec)); body = [x for _, x in ins]
        e = cs.variant_edits(n, body); assert e, n
        out = bytearray(bd)
        for i, nb in e.items(): out[ins[i][0]:ins[i][0] + 16] = nb
        fn = '%s/%s_s1' % (D, n)
        with open(fn, 'wb') as f: f.write(out)
        os.chmod(fn, 0o755); shutil.copy(base + '.layout.json', fn + '.layout.json')
        if os.path.exists(fn + '.checked'): os.remove(fn + '.checked')
        sha = hashlib.sha256(bytes(out)).hexdigest(); j['edits'][sha] = {'name': n, 'base_file': base, 'base_sha': hashlib.sha256(bd).hexdigest()}
        print('%-9s base %-5s instructions edited %s sha %s' % (n, b, sorted(e), sha[:8]))
    json.dump(j, open(PJ, 'w'), indent=1)
if __name__ == '__main__': build()
