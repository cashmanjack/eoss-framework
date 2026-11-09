"""Drives memdump (Setup_4 fallback 3): one process per step, smallest first.

WHAT WORKED (2026-11-09, driver 580.178.04): GPU addresses are randomized per process, so
addresses from ncu (locate/verify/windows below) do not carry over to a plain run. Instead:
  memdump read --fn --method kernel --len 64 --no-optix        self test: our kernel's code, from
                                                               CUfunction+0x68 (host memory)
  memdump read --fnrel -0x4000 --method kernel --len 0x4000    16 KB below our kernel = the OptiX
                                                               pipeline (raygen + TTU code)
The steps below are kept for reference.

  python memdump_run.py self                 self test on our own code (no OptiX)
  python memdump_run.py locate               ncu run to learn raygen _ss_0 address, plus a plain run
  python memdump_run.py verify --method M    read raygen _ss_0 and compare to the OptiX disk cache bytes
  python memdump_run.py windows --method M [--per-side N] [--win BYTES]
                                             read windows outward from _ss_0, stop a side on first failure
  python memdump_run.py decode               decode saved windows with the EoSS SM89 Decoder, list TTU

Every GPU step: GPU 1 only, checks nobody else is on it, runs under `timeout -k 2 30`.
Outputs go to /dev/shm/memdump (delete after).
"""
import argparse, json, os, re, sqlite3, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, 'memdump')
OUT = '/dev/shm/memdump'
STATE = os.path.join(OUT, 'state.json')
CACHE = os.path.expanduser('/var/tmp/OptixCache_%s/optix7cache.db' % os.environ.get('USER', ''))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1',
           LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))


def gpu1_idle():
    r = subprocess.run(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == ''


def run(args, prefix=()):
    if not gpu1_idle():
        sys.exit('GPU 1 is in use by someone else; not running')
    cmd = ['timeout', '-k', '2', '30', *prefix, EXE, *args]
    print('$', ' '.join(cmd))
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    print(r.stdout + r.stderr, end='')
    print('exit', r.returncode)
    return r


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(s):
    json.dump(s, open(STATE, 'w'), indent=1)


def sass_bytes(binary, func, start, n):
    """Encoded bytes of n instructions from offset start of func, from cuobjdump."""
    txt = subprocess.run(['cuobjdump', '-sass', binary], capture_output=True, text=True).stdout
    body = txt.split('Function : ' + func, 1)[1].split('Function :', 1)[0]
    words = [int(w, 16) for w in re.findall(r'/\* (0x[0-9a-f]{16}) \*/', body)]
    out = b''
    for k in range(0, len(words) - 1, 2):
        out += struct.pack('<QQ', words[k], words[k + 1])
    return out[start:start + 16 * n]


def cache_text(name_part):  # name_part: tuple of substrings that must all appear
    """Bytes of the newest cached .text section whose name contains name_part."""
    db = sqlite3.connect('file:%s?mode=ro' % CACHE, uri=True)
    rows = db.execute('select * from cache_data').fetchall()
    best = None
    for row in rows:
        for blob in row:
            if not isinstance(blob, bytes): continue
            i = 0
            while (i := blob.find(b'\x7fELF\x02\x01\x01', i)) >= 0:
                b = i; i += 4
                try:
                    shoff, = struct.unpack_from('<Q', blob, b + 0x28)
                    shes, shn, shstr = struct.unpack_from('<HHH', blob, b + 0x3a)
                    hdr = lambda k: struct.unpack_from('<IIQQQQ', blob, b + shoff + k * shes)
                    so = b + hdr(shstr)[4]
                    for k in range(shn):
                        nm, typ, fl, a, off, sz = hdr(k)
                        name = blob[so + nm:blob.find(b'\0', so + nm)].decode('latin1')
                        if name.startswith('.text') and all(p in name for p in name_part):
                            best = (name, blob[b + off:b + off + sz])
                except struct.error:
                    pass
    return best


def cmd_self(a):
    os.makedirs(OUT, exist_ok=True)
    want = sass_bytes(EXE, '_Z6fnaddrPyPi', 0xf0, 2)
    print('expected probe_fn bytes:', want.hex())
    for m in ('attr', 'copy', 'kernel'):
        f = os.path.join(OUT, 'self_%s.bin' % m)
        r = run(['read', '--self', '--no-optix', '--method', m, '--len', '64', '--out', f])
        if m != 'attr' and r.returncode == 0:
            got = open(f, 'rb').read()[:32]
            print('%s: %s' % (m, 'MATCH' if got == want else 'differs: ' + got.hex()))


def cmd_locate(a):
    os.makedirs(OUT, exist_ok=True)
    rep = os.path.join(OUT, 'locate')
    r = run(['launch'], prefix=('ncu', '-c', '20', '--section', 'SourceCounters',
                                '-f', '-o', rep))
    m = re.search(r'probe_fn at (0x[0-9a-f]+)', r.stdout)
    csv = subprocess.run(['ncu', '--import', rep + '.ncu-rep', '--page', 'source', '--print-source', 'sass', '--csv'],
                         capture_output=True, text=True).stdout
    funcs, k = {}, None
    for line in csv.splitlines():
        if line.startswith('"Kernel Name"'):
            k = line.split('"')[3]
        elif line.startswith('"0x') and k:
            ad = int(line.split('"')[1], 16)
            lo, hi = funcs.get(k, (ad, ad))
            funcs[k] = (min(lo, ad), max(hi, ad + 16))
    for k, (lo, hi) in sorted(funcs.items(), key=lambda x: x[1]):
        print('  %#x %#x %s' % (lo, hi, k))
    r2 = run(['launch'])
    m2 = re.search(r'probe_fn at (0x[0-9a-f]+)', r2.stdout)
    s = load_state()
    s['funcs'] = {k: [hex(lo), hex(hi)] for k, (lo, hi) in funcs.items()}
    s['probe_ncu'] = m and m.group(1); s['probe_plain'] = m2 and m2.group(1)
    save_state(s)
    print('probe_fn under ncu %s, plain %s -> %s' % (s['probe_ncu'], s['probe_plain'],
          'same layout' if s['probe_ncu'] == s['probe_plain'] else 'LAYOUT DIFFERS, addresses from ncu may be wrong'))


def ss0(s):
    k = [k for k in s.get('funcs', {}) if k.startswith('__raygen__rg_') and k.endswith('_ss_0')]
    if not k: sys.exit('run locate first')
    return k[0], int(s['funcs'][k[0]][0], 16)


def cmd_verify(a):
    s = load_state(); name, ad = ss0(s)
    want = cache_text(('.text.' + name,))   # exact name: includes our module's hash
    if not want: sys.exit('no _ss_0 section in the OptiX cache')
    print('cache section', want[0], len(want[1]), 'bytes')
    f = os.path.join(OUT, 'verify.bin')
    r = run(['read', '--method', a.method, '--addr', hex(ad), '--len', '256', '--out', f])
    if r.returncode or a.method == 'attr': return
    got = open(f, 'rb').read(); ref = want[1][:256]
    same = sum(got[i:i + 16] == ref[i:i + 16] for i in range(0, min(len(got), len(ref)), 16))
    print('%d of %d instructions identical (relocated ones may differ)' % (same, min(len(got), len(ref)) // 16))
    s['verified'] = same >= 8; s['method'] = a.method; save_state(s)


def cmd_windows(a):
    s = load_state(); _, ad = ss0(s)
    if not s.get('verified'): sys.exit('run verify first (and it must match)')
    base = ad & ~(a.win - 1)
    for step in (+1, -1):
        for n in range(a.per_side):
            w = base + (n if step > 0 else -(n + 1)) * a.win
            f = os.path.join(OUT, 'win_%x.bin' % w)
            if os.path.exists(f): continue
            r = run(['read', '--method', s['method'], '--addr', hex(w), '--len', str(a.win), '--out', f])
            if r.returncode:
                print('stop this side at %#x' % w); break


def cmd_decode(a):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    from eoss.isa.model import ISASpec
    from eoss.tui.decode import Decoder
    dec = Decoder(ISASpec.load(sm=89))
    wins = sorted((int(f[4:-4], 16), f) for f in os.listdir(OUT) if f.startswith('win_'))
    for w, f in wins:
        data = open(os.path.join(OUT, f), 'rb').read()
        for i in range(0, len(data), 16):
            raw = data[i:i + 16]
            if (int.from_bytes(raw[:2], 'little') & 0xfff) in (0x3d0, 0x3d1, 0x3d2, 0x3d3, 0x9d4) or a.all:
                d = dec.decode(i, raw)
                print('%#x %s  %s %s  [%s] failed=%d' % (w + i, raw[::-1].hex(), d.mnemonic, d.operands,
                                                        d.defn.name if d.defn else '-', d.n_failed))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('self'); sp.add_parser('locate')
    v = sp.add_parser('verify'); v.add_argument('--method', default='kernel', choices=['attr', 'copy', 'kernel'])
    w = sp.add_parser('windows'); w.add_argument('--per-side', type=int, default=8); w.add_argument('--win', type=int, default=4096)
    d = sp.add_parser('decode'); d.add_argument('--all', action='store_true')
    a = p.parse_args(); globals()['cmd_' + a.cmd](a)
