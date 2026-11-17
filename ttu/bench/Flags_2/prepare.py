"""Flags_2 PREPARE ONLY (predictions: model/predictions_flags2.md). Our own Setup_3 TTU sequence (the verbatim OptiX group, P0 true path) with the store words of other ray flags / masks.

    prepare.py OUTDIR            builds OUTDIR/<variant>/{setup3_s0, setup3_s1, setup3_dev.ptx, words.txt}, runs check_seq on every binary, deletes the .checked marker of every STAGE 1 binary
                                 (so run.py refuses it until check_seq is run again on purpose), and prints the sha256 of each.
    prepare.py OUTDIR stage0     (after prepare) runs the stage 0 binary of ONE variant (no TTU instruction) for the rays hit and back.

The sequence is identical for every variant (same bytes, same N 17 group); only the 8 words of TTUST 0 and 0x80 (words.txt, read by the host) change.
Words: notes/flagenc/store_words_by_variant.txt (private, computed by notes/setup3/store_words.py, F183). Excluded: TERM+DISCH (the P0 false path: different predicated stores) and cull both (invalid configuration).
Needs /dev/shm/t9/setup3 (host) and /dev/shm/t9/setup3_dev.ptx. No GPU is used by 'prepare'.
"""
import os, shutil, subprocess, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
VENV = os.path.expanduser('~/eoss/.venv/bin/python'); BASE = '/dev/shm/t9/setup3'; PTX = '/dev/shm/t9/setup3_dev.ptx'
WORDS = os.path.join(HERE, '..', '..', '..', 'notes', 'flagenc', 'store_words_by_variant.txt')
# order of the future stage 1 runs: control first
VARIANTS = ['none', 'TERMINATE', 'CULL_BACK', 'CULL_FRONT', 'CULL_OPAQUE', 'CULL_NON_OPAQUE', 'DISABLE_ANYHIT', 'ENFORCE_ANYHIT', 'mask15', 'mask1', 'mask128', 'mask0']


def words_of(name):
    for l in open(WORDS):
        t = l.split()
        if t[0] == name:
            assert t[1] == 'True', name   # P0 true path only
            w = [x for x in t[2:] if x != '|']; assert len(w) == 8, (name, w); return w
    sys.exit('no words for ' + name)


def sh(c): return subprocess.run(c, capture_output=True, text=True)


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == 'stage0':
        d = os.path.join(out, 'none')   # layout is the same for every variant: one stage 0 per ray
        for ray in ('hit', 'back'):
            r = sh([sys.executable, os.path.join(S3, 'run.py'), d + '/setup3_s0', ray, 'zero', d + '/stage0_%s_%s.json' % (ray, sys.argv[3] if len(sys.argv) > 3 else 'x')])
            print(ray, 'stage 0:', r.stdout[-600:].strip().replace('\n', ' | '), r.stderr[-300:]); 
            if r.returncode: sys.exit('STOP')
        sys.exit(0)
    for v in VARIANTS:
        d = os.path.join(out, v); os.makedirs(d, exist_ok=True)
        w = words_of(v); open(d + '/words.txt', 'w').write(' '.join(w) + '\n'); shutil.copy(PTX, d + '/setup3_dev.ptx')
        for stage in (0, 1):
            b = '%s/setup3_s%d' % (d, stage)
            p = sh([VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--out', b])
            if p.returncode: sys.exit(p.stdout + p.stderr)
            c = sh([VENV, os.path.join(S3, 'check_seq.py'), b]); last = c.stdout.strip().splitlines()[-1]
            if 'ALL PASS' not in last: sys.exit('check_seq FAILED %s\n%s' % (b, c.stdout[-2000:]))
            sha = hashlib.sha256(open(b, 'rb').read()).hexdigest()
            if stage == 1: os.remove(b + '.checked')
            print('%-16s stage %d check_seq ALL PASS sha256 %s %s' % (v, stage, sha[:16], '(marker deleted)' if stage == 1 else ''), flush=True)
