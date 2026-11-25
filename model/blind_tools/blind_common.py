"""Shared helpers for the blind test tools (model v1, model/FROZEN.md, model/blind/RULE.md). CPU and git only; no GPU code here."""
import os, re, sys, json, hashlib, subprocess
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SM_TABLE = {'NVIDIA RTX 6000 Ada Generation': 142, 'NVIDIA GeForce RTX 4090': 128, 'NVIDIA GeForce RTX 4070 Ti': 60}   # chip specification; the only GPU dependent quantity (FROZEN.md)
CLK = 2.5e9                                                                                                              # frozen constant (FROZEN.md), never changed

def sh(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, **kw)
def sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()[:12]          # 12 hex digits: the push scan refuses longer hex runs
def short_head(): return sh(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD']).stdout.strip()

def verify_frozen():
    """Every tracked file hashed in the FIRST code block of model/FROZEN.md must still have that hash (first 12 hex digits of sha256). Returns the list checked."""
    txt = open(os.path.join(REPO, 'model', 'FROZEN.md')).read()
    block = txt.split('```')[1]; checked = []
    for line in block.strip().splitlines():
        m = re.fullmatch(r'([0-9a-f]{12})  (\S+)', line.strip())
        if not m: continue
        want, rel = m.groups(); got = sha12(os.path.join(REPO, rel))
        if got != want: raise SystemExit('FROZEN file changed: %s (recorded %s, now %s). The blind test is void until a new freeze is committed.' % (rel, want, got))
        checked.append(rel)
    if not checked: raise SystemExit('no frozen hashes found in model/FROZEN.md')
    return checked

def sm_for(gpu_index, override=None):
    if override: return int(override)
    name = sh(['nvidia-smi', '-i', str(gpu_index), '--query-gpu=name', '--format=csv,noheader']).stdout.strip()
    if name not in SM_TABLE: raise SystemExit('GPU %r has no SM count in the frozen table (FROZEN.md); pass --sm explicitly' % name)
    return SM_TABLE[name]

def committed_and_pushed(relpath):
    """True if relpath is tracked, unmodified, and its last commit is contained in origin/main."""
    if sh(['git', '-C', REPO, 'ls-files', '--error-unmatch', relpath]).returncode != 0: return False, 'not tracked'
    if sh(['git', '-C', REPO, 'diff', '--quiet', 'HEAD', '--', relpath]).returncode != 0: return False, 'modified since its commit'
    c = sh(['git', '-C', REPO, 'log', '-1', '--format=%H', '--', relpath]).stdout.strip()
    if sh(['git', '-C', REPO, 'merge-base', '--is-ancestor', c, 'origin/main']).returncode != 0: return False, 'commit %s not in origin/main' % c[:7]
    return True, c[:7]

def commit(paths, msg, push=True):
    rel = [os.path.relpath(p, REPO) for p in paths]
    sh(['git', '-C', REPO, 'add'] + rel)
    r = sh(['git', '-C', REPO, 'commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>', '--'] + rel)
    if r.returncode != 0: raise SystemExit('commit failed: ' + r.stdout + r.stderr)
    h = short_head()
    if push:
        r = sh([os.path.join(REPO, 'notes', 'push.sh')]); tail = (r.stdout + r.stderr).strip().splitlines()
        print('push:', tail[-1] if tail else 'no output', flush=True)
        if r.returncode != 0: raise SystemExit('push failed (scan not clean?): ' + '\n'.join(tail[-5:]))
    return h

def six(x):
    """6 significant digits (the push scan flags long digit runs)."""
    return float('%.6g' % x) if isinstance(x, float) else x
