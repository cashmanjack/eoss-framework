"""BITTU on a rehosted kernel (R5 step): cuts the group bytes out of BUILD (layout json: group_index, group_len) into /dev/shm/bittu, sets BITTU_* and calls rtnn_run.py --inject.
    run_bittu_rehost.py BUILD BITTU.so REPORT.txt LABEL [rtnn_run host options]
The group bytes are the encoder's (identical to the checked group): they stay in /dev/shm and are deleted afterwards."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
from eoss.binary.patcher import BinaryPatcher
from eoss.binary import elf


def main():
    binp, so, report, label = sys.argv[1:5]; extra = sys.argv[5:]
    meta = json.load(open(binp + '.layout.json')); p = BinaryPatcher(binp)
    sec = [s for s in p.sections if s[2] == meta['section']][0]; body = [b for _, b in elf.iter_instructions(p.data, sec)]
    g, n = meta['group_index'], meta['group_len']; gf = '/dev/shm/bittu/group_%s.bin' % meta['sha256'][:8]
    os.makedirs('/dev/shm/bittu', exist_ok=True); open(gf, 'wb').write(b''.join(body[g:g + n]))
    env = dict(os.environ, BITTU_GROUP_FILE=gf, BITTU_GROUP_OFF=hex(16 * g), BITTU_GROUP_LEN=str(n), BITTU_OUT=report)
    cmd = ['/usr/bin/python3', os.path.join(HERE, 'rtnn_run.py'), binp, os.path.join(ROOT, 'notes', 'rehost'), label] + extra + ['--inject', so]
    try:
        sys.exit(subprocess.run(cmd, env=env).returncode)
    finally:
        os.remove(gf)


if __name__ == '__main__':
    main()
