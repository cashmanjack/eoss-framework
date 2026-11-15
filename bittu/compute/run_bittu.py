"""Run a checked setup3 binary under BITTU compute mode (run.py does the safety checks: check_seq marker, idle GPU, timeout).
    run_bittu.py BIN RAYS FILL OUT.json REPORT.txt [run.py options ...]       e.g. run_bittu.py /dev/shm/t9r/g_s1c hit zero out.json report.txt
Cuts the group's bytes out of BIN (layout json: group_index, group_len) into /dev/shm/bittu, sets BITTU_GROUP_FILE / OFF / LEN / OUT and calls run.py --inject bittu.so.
The group bytes are NVIDIA code: they stay in /dev/shm and are deleted afterwards."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
from eoss.binary.patcher import BinaryPatcher
from eoss.binary import elf


def main():
    binp, rays, fill, out, report = sys.argv[1:6]; extra = sys.argv[6:]
    like = os.environ.get('BITTU_LIKE')   # empty probe pair: BIN is the stage 0 twin, BITTU_LIKE the stage 1 binary whose group bytes give the register lists (same layout)
    mode = {'BITTU_MODE': 'empty'} if like else {}
    src = like or binp
    meta = json.load(open(src + '.layout.json')); p = BinaryPatcher(src)
    sec = [s for s in p.sections if s[2] == meta['section']][0]; body = [b for _, b in elf.iter_instructions(p.data, sec)]
    g, n = meta['group_index'], meta['group_len']; gf = '/dev/shm/bittu/group_%s.bin' % meta['sha256'][:8]
    os.makedirs('/dev/shm/bittu', exist_ok=True); open(gf, 'wb').write(b''.join(body[g:g + n]))
    env = dict(os.environ, **mode, BITTU_GROUP_FILE=gf, BITTU_GROUP_OFF=hex(16 * g), BITTU_GROUP_LEN=str(n), BITTU_OUT=report)
    cmd = [sys.executable if False else '/usr/bin/python3', os.path.join(ROOT, 'ttu', 'harness', 'setup3', 'run.py'), binp, rays, fill, out, '--inject', '/dev/shm/bittu/bittu.so'] + extra
    try:
        sys.exit(subprocess.run(cmd, env=env).returncode)
    finally:
        os.remove(gf)


if __name__ == '__main__':
    main()
