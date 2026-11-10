"""Plateau (b): same number of SMs spread across GPCs vs packed into few GPCs (chase_load, C chasers per SM).
GPC of SM s = (s // 2) % 12 (gpcmap.py). spread: one SM per TPC, TPCs taken in order (round robin over the 12 GPCs, SMs 0,2,4,...).
packed: all SMs of GPC 0 first (12 SMs), then GPC 1, ... """
import subprocess, sys
gpc = lambda s: (s // 2) % 12
spread = lambda g: [2 * b if b < 71 else 2 * (b - 71) + 1 for b in range(g)]
packed = lambda g: sorted(range(142), key=lambda s: (gpc(s), s))[:g]
C, N = int(sys.argv[1]), int(sys.argv[2])
for g in [6, 12, 24, 36, 60, 84]:
    for name, f in (('spread', spread), ('packed', packed)):
        sms = f(g)
        out = subprocess.run(['/dev/shm/chase_load', ','.join(map(str, sms)), str(C), str(N), '6'], capture_output=True, text=True).stdout.strip()
        print('%-6s gpcs %2d |' % (name, len({gpc(s) for s in sms})), out, flush=True)
