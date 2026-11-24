"""Print the TTU answer words of a rehosting raw json (lane 0, the words of the six TTULDs and the pending entries) and the lane / launch agreement."""
import json, struct, sys
f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
j = json.load(open(sys.argv[1]))
o, i = j['out'], j['in']
names = {32: '0x300 Rd2.lo u|face', 33: '0x300 Rd2.hi v|TAG', 44: '0x300 Rd.lo prim|type', 45: '0x300 Rd.hi -t', 68: '0x310 Rd2.lo', 69: '0x310 Rd2.hi', 30: '0x310 Rd.lo leaf lo', 31: '0x310 Rd.hi leaf hi|slot',
         2: '0x320 Rd.lo', 3: '0x320 Rd.hi hit flag', 62: '0x380 Rd2.lo', 63: '0x380 Rd2.hi', 34: '0x380 Rd.lo inst', 35: '0x380 Rd.hi inst', 58: '0x3C0 Rd2.lo', 59: '0x3C0 Rd2.hi', 56: '0x3C0 Rd.lo', 57: '0x3C0 Rd.hi',
         70: '0x3D0 Rd2.lo', 71: '0x3D0 Rd2.hi', 60: '0x3D0 Rd.lo', 61: '0x3D0 Rd.hi'}
print('ray', j['ray_name'], j['ray'], '| status', j['ttu_status'], '| gas', j['gas'], 'size', j['gas_size'], 'root', j['root'])
for r in sorted(names, key=lambda r: (names[r][:5], r)):
    ch = '' if o[0][r] == i[0][r] else 'CHANGED'
    print('  R%-3d %-26s in %08x out %08x %s' % (r, names[r], i[0][r], o[0][r], ch))
lanes = all(o[k][r] == o[0][r] for k in range(32) for r in names)
print('all 32 lanes identical (the 22 answer registers):', lanes, '| launches same:', all(l.get('same_as_first', True) for l in j.get('launches', [])), '| lanes_equal flags:', [l.get('lanes_equal') for l in j.get('launches', [])])
for L in j.get('launches', []): print('  launch ms', L.get('ms'), 'warp_clk', L.get('warp_clk', L.get('per_block', [{}])[0].get('warp_clk')))
