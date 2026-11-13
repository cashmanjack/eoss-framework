"""One config per GPU, detected at startup (no per machine constants in the harness).

    python gpucfg.py [--gpu N] [--json]     print the config of the chosen (or first idle) GPU
    import gpucfg; c = gpucfg.load(); c.env(); c.sm_of_block(b)

Detected (libcuda through ctypes, nvidia-smi): name, compute capability, SM count, L2 bytes, max SM and
memory clock, driver version, display processes. Known per architecture (profiles.json, each entry says
how it was verified): the TPC pairing and block placement rule, the CUfunction code address offset per
driver branch, the SM89 encodings file. OptiX library: the system one if its version equals the driver,
else the copy extracted to ~/opt/nvoptix-<driver>. The harness refuses anything it has no profile for.
"""
import ctypes, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATTR = {'sm_count': 16, 'l2_bytes': 38, 'clock_khz': 13, 'mem_clock_khz': 36, 'bus_bits': 37, 'cc_major': 75, 'cc_minor': 76}


def _smi(*a):
    return subprocess.run(['nvidia-smi'] + list(a), capture_output=True, text=True).stdout.strip()


def _cuda_attrs(index):
    """Attributes of the device with this PCI order index (CUDA_DEVICE_ORDER=PCI_BUS_ID), via the driver API."""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    cu = ctypes.CDLL('libcuda.so.1')
    assert cu.cuInit(0) == 0
    d = ctypes.c_int()
    assert cu.cuDeviceGet(ctypes.byref(d), index) == 0
    out = {}
    for k, a in ATTR.items():
        v = ctypes.c_int()
        assert cu.cuDeviceGetAttribute(ctypes.byref(v), a, d) == 0
        out[k] = v.value
    return out


def gpus():
    rows = _smi('--query-gpu=index,name,pci.bus_id,driver_version,clocks.max.sm,clocks.max.mem,memory.total', '--format=csv,noheader,nounits').splitlines()
    out = []
    for r in rows:
        i, name, bus, drv, csm, cmem, mem = [x.strip() for x in r.split(',')]
        out.append(dict(index=int(i), name=name, bus=bus, driver=drv, max_sm_mhz=int(csm), max_mem_mhz=int(cmem), mem_mib=int(mem)))
    return out


def users(index):
    """(compute process pids, graphics process names) on this GPU."""
    bus = [g['bus'] for g in gpus() if g['index'] == index][0]
    comp = [x for x in _smi('-i', str(index), '--query-compute-apps=pid', '--format=csv,noheader').splitlines() if x.strip()]
    gfx = []
    for ln in _smi('-i', str(index)).splitlines():   # the process table: type G (graphics) rows
        t = ln.split()
        if len(t) >= 6 and t[0] == '|' and t[1] == str(index) and t[4] in ('G', 'C+G'):
            gfx.append(t[6] if len(t) > 6 else '?')
    return comp, gfx


def pick_gpu():
    """TTU_GPU if set, else the GPU with no compute process, utilization 0 and the fewest graphics clients."""
    if os.environ.get('TTU_GPU'): return int(os.environ['TTU_GPU'])
    best = None
    for g in gpus():
        comp, gfx = users(g['index'])
        util = int(_smi('-i', str(g['index']), '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits') or 0)
        if comp or util > 10: continue   # a few percent can be left over from our own previous run
        key = (len(gfx), g['index'])
        if best is None or key < best[0]: best = (key, g['index'])
    if best is None: sys.exit('no idle GPU')
    return best[1]


class Config:
    def __init__(self, index):
        info = [g for g in gpus() if g['index'] == index][0]
        at = _cuda_attrs(index)
        prof = json.load(open(os.path.join(HERE, 'profiles.json')))
        self.index, self.name, self.driver, self.bus = index, info['name'], info['driver'], info['bus']
        self.sm_count, self.l2_bytes = at['sm_count'], at['l2_bytes']
        self.sm = 'sm%d%d' % (at['cc_major'], at['cc_minor'])
        self.max_sm_mhz, self.max_mem_mhz = info['max_sm_mhz'], info['max_mem_mhz']
        self.mem_bus_bits = at['bus_bits']
        p = prof['arch'].get(self.sm)
        if p is None: sys.exit('no TTU profile for %s (%s): refusing' % (self.sm, self.name))
        self.profile = p
        self.tpc_count = self.sm_count // p['sms_per_tpc']
        self.half = self.sm_count // 2
        self.cufunction_offset = int(prof['cufunction_offset'].get(self.driver.split('.')[0], '0'), 16) or None
        self.driver_validated = self.driver in prof['validated_drivers']
        self.encodings = os.path.normpath(os.path.join(HERE, '..', 'encodings', p['encodings']))
        self.optix_lib = self._optix_lib()
        if self.sm_count % 2: sys.exit('odd SM count: placement rule unknown')

    def _optix_lib(self):
        """Directory to put on LD_LIBRARY_PATH, or '' when the system libnvoptix matches the driver."""
        for d in ('/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/usr/lib64'):
            if os.path.exists(os.path.join(d, 'libnvoptix.so.' + self.driver)): return ''
        d = os.path.expanduser('~/opt/nvoptix-' + self.driver)
        if os.path.exists(os.path.join(d, 'libnvoptix.so.1')): return d
        sys.exit('no OptiX library for driver %s (extract it to %s)' % (self.driver, d))

    def sm_of_block(self, b):
        """SM a block lands on when one block per SM is launched (the first half of the blocks on the even SMs, then the odd ones)."""
        return 2 * b if b < self.half else 2 * (b - self.half) + 1

    def env(self, base=None):
        e = dict(base or os.environ, CUDA_VISIBLE_DEVICES=str(self.index), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        if self.optix_lib: e['LD_LIBRARY_PATH'] = self.optix_lib + ':' + e.get('LD_LIBRARY_PATH', '')
        return e

    def check_idle(self):
        comp, gfx = users(self.index)
        if comp: sys.exit('refused: GPU %d busy: %s' % (self.index, comp))
        if not self.driver_validated: sys.exit('refused: driver %s has no validated TTU group' % self.driver)
        return gfx

    def dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'profile'} | {'profile': self.profile}


def load(index=None):
    return Config(pick_gpu() if index is None else index)


if __name__ == '__main__':
    a = sys.argv[1:]
    c = load(int(a[a.index('--gpu') + 1]) if '--gpu' in a else None)
    print(json.dumps(c.dict(), indent=1))
