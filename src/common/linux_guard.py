"""OS adapter for the unchanged native C5; every guard reads live RAM values."""
import ctypes as C
import gc,os
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
LIB=C.CDLL(str(ROOT/'linux_ram.so'),use_errno=True)
LIB.ram_init.argtypes=[C.POINTER(C.c_char_p),C.c_int];LIB.ram_init.restype=C.c_int
LIB.ram_snapshot.argtypes=[C.POINTER(C.c_uint64)];LIB.ram_snapshot.restype=C.c_int

def groups():
    mount=Path('/sys/fs/cgroup')
    paths={mount}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            candidate=mount/line[3:].lstrip('/')
            if candidate.is_dir() and mount in candidate.parents:
                paths.add(candidate)
                paths.update(p for p in candidate.parents if p==mount or mount in p.parents)
    paths=sorted(p for p in paths if (p/'memory.max').exists())
    if not paths:raise RuntimeError('Expected cgroup v2 memory constraints are unavailable')
    return paths
PATHS=groups()
arguments=(C.c_char_p*len(PATHS))(*(str(p).encode() for p in PATHS))
error=LIB.ram_init(arguments,len(PATHS))
if error:raise OSError(error,'Cannot initialize fresh native RAM queries')

def snapshot():
    result=(C.c_uint64*3)()
    code=LIB.ram_snapshot(result)
    if code:raise OSError(code,'Cannot obtain fresh native RAM snapshot')
    return dict(rss_bytes=int(result[0]),available_bytes=int(result[1]),effective_total_bytes=int(result[2]))

def fits(needed,overhead=128<<20):
    budget=int(os.environ.get('MEANDER_BUDGET_BYTES','0'))
    if not budget:return True
    values=snapshot()
    return values['rss_bytes']+int(needed)+overhead<=budget and values['available_bytes']>=int(os.environ['MEANDER_RESERVE_BYTES'])+int(needed)+overhead

def check(needed,label='allocation'):
    import allocation_guard as G
    needed=int(needed)
    if needed<0:raise ValueError('Negative allocation estimate')
    if not int(os.environ.get('MEANDER_BUDGET_BYTES','0')):return
    G.COUNT+=1;G.MAX_REQUEST=max(G.MAX_REQUEST,needed)
    if fits(needed):return
    if G.PRESSURE_HANDLER is not None and not G._HANDLING:
        G._HANDLING=True
        try:G.PRESSURE_HANDLER(needed,128<<20)
        finally:G._HANDLING=False
        if fits(needed):return
    raise MemoryError(dict(label=label,needed_bytes=needed,**snapshot()))

def status():return SimpleNamespace(available=snapshot()['available_bytes'])

def make_space(self,needed,overhead):
    if fits(needed,overhead):return
    if self.cache_evictor:
        self.cache_evictor();gc.collect();self.stats['pressure_cache_clears']+=1
    while not fits(needed,overhead):
        candidates=[b for b in self.live.values() if b.array is not None and b.dtype.kind=='u' and b.pins==0]
        if not candidates:return
        max(candidates,key=lambda b:b.nbytes).spill();gc.collect()

def install():
    import allocation_guard as G
    G.check=check
    import sys
    sys.modules['fast_memory']=sys.modules[__name__]
    import bounded_storage
    bounded_storage.Store.make_space=make_space
