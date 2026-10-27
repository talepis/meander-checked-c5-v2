"""Live RSS + host/cgroup-v2 headroom; no matrix files or disk spill."""
import math, os
from pathlib import Path
import psutil
COUNT=0
MAX_REQUEST=0
PRESSURE_HANDLER=None
_HANDLING=False

def snapshot():
    vm=psutil.virtual_memory()
    available=int(vm.available); total=int(vm.total)
    if os.name=='posix' and Path('/sys/fs/cgroup/memory.max').is_file():
        roots={Path('/sys/fs/cgroup')}
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                path=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
                if path.is_dir():
                    roots.update(p for p in [path,*path.parents] if p==Path('/sys/fs/cgroup') or Path('/sys/fs/cgroup') in p.parents)
        for root in roots:
            cap=root/'memory.max';current=root/'memory.current'
            if cap.is_file() and current.is_file():
                raw=cap.read_text().strip()
                if raw!='max':
                    limit=int(raw);used=int(current.read_text());total=min(total,limit)
                    available=min(available,max(0,limit-used))
    return dict(rss_bytes=psutil.Process().memory_info().rss,
                available_bytes=available,effective_total_bytes=total)

def check(needed,label='allocation'):
    global COUNT,MAX_REQUEST
    needed=int(needed)
    if needed<0:raise ValueError('Negative allocation size')
    COUNT+=1;MAX_REQUEST=max(MAX_REQUEST,needed)
    budget=int(os.environ.get('MEANDER_BUDGET_BYTES','0'))
    if not budget:raise RuntimeError('A positive explicit memory budget is mandatory')
    reserve=int(os.environ.get('MEANDER_RESERVE_BYTES',str(16<<30)))
    v=snapshot();overhead=128<<20
    if v['rss_bytes']+needed+overhead>budget or v['available_bytes']<reserve+needed+overhead:
        raise MemoryError(dict(label=label,needed_bytes=needed,budget_bytes=budget,
                               reserve_bytes=reserve,**v))

def array(shape,dtype,label='array'):
    import numpy as np
    check(math.prod(map(int,shape))*np.dtype(dtype).itemsize,label)
