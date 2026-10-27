"""Allocation checks for this workspace's supervised, bounded workers."""
import os
import psutil

COUNT=0
MAX_REQUEST=0
PRESSURE_HANDLER=None
_HANDLING=False
def check(needed, label='allocation'):
    global COUNT,MAX_REQUEST
    needed=int(needed)
    if needed<0:raise ValueError('Negative allocation estimate')
    budget=int(os.environ.get('MEANDER_BUDGET_BYTES','0'))
    if not budget:return
    reserve=int(os.environ['MEANDER_RESERVE_BYTES'])
    overhead=128<<20
    mem=psutil.Process().memory_info()
    committed=max(mem.rss,mem.vms)
    available=psutil.virtual_memory().available
    COUNT+=1;MAX_REQUEST=max(MAX_REQUEST,needed)
    global _HANDLING
    if (committed+needed+overhead>budget or available<reserve+needed+overhead) and PRESSURE_HANDLER is not None and not _HANDLING:
        _HANDLING=True
        try:PRESSURE_HANDLER(needed,overhead)
        finally:_HANDLING=False
        mem=psutil.Process().memory_info();committed=max(mem.rss,mem.vms)
        available=psutil.virtual_memory().available
    if committed+needed+overhead>budget or available<reserve+needed+overhead:
        raise MemoryError(dict(label=label,needed_bytes=needed,committed_bytes=committed,available_bytes=available,budget_bytes=budget,reserve_bytes=reserve))

def array(shape,dtype,label='array'):
    import math,numpy as np
    # Conservative allowance for Python integer cells in arbitrary-precision tails.
    width=64 if np.dtype(dtype)==np.dtype(object) else np.dtype(dtype).itemsize
    check(math.prod(map(int,shape))*width,label)
