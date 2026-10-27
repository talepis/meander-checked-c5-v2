"""Isolated terminal-functional core, retaining the C5 storage and RAM kernels."""
import best_engine as R
import fanout_engine as BASE
import functional_core as CORE
import os

def meander(*args,**kwargs):
    backend=os.environ.get('MEANDER_TERMINAL_BACKEND','cpu')
    if backend!='cpu':
        import opencl_geometry as GPU
        GPU.reset_stats()
    previous=R.meander;R.meander=CORE.meander
    try:
        result=BASE.meander(*args,**kwargs)
        if backend!='cpu':
            assert GPU.STATS['gpu_live_bytes']==0
            result['accelerator_stats']=GPU.STATS.copy()
        return result
    finally:R.meander=previous
