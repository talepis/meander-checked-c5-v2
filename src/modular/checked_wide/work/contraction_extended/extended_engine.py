"""Isolated candidate: exact symmetry and algebraic reassociation of V2 reads."""
import tile_engine as BASE
import best_engine as R
import tile_transfer as X
import extended_v2 as CV
import variants_next as V
import bounded_storage as BS
import math
import numpy as np


def meander(*args, **kwargs):
    saved = R.CV, X.CV, V.clear, BS.Store.create
    CV.clear()
    old_order=CV.RUN_ORDER
    old_force=CV.FORCE_EXPANSION
    force=kwargs.pop('force_expansion',None)
    if force is not None:CV.FORCE_EXPANSION=bool(force)
    CV.RUN_ORDER=int(args[0] if args else kwargs["n"])
    original_clear=V.clear
    original_create=BS.Store.create
    def create(store,shape,dtype):
        dt=np.dtype(dtype)
        amount=math.prod(shape)*(64 if dt.kind=='O' else dt.itemsize)
        if store.ram+amount>store.ram_limit or amount>store.small_limit:
            released=CV.release(store)
            store.stats['pool_pressure_events']=store.stats.get('pool_pressure_events',0)+1
            store.stats['mirror_bytes_released_before_allocation']=store.stats.get('mirror_bytes_released_before_allocation',0)+released
        block=original_create(store,shape,dtype)
        if block.array is None:CV.release(store)
        return block
    def evict():
        CV.release()
        original_clear()
    V.clear=evict
    BS.Store.create=create
    R.CV = X.CV = CV
    try:
        result = BASE.meander(*args, **kwargs)
        result['force_extended_from_start']=CV.FORCE_EXPANSION
    finally:
        R.CV, X.CV, V.clear, BS.Store.create = saved
        CV.RUN_ORDER=old_order
        CV.FORCE_EXPANSION=old_force
    CV.clear()
    return result
