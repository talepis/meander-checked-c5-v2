import numpy as np
import kernels3 as k
import kernels_v2 as prev
import configure_v2 as c
E=c.engine
original_scatter=c.scatter
oldga,oldgc,oldgr=E.ms.gather_add,E.ms.gather_cols,E.ms.gather_rows
TRI=False


def max_parallel(A):
    return k.maximum_parallel(A) if A.size>500000 and A.dtype!=object and A.flags.c_contiguous else A.max(initial=0)

def tri_scatter(T,V,kind,maps,threshold=100000):
    if T.shape[0]==T.shape[1] and T.dtype!=object and V.dtype!=object:
        if kind=='rows':k.rows_lower(T,*c.inverse(maps,T.shape[0]),V)
        else:k.cols_lower(T,maps,V)
    else:original_scatter(T,V,kind,maps,threshold)

def ga(T,r0,c0,K,ir,ic,rs,re,cs,ce):
    if T.shape[0]==T.shape[1] and T.dtype!=object:
        k.gather_add_lower(T,r0,c0,K,ir,ic,rs,re,cs,ce)
    else:oldga(T,r0,c0,K,ir,ic,rs,re,cs,ce)

def gc(T,maps,K,ir,ic,rs,re,cs,ce):
    if T.shape[0]==T.shape[1] and T.dtype!=object:
        k.gather_cols_lower(T,maps,K,ir,ic,rs,re,cs,ce)
    else:oldgc(T,maps,K,ir,ic,rs,re,cs,ce)

def gr(T,maps,K,ir,ic,rs,re,cs,ce):
    if T.shape[0]==T.shape[1] and T.dtype!=object:
        k.gather_rows_lower(T,*c.inverse(maps,T.shape[0]),K,ir,ic,rs,re,cs,ce)
    else:oldgr(T,maps,K,ir,ic,rs,re,cs,ce)

def tileadd(T,V):
    if V.strides[0]<V.strides[1]:prev.add_tiled(T,V)
    else:prev.add_parallel(T,V)

def sumscatter(T,V,kind,maps,threshold=100000):
    if V.size>threshold and T.dtype!=object and V.dtype!=object:
        if kind=='rows' and V.strides[1]<=V.strides[0]:
            k.scatter_rows_sum(T,*c.inverse(maps,T.shape[0]),V);return
        if kind=='cols' and V.strides[0]<V.strides[1]:
            k.scatter_rows_sum(T.T,*c.inverse(maps,T.shape[1]),V.T);return
    original_scatter(T,V,kind,maps,threshold)


def configure(variant,config):
    parts=set(variant.split('+'))
    for part in parts:
        if part=='max':E.MAXIMUM=max_parallel
        elif part=='down':E.DOWNSIZE=True
        elif part=='u8':E._UINT8=True
        elif part=='tri':
            E.LOWER_ONLY=True;E.LOWER_ADD=k.add_lower;E.MIRROR=k.mirror_lower
            E._add_scatter=tri_scatter
            E.ms.gather_add=ga;E.ms.gather_cols=gc;E.ms.gather_rows=gr
        elif part=='tile':E.ADD_KERNEL=tileadd
        elif part=='sum':E._add_scatter=sumscatter
        elif part=='forcing':
            import local_forcing as f
            E.LOCAL_FORCING=f.inject
        elif part=='directbounds':
            from block_bounds3 import add_forcing_bounds
            E.LOCAL_BOUNDS=add_forcing_bounds
        elif part=='bound':
            from block_bounds3 import refine
            E.BOUND_REFINER=refine
        elif part=='order':
            from block_bounds3 import memory_order
            E.SOURCE_ORDER=memory_order
        elif part=='guard':E.MEMORY_GUARD=True
        elif part=='closed2':
            import local_forcing as f
            f.CLOSED_TWO=True
        elif part=='release':
            import local_forcing as f
            E.LOCAL_FORCING=f.inject_skip_fused;E.PRESTAGE=f.prepare_release;E.ON_RELEASE=f.fuse_on_release
        elif part=='tight':
            import local_forcing as f
            f.TIGHT=True;E.LOCAL_FORCING=f.inject_skip_fused;E.PRESTAGE=f.prestage
        elif part=='fused':
            import local_forcing as f
            E.LOCAL_FORCING=f.inject_skip_fused;E.PRESTAGE=f.prestage
        else:raise ValueError(part)

def clear():
    import sys
    if 'local_forcing' in sys.modules:sys.modules['local_forcing'].clear()
