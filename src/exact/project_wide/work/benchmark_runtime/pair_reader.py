"""Read only requested pairs, never their enclosing Cartesian coefficient grid."""
import time
import numpy as np
import allocation_guard as G
import bounded_storage as BS
import packed_boundary as PB
import extended_v2 as CV
import pair_kernels as K
from checked_caps import require_decoded

def read(A,rows,cols):
    if len(rows)!=len(cols):raise ValueError('paired index lengths')
    G.check(len(rows)*max(32,64 if A.dtype==object else A.dtype.itemsize),'requested coefficient pairs')
    if isinstance(A,BS.View):
        a,b,c,d=A.rect;rr=rows+a;cc=cols+c
        return read(A.base,cc,rr) if A.flip else read(A.base,rr,cc)
    if isinstance(A,PB.SubView):return read(A.base,rows+A.r0,cols+A.c0)
    if isinstance(A,CV.Compact):
        if any(p.array is None for p in A.parts):raise MemoryError('compact section is not resident')
        with A.store.borrowed(*A.parts):
            values,errors=K.decode(A.R.array,A.C.array,A.p[6],A.p[0],A.q[0],A.delta_p[0],A.delta_q[1],A.delta_q[2],A.delta_q[3],A.profile_P,A.profile_Q,A.choose,rows,cols,A.cap,np.empty(0,A.dtype))
        require_decoded(errors)
        return values
    if isinstance(A,PB.PackedBoundary):
        out=np.empty(len(rows),A.dtype);count=0
        for a,b,c,d,part in A.regions:
            ix=np.flatnonzero((rows>=a)&(rows<b)&(cols>=c)&(cols<d))
            if len(ix):out[ix]=read(part,rows[ix]-a,cols[ix]-c);count+=len(ix)
        if count!=len(rows):raise ArithmeticError('requested pair enters an omitted region')
        return out
    if A.array is not None:
        with A.store.borrowed(A):return A.array[rows,cols]
    # Ordinary-file fallback: each needed source tile is fetched once per batch.
    out=np.empty(len(rows),A.dtype)
    tile_ids=(rows//A.tile_rows)*A.grid_cols+cols//A.tile_cols
    order=np.argsort(tile_ids,kind='stable');sorted_ids=tile_ids[order]
    ends=np.concatenate(([0],np.flatnonzero(sorted_ids[1:]!=sorted_ids[:-1])+1,[len(rows)])) if len(rows) else np.array([0])
    for lo,hi in zip(ends,ends[1:]):
        ii=order[lo:hi];code=int(tile_ids[ii[0]])
        br=(code//A.grid_cols)*A.tile_rows;bc=(code%A.grid_cols)*A.tile_cols
        tile=A.read(br,min(br+A.tile_rows,A.shape[0]),bc,min(bc+A.tile_cols,A.shape[1]))
        out[ii]=tile[rows[ii]-br,cols[ii]-bc]
        del tile
    return out
