"""One exact source union, multiple RAM destination updates; no retained cache."""
import time
import numpy as np
import numba as nb
import allocation_guard as G
import packed_boundary as PB

STATS={}
def clear():
    STATS.clear();STATS.update(sources=0,operations=0,source_cells=0,independent_cells=0,
        reused_cells=0,seconds=0.,fallback_sources=0,largest_buffer_bytes=0)

@nb.njit(cache=True,parallel=True,nogil=True)
def rows_kernel(T,B,maps,order,ends):
    # Each destination row has a unique owner, including repeated map images.
    for k in nb.prange(len(ends)-1):
        lo=ends[k];hi=ends[k+1];row=maps[order[lo]]
        for z in range(lo,hi):
            i=order[z]
            for j in range(B.shape[1]):T[row,j]+=B[i,j]

@nb.njit(cache=True,parallel=True,nogil=True)
def cols_kernel(T,B,maps):
    # Row strips own disjoint destinations; duplicate columns remain serial.
    for block in nb.prange((B.shape[0]+31)//32):
        first=block*32;last=min(first+32,B.shape[0])
        for j in range(B.shape[1]):
            c=maps[j]
            for i in range(first,last):T[i,c]+=B[i,j]

class ArrayView:
    def __init__(self,array):self.array=array;self.shape=array.shape;self.dtype=array.dtype
    def read(self,r0,r1,c0,c1):return self.array[r0:r1,c0:c1]

def resident(T):
    return T.dtype.kind=='u' and all(p.array is not None for *_,p in T.regions) if isinstance(T,PB.PackedBoundary) else T.dtype.kind=='u' and T.array is not None

def base_rectangle(V):
    a,b,c,d=V.rect
    return (c,d,a,b) if V.flip else (a,b,c,d)

def unions(rects):
    edges=sorted({v for r in rects for v in r[:2]})
    for a,b in zip(edges,edges[1:]):
        intervals=sorted((c,d) for r,s,c,d in rects if r<=a and b<=s and c<d)
        merged=[]
        for c,d in intervals:
            if merged and c<=merged[-1][1]:merged[-1]=(merged[-1][0],max(d,merged[-1][1]))
            else:merged.append((c,d))
        for c,d in merged:yield a,b,c,d

def scatter_array(T,B,kind,maps,other_offset):
    assert T.ndim==B.ndim==2
    if kind=='rows':
        A=T[:,other_offset:other_offset+B.shape[1]]
        order=np.argsort(maps,kind='stable')
        sorted_maps=maps[order]
        ends=np.concatenate(([0],np.flatnonzero(sorted_maps[1:]!=sorted_maps[:-1])+1,[len(maps)])).astype(np.int64)
        if len(maps):rows_kernel(A,B,maps,order,ends)
    else:
        A=T[other_offset:other_offset+B.shape[0],:]
        cols_kernel(A,B,maps)

def scatter(T,B,kind,maps,other_offset):
    if not isinstance(T,PB.PackedBoundary):return scatter_array(T.array,B,kind,maps,other_offset)
    if len(maps):assert int(maps.min())>=T.u
    shifted=maps-T.u
    # Split only the unmapped coordinate; every mapped image is in the D region.
    axis=1 if kind=='rows' else 0
    end=other_offset+B.shape[axis];u=T.u
    names=('BL','BR') if kind=='rows' else ('TR','BR')
    for name,start,stop in ((names[0],0,u),(names[1],u,T.shape[axis])):
        a=max(start,other_offset);b=min(stop,end)
        if a>=b:continue
        sub=B[:,a-other_offset:b-other_offset] if kind=='rows' else B[a-other_offset:b-other_offset,:]
        scatter_array(T.blocks[name].array,sub,kind,shifted,a-start)

def apply(A,requests,old_add,minimum_saved_cells=1_000_000):
    """Return False before mutation unless all targets and workspace are usable."""
    rects=[base_rectangle(req[1]) for req in requests]
    bands=list(unions(rects))
    total=sum((b-a)*(d-c) for a,b,c,d in rects)
    union=sum((b-a)*(d-c) for a,b,c,d in bands)
    if total-union<minimum_saved_cells or total<1.2*union:return False
    if not all(resident(T) for T,*_ in requests):return False
    store=A.store
    # Two source-window allowances cover decoder metadata and local maps as well
    # as the single decoded slab. This is workspace, never a new large matrix.
    needed=2*store.chunk+max(A.shape)*64
    with store.borrowed(A,*(T for T,*_ in requests)):
        try:G.check(needed,'RAM source union and scatter workspace')
        except MemoryError:return False
        if not all(resident(T) for T,*_ in requests):return False
        start=time.perf_counter()
        for a,b,c,d in bands:
            step=max(1,store.chunk//max(1,(d-c)*A.dtype.itemsize))
            for r0 in range(a,b,step):
                r1=min(b,r0+step);values=A.read(r0,r1,c,d)
                STATS['largest_buffer_bytes']=max(STATS['largest_buffer_bytes'],values.nbytes)
                for (T,V,kind,off,maps,backend),(ra,rb,ca,cb) in zip(requests,rects):
                    lo=max(r0,ra);hi=min(r1,rb);left=max(c,ca);right=min(d,cb)
                    if lo>=hi or left>=right:continue
                    B=values[lo-r0:hi-r0,left-c:right-c]
                    if V.flip:B=B.T;vr=left-ca;vc=lo-ra
                    else:vr=lo-ra;vc=left-ca
                    if kind=='add':old_add(T,ArrayView(B),(off[0]+vr,off[1]+vc))
                    elif kind=='rows':scatter(T,B,kind,maps[vr:vr+B.shape[0]],vc)
                    else:scatter(T,B,kind,maps[vc:vc+B.shape[1]],vr)
                    del B
                del values
        STATS['sources']+=1;STATS['operations']+=len(requests)
        STATS['source_cells']+=union;STATS['independent_cells']+=total
        STATS['reused_cells']+=total-union;STATS['seconds']+=time.perf_counter()-start
    return True

clear()
