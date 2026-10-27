"""File-tile destination updates under the retained coefficient/RAM budgets.

No persistent new cache. RAM operations retain the best engine's kernels.
Every native destination sum retains its original BEFORE-addition bound.
"""
import math
import numpy as np
import numba as nb
import allocation_guard as G
import bounded_storage as BS
import compact_v2 as CV
import packed_boundary as PB
import kernels_v2 as KV

OLD_ADD=BS.add;OLD_SCATTER=BS.scatter
OLD_RECT=BS.Block.rectangles;OLD_PACK_RECT=PB.PackedBoundary.rectangles
STATS={}
def clear():
    STATS.clear();STATS.update(file_add_tiles=0,file_scatter_tiles=0,skipped_empty_axis_tiles=0,
        scatter_fallbacks=0,inverse_peak_bytes=0,largest_gather_bytes=0,persistent_cache_bytes=0)

def full_tile(T,ri,ci):
    G.check(T.tile_bytes,'destination file tile')
    A=np.empty((T.tile_rows,T.tile_cols),T.dtype)
    offset=(ri*T.grid_cols+ci)*T.tile_bytes
    T.file.seek(offset);T._read_exact(memoryview(A).cast('B'));T.store.stats['read_bytes']+=T.tile_bytes
    return A,offset

def write_tile(T,A,offset):
    T.file.seek(offset);T._write_exact(memoryview(A).cast('B'));T.store.stats['write_bytes']+=T.tile_bytes

def indexed(V,rows,cols):
    # Flatten the two view kinds while preserving logical ordering and flips.
    if isinstance(V,BS.View):
        a,b,c,d=V.rect;rr=rows+a;cc=cols+c
        return indexed(V.base,cc,rr).T if V.flip else indexed(V.base,rr,cc)
    if isinstance(V,PB.SubView):return indexed(V.base,rows+V.r0,cols+V.c0)
    if isinstance(V,PB.PackedBoundary):
        if np.any(rows<V.u) and np.any(cols<V.u):raise ArithmeticError('read of pruned OO region')
        G.array((len(rows),len(cols)),V.dtype,'packed source gather')
        out=np.empty((len(rows),len(cols)),V.dtype);covered=0
        for a,b,c,d,part in V.regions:
            ii=np.flatnonzero((rows>=a)&(rows<b));jj=np.flatnonzero((cols>=c)&(cols<d))
            if len(ii) and len(jj):
                out[np.ix_(ii,jj)]=indexed(part,rows[ii]-a,cols[jj]-c);covered+=len(ii)*len(jj)
        assert covered==out.size
        return out
    return CV.indexed(V,rows,cols)

def add(T,V,off=(0,0)):
    if T.array is not None or T.dtype==object or V.dtype==object:return OLD_ADD(T,V,off)
    nr,nc=V.shape;tr,tc=off
    if not nr or not nc:return
    T._validate(tr,tr+nr,tc,tc+nc)
    for ri in range(tr//T.tile_rows,(tr+nr+T.tile_rows-1)//T.tile_rows):
        br=ri*T.tile_rows;a=max(tr,br);b=min(tr+nr,br+T.tile_rows)
        for ci in range(tc//T.tile_cols,(tc+nc+T.tile_cols-1)//T.tile_cols):
            bc=ci*T.tile_cols;c=max(tc,bc);d=min(tc+nc,bc+T.tile_cols)
            A,offset=full_tile(T,ri,ci);B=V.read(a-tr,b-tr,c-tc,d-tc)
            np.add(A[a-br:b-br,c-bc:d-bc],B,out=A[a-br:b-br,c-bc:d-bc],casting='unsafe')
            write_tile(T,A,offset);STATS['file_add_tiles']+=1
            del A,B

@nb.njit(cache=True)
def gather_rows(A,B,ptr,first,base,nr,nc):
    for i in range(nr):
        for j in range(nc):
            value=np.uint64(A[i,j])
            for z in range(ptr[first+i]-base,ptr[first+i+1]-base):value+=np.uint64(B[z,j])
            A[i,j]=value

@nb.njit(cache=True)
def gather_cols(A,B,ptr,first,base,nr,nc):
    for i in range(nr):
        for j in range(nc):
            value=np.uint64(A[i,j])
            for z in range(ptr[first+j]-base,ptr[first+j+1]-base):value+=np.uint64(B[i,z])
            A[i,j]=value

def scatter(T,V,kind,maps,backend):
    if T.array is not None or T.dtype==object or V.dtype==object:return OLD_SCATTER(T,V,kind,maps,backend)
    axis=0 if kind=='rows' else 1
    if kind not in ('rows','cols') or len(maps)!=V.shape[axis]:raise ValueError('invalid scatter axis')
    n=T.shape[axis]
    if V.shape[1-axis]>T.shape[1-axis]:raise ValueError('source exceeds other target axis')
    if len(maps) and (int(maps.min())<0 or int(maps.max())>=n):raise ValueError('invalid map image')
    estimate=(n+1)*16+len(maps)*4
    # Local inverse workspace; fall back rather than exceed the old buffer budget.
    if estimate>T.store.chunk//4:
        STATS['scatter_fallbacks']+=1;return OLD_SCATTER(T,V,kind,maps,backend)
    G.check(estimate,'temporary file scatter inverse')
    ptr,ix=KV.inverse_map(maps,n)
    STATS['inverse_peak_bytes']=max(STATS['inverse_peak_bytes'],ptr.nbytes+ix.nbytes)
    axis_tile=T.tile_rows if axis==0 else T.tile_cols
    other_tile=T.tile_cols if axis==0 else T.tile_rows
    for first in range(0,n,axis_tile):
        last=min(n,first+axis_tile);lo=int(ptr[first]);hi=int(ptr[last])
        if lo==hi:STATS['skipped_empty_axis_tiles']+=1;continue
        need=(hi-lo)*other_tile*V.dtype.itemsize
        if need>T.store.chunk//2:
            # Decide before modifying any tile for this operation.
            # Maximum indegree certificate normally keeps the window below 32 MiB.
            STATS['scatter_fallbacks']+=1
            del ptr,ix
            return OLD_SCATTER(T,V,kind,maps,backend)
    for first in range(0,n,axis_tile):
        last=min(n,first+axis_tile);lo=int(ptr[first]);hi=int(ptr[last])
        if lo==hi:continue
        source=ix[lo:hi]
        for second in range(0,V.shape[1-axis],other_tile):
            end=min(V.shape[1-axis],second+other_tile)
            rows=source if axis==0 else np.arange(second,end,dtype=np.int64)
            cols=np.arange(second,end,dtype=np.int64) if axis==0 else source
            G.check((hi-lo)*(end-second)*V.dtype.itemsize+T.tile_bytes,'bounded file scatter gather')
            B=indexed(V,rows,cols)
            STATS['largest_gather_bytes']=max(STATS['largest_gather_bytes'],B.nbytes)
            ri,ci=(first//T.tile_rows,second//T.tile_cols) if axis==0 else (second//T.tile_rows,first//T.tile_cols)
            A,offset=full_tile(T,ri,ci)
            if axis==0:gather_rows(A,B,ptr,first,lo,last-first,end-second)
            else:gather_cols(A,B,ptr,first,lo,end-second,last-first)
            write_tile(T,A,offset);STATS['file_scatter_tiles']+=1
            del A,B

def rectangles(T):
    if T.array is not None:yield from OLD_RECT(T);return
    for r in range(0,T.shape[0],T.tile_rows):
        for c in range(0,T.shape[1],T.tile_cols):
            yield r,min(T.shape[0],r+T.tile_rows),c,min(T.shape[1],c+T.tile_cols)

def packed_rectangles(T):
    for r0,r1,c0,c1,part in T.regions:
        for a,b,c,d in rectangles(part):yield r0+a,r0+b,c0+c,c0+d

clear()
