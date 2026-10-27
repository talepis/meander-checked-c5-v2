"""Read each immutable predecessor tile once for all retained V2 sections."""
import numpy as np
import numba as nb
import allocation_guard as G


def estimate(owner, transitions):
    # Sorting, coordinate copies, indices, tile IDs, and group views are covered.
    n, m = owner.shape
    per = n + m + len(owner.p[5]) + len(owner.q[5]) + len(owner.p[6]) + len(owner.q[6])
    if getattr(owner, "diagonal", False):per=n+len(owner.q[6])+len(owner.p[5])+len(owner.q[5])
    if getattr(owner, "H", True) is None:
        per-=len(owner.p[5])+len(owner.q[5])
    return 96 * per * len(transitions) + (8 << 20)


def transitions(owner):
    a, b, h = owner.key
    return ((a-1,b-1,h-1), (a-1,b+1,h+1), (a+1,b-1,h+1))


def canonical(key):
    a,b,h=key
    return min(a,b),max(a,b),h


def eligible(owner, A, key):
    same = [x for x in transitions(owner) if canonical(x)==canonical(key)]
    return estimate(owner, same) <= owner.store.chunk


@nb.njit(cache=True)
def accumulate(values, tile, dstrows, dstcols, srcrows, srccols, flip, cap):
    limit=np.uint64(cap)
    for i in range(len(dstrows)):
        for j in range(len(dstcols)):
            v=np.uint64(tile[srcrows[i],srccols[j]])
            r=dstcols[j] if flip else dstrows[i]
            c=dstrows[i] if flip else dstcols[j]
            old=np.uint64(values[r,c])
            if v>limit or old>limit-v:return 1
            values[r,c]=old+v
    return 0


def axis_groups(coords, positions, size):
    order=np.argsort(coords,kind='stable')
    coords=coords[order];positions=positions[order]
    tiles=coords//size
    edges=np.concatenate(([0],np.flatnonzero(tiles[1:]!=tiles[:-1])+1,[len(coords)])) if len(coords) else [0]
    groups={}
    for a,b in zip(edges[:-1],edges[1:]):
        t=int(tiles[a]);groups[t]=(coords[a:b]-t*size,positions[a:b])
    return groups


def requests(owner, selected):
    import engine_v2 as E
    a,b,h=owner.key;n,m=owner.shape
    up,uq=E.mt.B(a-1,h-1),E.mt.B(b-1,h-1)
    # No coefficient arrays are constructed by this metadata step.
    sections=[(owner.R,np.arange(n,dtype=np.int64),owner.q[6])]
    if owner.H is not None:sections.append((owner.H,owner.p[5],owner.q[5]))
    if not owner.diagonal:
        sections.append((owner.C,owner.p[6],np.arange(m,dtype=np.int64)))
    answer=[]
    for key in sorted(selected):
        if key==(a-1,b-1,h-1):r0,r1,c0,c1,dr,dc=0,up,0,uq,0,0
        elif key==(a-1,b+1,h+1):r0,r1,c0,c1,dr,dc=up,n,0,m,up,0
        elif key==(a+1,b-1,h+1):r0,r1,c0,c1,dr,dc=0,n,uq,m,0,uq
        else:raise ValueError(key)
        flip=key[0]>key[1]
        for part,rr,cc in sections:
            ri=np.flatnonzero((rr>=r0)&(rr<r1));ci=np.flatnonzero((cc>=c0)&(cc<c1))
            if not len(ri) or not len(ci):continue
            rows=rr[ri]-dr;cols=cc[ci]-dc
            if flip:rows,cols,ri,ci=cols,rows,ci,ri
            answer.append((part,rows,cols,ri,ci,flip))
    return answer


def apply(owner, A, selected, stats):
    amount=estimate(owner,selected)
    assert amount<=owner.store.chunk
    G.check(amount,'joint immutable predecessor sample workspace')
    groups=[]
    for part,rows,cols,ri,ci,flip in requests(owner,selected):
        groups.append((part,axis_groups(rows,ri,A.tile_rows),axis_groups(cols,ci,A.tile_cols),flip))
    rowtiles=sorted({r for _,rg,_,_ in groups for r in rg})
    stats['joint_file_predecessors']=stats.get('joint_file_predecessors',0)+1
    for row in rowtiles:
        active=[(p,rg[row],cg,f) for p,rg,cg,f in groups if row in rg]
        cols=sorted({c for _,_,cg,_ in active for c in cg})
        for col in cols:
            G.check(A.tile_bytes,'joint predecessor file tile')
            tile=np.empty((A.tile_rows,A.tile_cols),A.dtype)
            A.file.seek((row*A.grid_cols+col)*A.tile_bytes)
            A._read_exact(memoryview(tile).cast('B'))
            A.store.stats['read_bytes']+=A.tile_bytes
            stats['joint_file_tile_reads']=stats.get('joint_file_tile_reads',0)+1
            for part,(sr,dr),cg,flip in active:
                if col not in cg:continue
                sc,dc=cg[col]
                assert not accumulate(part.array,tile,dr,dc,sr,sc,flip,owner.cap),'joint sample exceeds the retained cap'
                stats['joint_sample_cell_updates']=stats.get('joint_sample_cell_updates',0)+len(dr)*len(dc)
            del tile
