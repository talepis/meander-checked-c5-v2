"""Construct exactly the close_top map, using ballot ranking; no table of M_n."""
import numpy as np
import numba as nb
import adaptive_exact as ae
import allocation_guard as guard
@nb.njit(cache=True)
def build_map(m,H,N,B):
    out=np.empty(N,np.int64);bits=np.empty(m,np.uint8);stack=np.empty(m,np.int64)
    for rank in range(N):
        h=H;r=rank
        for mm in range(m,0,-1):
            u=B[mm-1,h-1] if h else 0
            if r<u:bits[mm-1]=1;h-=1
            else:bits[mm-1]=0;r-=u;h+=1
        depth=0
        for i in range(m):
            if bits[i]:stack[depth]=i;depth+=1
            else:depth-=1
        assert depth==H and H>=2
        bits[stack[depth-1]]=0
        h=H-2;r=0
        for mm in range(m,0,-1):
            if bits[mm-1]:h-=1
            else:
                if h:r+=B[mm-1,h-1]
                h+=1
        assert h==0
        out[rank]=r
    return out

def extend(n):
    needed=set()
    for t in range(2*n-1):
        plan,_=ae.operations(n,t,ae.mt.canon_keys(n,t))
        for ops in plan.values():
            for dest,kind,flip,rect,off,mp in ops:
                if mp is not None:needed.add(mp)
    created=[]
    for m,h in sorted(needed):
        if (m,h) in ae.mt._close:continue
        B=np.zeros((m+1,m+3),np.int64)
        for j in range(m+1):
            for k in range(m+2):B[j,k]=ae.mt.B(j,k)
        N=ae.mt.B(m,h)
        guard.array((N,),np.int64,('close-map',m,h))
        ae.mt._close[m,h]=build_map(m,h,N,B)
        created.append({'m':m,'h':h,'size':N})
    return created
