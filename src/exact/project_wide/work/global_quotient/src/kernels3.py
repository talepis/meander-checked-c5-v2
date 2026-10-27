"""Exact integer kernels tested against unrestricted NumPy accumulation."""
import numpy as np
import numba as nb

@nb.njit(cache=True,parallel=True)
def maximum_parallel(A):
    n=A.size
    chunk=16384
    count=(n+chunk-1)//chunk
    partial=np.zeros(count,np.uint64)
    X=A.reshape(-1)
    for b in nb.prange(count):
        v=np.uint64(0)
        for i in range(b*chunk,min((b+1)*chunk,n)):
            if X[i]>v:v=X[i]
        partial[b]=v
    result=np.uint64(0)
    for b in range(count):
        if partial[b]>result:result=partial[b]
    return result

@nb.njit(cache=True,parallel=True)
def add_lower(T,V,tr,tc):
    for i in nb.prange(V.shape[0]):
        high=min(V.shape[1],tr+i-tc+1)
        for j in range(high):T[tr+i,tc+j]+=V[i,j]

@nb.njit(cache=True,parallel=True)
def mirror_lower(T):
    tile=64;n=T.shape[0];nt=(n+tile-1)//tile
    for block in nb.prange(nt*nt):
        br=block//nt;bc=block%nt
        if br<=bc:
            for i in range(br*tile,min(n,(br+1)*tile)):
                for j in range(max(i+1,bc*tile),min(n,(bc+1)*tile)):
                    T[i,j]=T[j,i]

@nb.njit(cache=True,parallel=True)
def rows_lower(T,ptr,ix,V):
    for i in nb.prange(T.shape[0]):
        for z in range(ptr[i],ptr[i+1]):
            r=ix[z]
            for j in range(min(i+1,V.shape[1])):T[i,j]+=V[r,j]

@nb.njit(cache=True,parallel=True)
def cols_lower(T,maps,V):
    for i in nb.prange(V.shape[0]):
        for j in range(V.shape[1]):
            c=maps[j]
            if c<=i:T[i,c]+=V[i,j]

@nb.njit(cache=True,parallel=True)
def gather_add_lower(T,r0,c0,K,ir,ic,rs,re,cs,ce):
    for i in nb.prange(re-rs):
        ki=ir[rs+i]
        for j in range(min(ce-cs,r0+i-c0+1)):
            T[r0+i,c0+j]+=K[ki,ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def gather_cols_lower(T,maps,K,ir,ic,rs,re,cs,ce):
    for i in nb.prange(re-rs):
        ki=ir[rs+i]
        for j in range(ce-cs):
            c=maps[j]
            if c<=i:T[i,c]+=K[ki,ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def gather_rows_lower(T,ptr,ix,K,ir,ic,rs,re,cs,ce):
    for i in nb.prange(T.shape[0]):
        for z in range(ptr[i],ptr[i+1]):
            ki=ir[rs+ix[z]]
            for j in range(min(i+1,ce-cs)):T[i,j]+=K[ki,ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def scatter_rows_sum(T,ptr,ix,V):
    """Accumulate locally into tiles, write the target once per operation."""
    tile=128;ncol=V.shape[1];nt=(ncol+tile-1)//tile
    for block in nb.prange(T.shape[0]*nt):
        i=block//nt;j0=(block%nt)*tile;length=min(tile,ncol-j0)
        temp=np.zeros(tile,np.uint64)
        for z in range(ptr[i],ptr[i+1]):
            r=ix[z]
            for j in range(length):temp[j]+=V[r,j0+j]
        for j in range(length):T[i,j0+j]+=temp[j]
