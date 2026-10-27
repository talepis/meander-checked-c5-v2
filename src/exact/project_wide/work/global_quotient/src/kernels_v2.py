"""Exact locality-preserving gathers and certified compact profile products."""
import numpy as np
import numba as nb
from functools import lru_cache

@nb.njit(cache=True)
def inverse_map(m, size):
    ptr=np.zeros(size+1,np.int64)
    for j in range(len(m)):ptr[m[j]+1]+=1
    for i in range(size):ptr[i+1]+=ptr[i]
    ix=np.empty(len(m),np.int32);pos=ptr[:-1].copy()
    for j in range(len(m)):
        k=m[j];ix[pos[k]]=j;pos[k]+=1
    return ptr,ix

@nb.njit(cache=True,parallel=True)
def scatter_rows_inverse(T,ptr,ix,V):
    # Threads own target rows; each read of a source row is contiguous.
    for r in nb.prange(T.shape[0]):
        for z in range(ptr[r],ptr[r+1]):
            i=ix[z]
            for j in range(V.shape[1]):T[r,j]+=V[i,j]

@nb.njit(cache=True,parallel=True)
def scatter_cols_inverse(T,ptr,ix,V):
    for i in nb.prange(V.shape[0]):
        for j in range(T.shape[1]):
            s=np.uint64(0)
            for z in range(ptr[j],ptr[j+1]):s+=np.uint64(V[i,ix[z]])
            T[i,j]+=s

@nb.njit(cache=True,parallel=True)
def add_parallel(T,V):
    for i in nb.prange(T.shape[0]):
        for j in range(T.shape[1]):T[i,j]+=V[i,j]

@nb.njit(cache=True,parallel=True)
def add_tiled(T,V):
    tile=64;nc=(T.shape[1]+tile-1)//tile
    for ij in nb.prange(((T.shape[0]+tile-1)//tile)*nc):
        i0=(ij//nc)*tile;j0=(ij%nc)*tile
        for i in range(i0,min(i0+tile,T.shape[0])):
            for j in range(j0,min(j0+tile,T.shape[1])):T[i,j]+=V[i,j]

@nb.njit(cache=True,parallel=True)
def seed_rows_inverse(T,ptr,ix,K,ir,ic,rs,re,cs,ce):
    for r in nb.prange(T.shape[0]):
        for z in range(ptr[r],ptr[r+1]):
            ki=ir[rs+ix[z]]
            for j in range(ce-cs):T[r,j]+=K[ki,ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def seed_rows_grouped(T,ptr,ix,K,ir,ic,rs,re,cs,ce):
    for r in nb.prange(T.shape[0]):
        # Sum before expanding repeated column profiles.
        if ptr[r]==ptr[r+1]:continue
        tmp=np.zeros(K.shape[1],np.uint64)
        for z in range(ptr[r],ptr[r+1]):
            ki=ir[rs+ix[z]]
            for k in range(K.shape[1]):tmp[k]+=K[ki,k]
        for j in range(ce-cs):T[r,j]+=tmp[ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def seed_cols_grouped(T,ptr,ix,K,ir,ic,rs,re,cs,ce):
    # Compute the small column-side contraction in independent rows.
    reduced=np.zeros((K.shape[0],T.shape[1]),np.uint64)
    for k in nb.prange(K.shape[0]):
        for j in range(T.shape[1]):
            s=np.uint64(0)
            for z in range(ptr[j],ptr[j+1]):s+=np.uint64(K[k,ic[cs+ix[z]]])
            reduced[k,j]=s
    for i in nb.prange(re-rs):
        ki=ir[rs+i]
        for j in range(T.shape[1]):T[i,j]+=reduced[ki,j]

@nb.njit(cache=True,parallel=True)
def fill_symmetric(K,P,choose,mod):
    for i in nb.prange(len(P)):
        for j in range(i+1):
            s=np.uint64(1)
            for k in range(P.shape[1]):s*=choose[int(P[i,k])+int(P[j,k]),int(P[i,k])]
            if mod:s%=np.uint64(mod)
            K[i,j]=s;K[j,i]=s
