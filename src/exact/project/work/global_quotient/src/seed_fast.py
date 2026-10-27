"""Exact constant-denominator reciprocals and narrower analytic kernels."""
from functools import lru_cache
import math
import numpy as np
import numba as nb
import adaptive_exact as ae
old,ms=ae.old,ae.ms

@lru_cache(None)
def reciprocals(n):
    k=2*(n-1)//3
    if k*(1<<k)>=(1<<32):raise ValueError('32-bit numerator certificate unavailable')
    R=np.zeros(n+3,np.uint64)
    for d in range(1,n+3):R[d]=((1<<32)+d-1)//d
    return R

@nb.njit(cache=True,parallel=True)
def compact_profile_kernels(P,Q,C,k0example):
    K0=np.empty((len(P),len(Q)),k0example.dtype)
    K1=np.empty((len(P),len(Q)),np.uint32)
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            Z=np.uint64(0);L=np.uint64(1);A=B=np.uint64(0)
            for k in range(P.shape[1]):
                u=int(P[i,k]);v=int(Q[j,k]);m=u+v;c=C[m,u]
                loc=c*m+C[m,u+1]*u+C[m,v+1]*v+2*((v+1)*C[m+1,v+2]+(u+1)*C[m+1,u+2])
                Z=c*Z+A*C[m+1,u]+B*C[m+1,v]+L*loc
                A=L*C[m+1,v];B=L*C[m+1,u];L*=c
            K0[i,j]=L;K1[i,j]=Z
    return K0,K1

@nb.njit(cache=True,parallel=True)
def expand_recip(K0,K1,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,R,dtype_array):
    A=np.empty((len(ip),len(iq)),dtype_array.dtype)
    for i in nb.prange(len(ip)):
        ci=ip[i]
        for j in range(len(iq)):
            cj=iq[j];base=np.uint64(K0[ci,cj])
            x=np.uint64(K1[ci,cj])+base*np.uint64(SP[i]+SQ[j])
            for z in range(op[i],op[i+1]):
                k=lp[z];numer=base*np.uint64(Q[cj,k])
                x+=((numer*R[int(P[ci,k])+1])>>np.uint64(32))*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z];numer=base*np.uint64(P[ci,k])
                x+=((numer*R[int(Q[cj,k])+1])>>np.uint64(32))*np.uint64(vq[z])
            A[i,j]=x
    return A

def analytic_e1(a,b,h,n,mod,dtype,compact=True,recip=True):
    if mod is not None:raise ValueError('This optimized seed wrapper is for the exact worker')
    ip,P=ms.profiles(a,h);iq,Q=ms.profiles(b,h)
    if compact:
        if old.shell_one_bound(n)>=(1<<32):raise ValueError('32-bit analytic kernel certificate unavailable')
        k=(a+b-2*h)//2
        dt=np.uint16 if math.comb(k,k//2)<(1<<16) else np.uint32
        K0,K1=compact_profile_kernels(P,Q,old.choose(n),np.empty(0,dt))
    else:K0,K1=old.profile_kernels(P,Q,old.choose(n))
    fp=old.features(a,h);fq=old.features(b,h)
    if recip:A=expand_recip(K0,K1,ip,iq,P,Q,*fp,*fq,reciprocals(n),np.empty(0,dtype))
    else:A=old.expand_e1(K0,K1,ip,iq,P,Q,*fp,*fq,0,np.empty(0,dtype))
    return A,2*K0.size

# Tiled scatter rows: parallelism is across disjoint column intervals.
@nb.njit(cache=True,parallel=True)
def scatter_rows_tiled(T,maps,V):
    nr,nc=V.shape
    tile=128
    for chunk in nb.prange((nc+tile-1)//tile):
        lo=chunk*tile;hi=min(nc,lo+tile)
        for i in range(nr):
            row=maps[i]
            for j in range(lo,hi):T[row,j]+=V[i,j]

@nb.njit(cache=True)
def scatter_rows_rowmajor(T,maps,V):
    for i in range(V.shape[0]):
        row=maps[i]
        for j in range(V.shape[1]):T[row,j]+=V[i,j]

@nb.njit(cache=True,parallel=True)
def scatter_cols_transposed_tiled(T,maps,V):
    nr,nc=V.shape;tile=128
    for chunk in nb.prange((nr+tile-1)//tile):
        lo=chunk*tile;hi=min(nr,lo+tile)
        for j in range(nc):
            col=maps[j]
            for i in range(lo,hi):T[i,col]+=V[i,j]
