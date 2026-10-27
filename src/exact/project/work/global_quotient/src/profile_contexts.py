"""Experiment: shared prefix contexts for root-profile pair evaluations."""
from functools import lru_cache
import math
import numpy as np
import numba as nb
from incoming_e1 import step
import allocation_guard as guard

@lru_cache(None)
def family(k,levels):
    if levels==0:
        return np.array([1],np.uint64),np.array([0],np.int16),np.array([0],np.int32),np.array([0],np.int16)
    if levels==1:
        return np.array([(1<<(c+1))-1 for c in range(k+1)],np.uint64),np.arange(k+1,dtype=np.int16),np.zeros(k+1,np.int32),np.arange(k+1,dtype=np.int16)
    codes,totals,_,_=family(k,levels-1)
    rows=[]
    for p,(code,count) in enumerate(zip(codes,totals)):
        for c in range(k-int(count)+1):rows.append(((int(code)<<(c+1))|((1<<c)-1),int(count)+c,p,c))
    rows.sort()
    return tuple(np.array([row[i] for row in rows],dtype=dt) for i,dt in enumerate((np.uint64,np.int16,np.int32,np.int16)))

@nb.njit(cache=True,parallel=True)
def advance(V,ip,up,iq,uq,C):
    out=np.empty((4,len(ip),len(iq)),np.uint32)
    for i in nb.prange(len(ip)):
        for j in range(len(iq)):
            p=ip[i];q=iq[j]
            z,l,a,b=step(np.uint64(V[0,p,q]),np.uint64(V[1,p,q]),np.uint64(V[2,p,q]),np.uint64(V[3,p,q]),int(up[i]),int(uq[j]),C)
            out[0,i,j]=z;out[1,i,j]=l;out[2,i,j]=a;out[3,i,j]=b
    return out

@nb.njit(cache=True)
def parent_codes(codes):
    ans=np.empty(len(codes),np.uint64)
    for i in range(len(codes)):
        c=codes[i]
        while c>1 and c & np.uint64(1):c>>=np.uint64(1)
        ans[i]=c>>np.uint64(1)
    return ans

def build(kp,kq,h,pc,qc,C):
    cap=4*(kp+kq+h+1)**2*(1<<(kp+kq))
    if cap>=1<<32:raise ValueError('Context uint32 certificate unavailable')
    cur=np.array([0,1,0,0],np.uint32).reshape(4,1,1);prev=cur
    for level in range(1,h+1):
        _,_,ip,up=family(kp,level);_,_,iq,uq=family(kq,level)
        guard.array((4,len(ip),len(iq)),np.uint32,('context-advance',kp,kq,level))
        prev,cur=cur,advance(cur,ip,up,iq,uq,C)
    if h:
        pp=parent_codes(pc);qq=parent_codes(qc)
        ip=np.searchsorted(family(kp,h)[0],pp).astype(np.int32)
        iq=np.searchsorted(family(kq,h)[0],qq).astype(np.int32)
        if h>1:
            ipp=np.searchsorted(family(kp,h-1)[0],parent_codes(pp)).astype(np.int32)
            iqq=np.searchsorted(family(kq,h-1)[0],parent_codes(qq)).astype(np.int32)
        else:ipp=np.zeros(len(pc),np.int32);iqq=np.zeros(len(qc),np.int32)
    else:
        ip=ipp=np.zeros(len(pc),np.int32);iq=iqq=np.zeros(len(qc),np.int32)
    return cur,prev,ip,iq,ipp,iqq
