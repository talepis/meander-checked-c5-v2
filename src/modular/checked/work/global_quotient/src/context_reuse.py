"""Bounded reuse of exact profile contexts, built during each timed run."""
from collections import OrderedDict
from functools import lru_cache
import numpy as np
import numba as nb
import profile_contexts as C
import math
import allocation_guard as guard

LIMIT=64<<20
CACHE=OrderedDict();BYTES=0;STATS={}
OLD_BUILD=C.build;OLD_FAMILY=C.family

def clear():
    global BYTES
    CACHE.clear();BYTES=0;family.cache_clear()
    STATS.clear();STATS.update(hits=0,misses=0,computed_entries=0,peak_bytes=0)

@nb.njit(cache=True)
def family_step(codes,totals,k):
    N=0
    for x in totals:N+=k-int(x)+1
    cc=np.empty(N,np.uint64);tt=np.empty(N,np.int16);pp=np.empty(N,np.int32);rr=np.empty(N,np.int16)
    z=0
    for i in range(len(codes)):
        for r in range(k-int(totals[i])+1):
            cc[z]=(codes[i]<<np.uint64(r+1))|((np.uint64(1)<<np.uint64(r))-np.uint64(1))
            tt[z]=totals[i]+r;pp[z]=i;rr[z]=r;z+=1
    return cc,tt,pp,rr

@lru_cache(None)
def family(k,levels):
    if levels<=1:return OLD_FAMILY(k,levels)
    guard.check(math.comb(k+levels,levels)*64,('context-family',k,levels))
    cc,tt,_,_=family(k,levels-1)
    values=family_step(cc,tt,k);order=np.argsort(values[0])
    return tuple(x[order] for x in values)

def context(kp,kq,h,choose):
    global BYTES
    if h==0:return np.array([0,1,0,0],np.uint32).reshape(4,1,1)
    key=(kp,kq,h)
    if key in CACHE:
        STATS['hits']+=1;CACHE.move_to_end(key);return CACHE[key]
    STATS['misses']+=1
    last=context(kp,kq,h-1,choose)
    _,_,ip,up=C.family(kp,h);_,_,iq,uq=C.family(kq,h)
    cur=C.advance(last,ip,up,iq,uq,choose)
    STATS['computed_entries']+=cur.size
    if cur.nbytes<=LIMIT:
        while CACHE and BYTES+cur.nbytes>LIMIT:
            _,v=CACHE.popitem(last=False);BYTES-=v.nbytes
        CACHE[key]=cur;BYTES+=cur.nbytes;STATS['peak_bytes']=max(STATS['peak_bytes'],BYTES)
    return cur

def build(kp,kq,h,pc,qc,choose):
    cap=4*(kp+kq+h+1)**2*(1<<(kp+kq))
    if cap>=1<<32:raise ArithmeticError('Context uint32 certificate unavailable')
    cur=context(kp,kq,h,choose);prev=context(kp,kq,max(0,h-1),choose)
    if h:
        pp=C.parent_codes(pc);qq=C.parent_codes(qc)
        ip=np.searchsorted(C.family(kp,h)[0],pp).astype(np.int32)
        iq=np.searchsorted(C.family(kq,h)[0],qq).astype(np.int32)
        if h>1:
            ipp=np.searchsorted(C.family(kp,h-1)[0],C.parent_codes(pp)).astype(np.int32)
            iqq=np.searchsorted(C.family(kq,h-1)[0],C.parent_codes(qq)).astype(np.int32)
        else:ipp=np.zeros(len(pc),np.int32);iqq=np.zeros(len(qc),np.int32)
    else:
        ip=ipp=np.zeros(len(pc),np.int32);iq=iqq=np.zeros(len(qc),np.int32)
    return cur,prev,ip,iq,ipp,iqq

def configure(mib=64,compiled=True):
    global LIMIT
    LIMIT=mib<<20
    if compiled:C.family=family
    C.build=build;clear()
