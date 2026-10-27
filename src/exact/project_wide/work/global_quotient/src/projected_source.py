"""Exact e=3 source via closing each target prefix once.
The only excess-one -> excess-three move is DD on two terminal closes.
Thus source3(P,Q) = V1(P+')', Q+')'), evaluated on smaller root profiles.
"""
from functools import lru_cache
import time
import numpy as np
import numba as nb
import incoming_e1 as I
import recursive_profiles as rp
import source_separation as S
import engine_v2 as E
import allocation_guard as guard
from profile_contexts import build

@lru_cache(None)
def closed(a,h):
    if h<=0:raise ValueError('A closing extension requires positive height')
    guard.check(E.mt.B(a,h)*64,('projected-prefix',a,h))
    codes,ids=rp.compressed_codes(a,h)
    newcodes,remap=np.unique(rp.close_codes(codes),return_inverse=True)
    newids=remap.astype(np.int32)[ids]
    P=E.ms.decode_profiles(newcodes,h-1)
    ss,rr=rp.feature_states(a,h)
    s2,r2=rp.step_features(np.empty(0,np.uint64),np.empty(0,np.uint64),ss,rr,codes,ids)
    return newids,P,(s2,*rp.sparse_rootcodes(r2,h-1)),newcodes

@nb.njit(cache=True,parallel=True)
def leaf_kernel(P,Q,C,ctx,ip,iq):
    K=np.empty((2,len(P),len(Q)),np.uint64);h=P.shape[1]-1
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            pi=ip[i];qj=iq[j]
            z,l,a,b=I.step(np.uint64(ctx[0,pi,qj]),np.uint64(ctx[1,pi,qj]),np.uint64(ctx[2,pi,qj]),np.uint64(ctx[3,pi,qj]),int(P[i,h]),int(Q[j,h]),C)
            K[0,i,j]=l;K[1,i,j]=z
    return K

@nb.njit(cache=True,parallel=True)
def row(K,ids,P,Q,SS,off,lev,val):
    out=np.empty((len(ids),len(Q)),np.uint64)
    for i in nb.prange(len(ids)):
        ci=ids[i]
        for cj in range(len(Q)):
            base=np.uint64(K[0,ci,cj]);x=np.uint64(K[1,ci,cj])+base*np.uint64(SS[i])
            for z in range(off[i],off[i+1]):
                k=lev[z];x+=(base*np.uint64(Q[cj,k])//np.uint64(int(P[ci,k])+1))*np.uint64(val[z])
            out[i,cj]=x
    return out

@nb.njit(cache=True,parallel=True)
def col(K,ids,P,Q,SS,off,lev,val):
    out=np.empty((len(P),len(ids)),np.uint64)
    for ci in nb.prange(len(P)):
        for j in range(len(ids)):
            cj=ids[j];base=np.uint64(K[0,ci,cj]);x=base*np.uint64(SS[j])
            for z in range(off[j],off[j+1]):
                k=lev[z];x+=(base*np.uint64(P[ci,k])//np.uint64(int(Q[cj,k])+1))*np.uint64(val[z])
            out[ci,j]=x
    return out

@nb.njit(cache=True,parallel=True)
def direct(T,K,ip,iq,P,Q,sp,op,lp,vp,sq,oq,lq,vq):
    mxs=np.zeros(len(ip),np.uint64)
    for i in nb.prange(len(ip)):
        ci=ip[i];mx=np.uint64(0)
        for j in range(len(iq)):
            cj=iq[j];base=np.uint64(K[0,ci,cj])
            x=np.uint64(T[i,j])+np.uint64(K[1,ci,cj])+base*np.uint64(sp[i]+sq[j])
            for z in range(op[i],op[i+1]):
                k=lp[z];x+=(base*np.uint64(Q[cj,k])//np.uint64(int(P[ci,k])+1))*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z];x+=(base*np.uint64(P[ci,k])//np.uint64(int(Q[cj,k])+1))*np.uint64(vq[z])
            T[i,j]=x
            if x>mx:mx=x
        mxs[i]=mx
    m=np.uint64(0)
    for v in mxs:
        if v>m:m=v
    return m

ORIGINAL_APPLY=S.apply
COUNT={}

def clear():
    closed.cache_clear();COUNT.clear();COUNT.update(blocks=0,old_profile_entries=0,new_profile_entries=0,old_factor_entries=0,new_factor_entries=0)

def data(n,key):
    a,b,h=key
    if I.source_certificate(key,3)>=1<<64:raise ArithmeticError('Source3 uint64 certificate unavailable')
    ip,P,fp,pc=closed(a,h);iq,Q,fq,qc=closed(b,h)
    guard.array((2,len(P),len(Q)),np.uint64,('projected-kernel',key))
    context,prev,pp,qq,_,_=build((a-h)//2,(b-h)//2,h-1,pc,qc,E.old.choose(n))
    guard.array((2,len(P),len(Q)),np.uint64,('projected-kernel-after-context',key))
    K=leaf_kernel(P,Q,E.old.choose(n),context,pp,qq)
    return K,ip,iq,P,Q,fp,fq

def apply(n,key,e,T,parents=None):
    if e!=3:return ORIGINAL_APPLY(n,key,e,T,parents)
    assert parents is None
    t=time.perf_counter();K,ip,iq,P,Q,fp,fq=data(n,key)
    S.STATS['prepare_seconds']+=time.perf_counter()-t
    oldp=len(E.ms.profiles(key[0],key[2])[1]);oldq=len(E.ms.profiles(key[1],key[2])[1])
    COUNT['blocks']+=1;COUNT['old_profile_entries']+=oldp*oldq;COUNT['new_profile_entries']+=len(P)*len(Q)
    COUNT['old_factor_entries']+=len(ip)*oldq+oldp*len(iq);COUNT['new_factor_entries']+=len(ip)*len(Q)+len(P)*len(iq)
    size=len(ip)*len(iq);small=len(ip)*len(Q)+len(P)*len(iq)
    t=time.perf_counter()
    if size>=S.MIN_SIZE and small<size*S.RATIO:
        guard.check(small*8,('projected-row-column',key))
        R=row(K,ip,P,Q,*fp)
        C=S.reflected_column(R,K[1],ip) if key[0]==key[1] else col(K,iq,P,Q,*fq)
        S.STATS['separate_seconds']+=time.perf_counter()-t
        S.STATS['one_sided_evaluations']+=R.size+(0 if key[0]==key[1] else C.size)
        if key[0]==key[1]:S.STATS['reflected_copy_entries']+=C.size
        S.STATS['separated_entries']+=small
        S.STATS['peak_temporary_bytes']=max(S.STATS['peak_temporary_bytes'],R.nbytes+C.nbytes+K.nbytes)
        t=time.perf_counter();mx=S.add_separated(T,R,C,ip,iq)
    else:
        S.STATS['direct_evaluations']+=size
        mx=direct(T,K,ip,iq,P,Q,*fp,*fq)
    S.STATS['full_entries']+=size;S.STATS['apply_seconds']+=time.perf_counter()-t
    return int(mx)

def configure():S.apply=apply;clear()
