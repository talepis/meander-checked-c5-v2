"""Exact source-only feature equivalence; NOT an equivalence of frontier states."""
from functools import lru_cache
import numpy as np
import numba as nb
import engine_v2 as E
import incoming_e1 as I
import recursive_profiles as rp

@nb.njit(cache=True)
def signatures(ip,P,S,off,lev,val,codes,e):
    h=P.shape[1]-1
    out=np.zeros((len(ip),h+4),np.uint32)
    for i in range(len(ip)):
        out[i,0]=ip[i];out[i,1]=S[i]
        for z in range(off[i],off[i+1]):out[i,2+lev[z]]=val[z]
        if e==2:
            x=codes[i];shift=0
            for _ in range(P[ip[i],h]):
                x>>=np.uint32(1);shift+=1
                while x>1 and x&np.uint32(1):x>>=np.uint32(1);shift+=1
            out[i,h+3]=(np.uint32(1)<<np.uint32(shift))|(codes[i]&((np.uint32(1)<<np.uint32(shift))-np.uint32(1)))
    return out

@nb.njit(cache=True)
def take_features(S,off,lev,val,reps):
    count=0
    for i in reps:count+=off[i+1]-off[i]
    ss=np.empty(len(reps),S.dtype);oo=np.zeros(len(reps)+1,np.int64)
    ll=np.empty(count,lev.dtype);vv=np.empty(count,val.dtype);j=0
    for k in range(len(reps)):
        i=reps[k];ss[k]=S[i]
        for z in range(off[i],off[i+1]):ll[j]=lev[z];vv[j]=val[z];j+=1
        oo[k+1]=j
    return ss,oo,ll,vv

@lru_cache(None)
def groups(a,h,e):
    ip,P=E.ms.profiles(a,h);fp=rp.features(a,h)
    codes=I.degree_codes(a,h) if e==2 else np.empty(0,np.uint32)
    tab=signatures(ip,P,*fp,codes,e)
    _,ix,inv=np.unique(tab,axis=0,return_index=True,return_inverse=True)
    return ix.astype(np.int32),inv.astype(np.int32),take_features(*fp,ix)

@nb.njit(cache=True,parallel=True)
def add_grouped(T,R,C,ip,iq,gp,gq):
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        pi=gp[i];ci=ip[i];mx=np.uint64(0)
        for j in range(T.shape[1]):
            x=np.uint64(T[i,j])+np.uint64(R[pi,iq[j]])+np.uint64(C[ci,gq[j]])
            T[i,j]=x
            if x>mx:mx=x
        maxima[i]=mx
    result=np.uint64(0)
    for k in range(len(maxima)):
        if maxima[k]>result:result=maxima[k]
    return result

@nb.njit(cache=True,parallel=True)
def fuse_grouped(T,R,C,ip,iq,gp,gq,UU,X,Y,Up,Uq):
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];pi=gp[i];mx=np.uint64(0)
        for j in range(T.shape[1]):
            x=np.uint64(R[pi,iq[j]])+np.uint64(C[ci,gq[j]])
            if i<Up and j<Uq:
                if UU.size:x+=np.uint64(UU[i,j])
            else:
                if i>=Up and X.size:x+=np.uint64(X[i-Up,j])
                if j>=Uq and Y.size:x+=np.uint64(Y[i,j-Uq])
            T[i,j]=x
            if x>mx:mx=x
        maxima[i]=mx
    result=np.uint64(0)
    for k in range(len(maxima)):
        if maxima[k]>result:result=maxima[k]
    return result

@nb.njit(cache=True)
def packed_signatures(ip,P,S,off,lev,val,codes,e,sbits,rbits,dbits):
    h=P.shape[1]-1;out=np.empty(len(ip),np.uint64)
    for i in range(len(ip)):
        rc=np.uint64(1);z=off[i]
        for level in range(h+1):
            d=0
            if z<off[i+1] and lev[z]==level:d=int(val[z]);z+=1
            rc=(rc<<np.uint64(d+1))|((np.uint64(1)<<np.uint64(d))-np.uint64(1))
        dc=np.uint64(0)
        if e==2:
            x=codes[i];shift=0
            for _ in range(P[ip[i],h]):
                x>>=np.uint32(1);shift+=1
                while x>1 and x&np.uint32(1):x>>=np.uint32(1);shift+=1
            dc=(np.uint64(1)<<np.uint64(shift))|(np.uint64(codes[i])&((np.uint64(1)<<np.uint64(shift))-np.uint64(1)))
        x=(np.uint64(ip[i])<<np.uint64(sbits))|np.uint64(S[i])
        x=(x<<np.uint64(rbits))|rc
        out[i]=(x<<np.uint64(dbits))|dc
    return out

PACKED=False
old_groups=groups
@lru_cache(None)
def fast_groups(a,h,e):
    ip,P=E.ms.profiles(a,h);fp=rp.features(a,h);k=(a-h)//2
    sbits=max(1,(k*(k+3)).bit_length());rbits=h+k+2;dbits=(k+1 if e==2 else 0)
    pbits=max(1,(len(P)-1).bit_length())
    if pbits+sbits+rbits+dbits>64:raise ArithmeticError('No lossless uint64 source-signature certificate')
    codes=I.degree_codes(a,h) if e==2 else np.empty(0,np.uint32)
    tab=packed_signatures(ip,P,*fp,codes,e,sbits,rbits,dbits)
    _,ix,inv=np.unique(tab,return_index=True,return_inverse=True)
    return ix.astype(np.int32),inv.astype(np.int32),take_features(*fp,ix)

def selected_groups(a,h,e):
    return fast_groups(a,h,e) if PACKED else old_groups(a,h,e)
def clear_groups():old_groups.cache_clear();fast_groups.cache_clear()
selected_groups.cache_clear=clear_groups
groups=selected_groups
