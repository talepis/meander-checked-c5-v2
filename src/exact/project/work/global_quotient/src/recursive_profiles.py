"""Exact ballot-profile transducer; avoid unranking every word and sorting all words.
Ranks split by final '(' / ')'. Root profiles update deterministically under each.
"""
from functools import lru_cache
import numpy as np
import numba as nb
import engine_v2 as ae
import allocation_guard as guard

@nb.njit(cache=True)
def close_codes(codes):
    out=np.empty(len(codes),np.uint64)
    for i in range(len(codes)):
        x=codes[i]
        while x & np.uint64(1):x>>=np.uint64(1)
        out[i]=x|np.uint64(1)
    return out

@nb.njit(cache=True)
def propagate_ids(upids,downids,remap,offset):
    ans=np.empty(len(upids)+len(downids),np.int32)
    for i in range(len(upids)):ans[i]=remap[upids[i]]
    for i in range(len(downids)):ans[len(upids)+i]=remap[offset+downids[i]]
    return ans

EMPTY=(np.empty(0,np.uint64),np.empty(0,np.int32))
@lru_cache(None)
def compressed_codes(a,h):
    if a<0 or h<0 or h>a or ((a-h)&1):return EMPTY
    if a==0:return np.array([1],np.uint64),np.array([0],np.int32)
    guard.check(ae.mt.B(a,h)*64,('compressed-prefix',a,h))
    uc,ui=compressed_codes(a-1,h-1)
    dc,di=compressed_codes(a-1,h+1)
    candidates=np.concatenate((uc<<np.uint64(1),close_codes(dc)))
    unique,remap=np.unique(candidates,return_inverse=True)
    ids=propagate_ids(ui,di,remap.astype(np.int32),len(uc))
    return unique,ids

@lru_cache(None)
def profiles(a,h):
    codes,ids=compressed_codes(a,h)
    return ids,ae.ms.decode_profiles(codes,h)

def clear():
    profiles.cache_clear();compressed_codes.cache_clear()

def constant_inject(key,ops,target_for,n):
    a,b,h=key
    if a!=h and b!=h:return False
    for dest,kind,flip,rect,off,mp in ops:
        T=target_for(dest);r0,r1,c0,c1=rect
        if kind=='add':
            tr,tc=off;T[tr:tr+r1-r0,tc:tc+c1-c0]+=1
        elif kind=='cols':
            counts=np.bincount(ae.mt.close_map(*mp),minlength=T.shape[1]).astype(T.dtype)
            T[:r1-r0,:]+=counts[None,:]
        else:
            counts=np.bincount(ae.mt.close_map(*mp),minlength=T.shape[0]).astype(T.dtype)
            T[:,:c1-c0]+=counts[:,None]
    return True

@nb.njit(cache=True)
def step_features(Su,Ru,Sd,Rd,downcodes,downids):
    N=len(Su)+len(Sd)
    S=np.empty(N,np.uint32);R=np.empty(N,np.uint64)
    for i in range(len(Su)):
        S[i]=Su[i];R[i]=Ru[i]<<np.uint64(1)
    for i in range(len(Sd)):
        c=downcodes[downids[i]];d=0
        while c & np.uint64(1):c>>=np.uint64(1);d+=1
        r=Rd[i]
        while r & np.uint64(1):r>>=np.uint64(1)
        r>>=np.uint64(1)
        R[len(Su)+i]=(r<<np.uint64(d))|((np.uint64(1)<<np.uint64(d))-np.uint64(1))
        S[len(Su)+i]=Sd[i]+d*(d+3)
    return S,R

@lru_cache(None)
def feature_states(a,h):
    if a<0 or h<0 or h>a or ((a-h)&1):return np.empty(0,np.uint32),np.empty(0,np.uint64)
    if a==0:return np.array([0],np.uint32),np.array([1],np.uint64)
    guard.check(ae.mt.B(a,h)*12,('feature-states',a,h))
    Su,Ru=feature_states(a-1,h-1);Sd,Rd=feature_states(a-1,h+1)
    dc,di=compressed_codes(a-1,h+1)
    return step_features(Su,Ru,Sd,Rd,dc,di)

@nb.njit(cache=True)
def sparse_rootcodes(R,H):
    off=np.zeros(len(R)+1,np.int64)
    for i in range(len(R)):
        x=R[i];last=0
        while x>1:
            bit=int(x & np.uint64(1))
            if bit and not last:off[i+1]+=1
            last=bit;x>>=np.uint64(1)
    for i in range(len(R)):off[i+1]+=off[i]
    lev=np.empty(off[-1],np.int16);val=np.empty(off[-1],np.int16)
    for i in range(len(R)):
        x=R[i];h=H;z=off[i];count=0
        # Emit levels in increasing order to match the previous interface.
        for k in range(H,-1,-1):
            count=0
            while x>1 and x & np.uint64(1):count+=1;x>>=np.uint64(1)
            if count:
                lev[z]=k;val[z]=count;z+=1
            x>>=np.uint64(1)
        lo=off[i];hi=off[i+1]-1
        while lo<hi:
            v=lev[lo];lev[lo]=lev[hi];lev[hi]=v
            v=val[lo];val[lo]=val[hi];val[hi]=v
            lo+=1;hi-=1
    return off,lev,val

@lru_cache(None)
def features(a,h):
    S,R=feature_states(a,h)
    guard.check(len(R)*(8+4*(h+1))+8,('sparse-features',a,h))
    return S,*sparse_rootcodes(R,h)

def clear_all():
    clear();feature_states.cache_clear();features.cache_clear()
