"""Exact source-feature transducer. Not a quotient of frontier states.

A rich signature retains (root counts by level, ordered degrees of all roots,
sigma minus number of roots). It is closed under both ballot extensions.
Only DISTINCT predecessor signatures are transformed and sorted.
"""
from functools import lru_cache
import numpy as np
import numba as nb
import engine_v2 as E
import source_signatures as G
import recursive_profiles as rp
import incoming_e1 as I


def widths(a,h):
    k=(a-h)//2
    return max(1,(k*(k+3)).bit_length()), k+1

@nb.njit(cache=True)
def extend_codes(codes, sbits, dbits, new_sbits, new_dbits, opening):
    out=np.empty(len(codes),np.uint64)
    smask=(np.uint64(1)<<np.uint64(sbits))-np.uint64(1)
    dmask=(np.uint64(1)<<np.uint64(dbits))-np.uint64(1)
    for i in range(len(codes)):
        code=codes[i];s=code&smask;deg=(code>>np.uint64(sbits))&dmask
        p=code>>np.uint64(sbits+dbits)
        if opening:
            p<<=np.uint64(1)
        else:
            d=0
            while p & np.uint64(1):
                p>>=np.uint64(1);d+=1
            p|=np.uint64(1)
            for _ in range(d):
                deg>>=np.uint64(1)
                while deg>1 and deg&np.uint64(1):deg>>=np.uint64(1)
            deg=(deg<<np.uint64(d+1))|((np.uint64(1)<<np.uint64(d+1))-np.uint64(2))
            s+=np.uint64(d*(d+3))
        out[i]=(p<<np.uint64(new_dbits+new_sbits))|(deg<<np.uint64(new_sbits))|s
    return out

EMPTY=(np.empty(0,np.uint64),np.empty(0,np.int32),np.empty(0,np.int32))
@lru_cache(None)
def rich(a,h):
    if a<0 or h<0 or h>a or ((a-h)&1):return EMPTY
    if a>30:raise ValueError('This transducer is certified for ballot length <=30')
    if not a:
        sb,db=widths(0,0)
        return np.array([(1<<(sb+db))|(1<<sb)],np.uint64),np.array([0],np.int32),np.array([0],np.int32)
    uc,ui,ur=rich(a-1,h-1);dc,di,dr=rich(a-1,h+1)
    sb,db=widths(a,h)
    if a+2+sb>64:raise ArithmeticError('Signature would exceed 64 bits')
    up=extend_codes(uc,*widths(a-1,h-1),sb,db,True) if len(uc) else np.empty(0,np.uint64)
    down=extend_codes(dc,*widths(a-1,h+1),sb,db,False) if len(dc) else np.empty(0,np.uint64)
    candidates=np.concatenate((up,down))
    code,first,remap=np.unique(candidates,return_index=True,return_inverse=True)
    ids=rp.propagate_ids(ui,di,remap.astype(np.int32),len(uc))
    reps=np.concatenate((ur,dr+len(ui)))[first]
    return code,ids,reps

@lru_cache(None)
def groups(a,h,e):
    code,ids,reps=rich(a,h)
    ip,P=E.ms.profiles(a,h);fp=rp.features(a,h)
    shortfp=G.take_features(*fp,reps)
    degrees=I.degree_codes(a,h)[reps] if e==2 else np.empty(0,np.uint32)
    k=(a-h)//2
    sbits=max(1,(k*(k+3)).bit_length());rbits=h+k+2;dbits=k+1 if e==2 else 0
    if max(1,(len(P)-1).bit_length())+sbits+rbits+dbits>64:raise ArithmeticError('Source signature too wide')
    tab=G.packed_signatures(ip[reps],P,*shortfp,degrees,e,sbits,rbits,dbits)
    _,first,remap=np.unique(tab,return_index=True,return_inverse=True)
    rr=reps[first]
    inv=remap.astype(np.int32)[ids]
    return rr,inv,G.take_features(*fp,rr)

def clear():rich.cache_clear();groups.cache_clear()
