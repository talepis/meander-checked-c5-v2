"""Closed source terms from the eliminated excess-zero shell.

For target root profiles u,v at common height h, B_i=binom(u_i+v_i,u_i),
L=prod_{i<h} B_i, Lp=prod_{i<h-1} B_i:
 S1 = L*(binom(u_h+v_h,u_h+1)+binom(u_h+v_h,v_h+1))
      + [h>0,u_h=0]*Lp*binom(u_{h-1}+v_{h-1}+1,u_{h-1})
      + [h>0,v_h=0]*Lp*binom(u_{h-1}+v_{h-1}+1,v_{h-1}).
 S2 = [h>0]*Lp*binom(u_{h-1}+v_{h-1}+2,u_{h-1}+1).
S1 injects at excess 1; S2 at excess 2. These are source terms, NOT V1/V2.
"""
from functools import lru_cache
import numpy as np
import numba as nb
import engine_v2 as E

@nb.njit(cache=True,parallel=True)
def source_kernel(P,Q,C,e,example):
    K=np.empty((len(P),len(Q)),example.dtype)
    h=P.shape[1]-1
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            L=np.uint64(1);Lp=np.uint64(1)
            for k in range(h):
                if k==h-1:Lp=L
                u=int(P[i,k]);v=int(Q[j,k]);L*=C[u+v,u]
            if e==1:
                u=int(P[i,h]);v=int(Q[j,h])
                x=L*(C[u+v,u+1]+C[u+v,v+1])
                if h:
                    p=int(P[i,h-1]);q=int(Q[j,h-1])
                    if u==0:x+=Lp*C[p+q+1,p]
                    if v==0:x+=Lp*C[p+q+1,q]
            else:
                x=np.uint64(0)
                if h:
                    p=int(P[i,h-1]);q=int(Q[j,h-1]);x=Lp*C[p+q+2,p+1]
            K[i,j]=x
    return K

@nb.njit(cache=True)
def raw_step(U,D):
    R=np.empty(len(U)+len(D),np.uint32)
    for i in range(len(U)):R[i]=U[i]<<np.uint32(1)
    for i in range(len(D)):
        x=D[i]
        while x & np.uint32(1):x>>=np.uint32(1)
        R[len(U)+i]=x|np.uint32(1)
    return R

@lru_cache(None)
def raw_codes(m,h):
    if m<0 or h<0 or m<h or (m-h)%2:return np.empty(0,np.uint32)
    if m>30:raise ValueError('raw profile codes require m<=30')
    if m==0:return np.array([1],np.uint32)
    return raw_step(raw_codes(m-1,h-1),raw_codes(m-1,h+1))

@nb.njit(cache=True)
def one_sided(codes,h,e):
    R=np.empty(len(codes),np.uint32)
    for j in range(len(codes)):
        c=codes[j];u=0;v=0
        while c>1 and c&np.uint32(1):u+=1;c>>=np.uint32(1)
        c>>=np.uint32(1)
        while c>1 and c&np.uint32(1):v+=1;c>>=np.uint32(1)
        if e==1:
            x=u
            if h:
                x+=1
                if u==0:x+=v+1
        else:x=v+2 if h else 0
        R[j]=x
    return R

CLOSED_TWO=False

@lru_cache(None)
def closed_profiles(m,h):
    from recursive_profiles import compressed_codes,close_codes
    codes,ids=compressed_codes(m,h)
    uq,remap=np.unique(close_codes(codes),return_inverse=True)
    return remap[ids].astype(np.int32),E.ms.decode_profiles(uq,h-1)

PROFILE_SECONDS=FILL_SECONDS=ADD_SECONDS=0.0
KERNEL_ENTRIES=EXPANDED_ENTRIES=0

def inject(n,t,U,target_for):
    global PROFILE_SECONDS,FILL_SECONDS,ADD_SECONDS,KERNEL_ENTRIES,EXPANDED_ENTRIES
    import time
    for key in sorted(U):
        a,b,h=key;e=(t-a-b+h)//2
        if e not in (1,2) or (e==2 and h==0):continue
        start=time.perf_counter()
        if a==h:
            code=raw_codes(b,h);V=one_sided(code,h,e)
            PROFILE_SECONDS+=time.perf_counter()-start
            T=target_for(key);T[0,:]+=V.astype(T.dtype,copy=False)
            EXPANDED_ENTRIES+=len(V)
            continue
        fn=closed_profiles if e==2 and CLOSED_TWO else E.ms.profiles
        ip,P=fn(a,h);iq,Q=fn(b,h)
        PROFILE_SECONDS+=time.perf_counter()-start;start=time.perf_counter()
        dt=E.dtype_for(U[key],'mixed')
        if dt==object:dt=np.uint64
        if e==2 and CLOSED_TWO:
            K=np.empty((len(P),len(Q)),dt)
            E.ms.fill_kernel(K,P,Q,E.ms.choose_table(n),0)
        else:K=source_kernel(P,Q,E.old.choose(n),e,np.empty(0,dt))
        FILL_SECONDS+=time.perf_counter()-start;start=time.perf_counter()
        T=target_for(key)
        if T.dtype==object:T+=K[ip[:,None],iq[None,:]].astype(object)
        else:E.ms.gather_add(T,0,0,K,ip,iq,0,len(ip),0,len(iq))
        ADD_SECONDS+=time.perf_counter()-start
        KERNEL_ENTRIES+=K.size;EXPANDED_ENTRIES+=T.size

def clear():
    global PROFILE_SECONDS,FILL_SECONDS,ADD_SECONDS,KERNEL_ENTRIES,EXPANDED_ENTRIES
    global FUSED_ENTRIES,DEPENDENTS,PLANNED,DONE,FUSED_KEYS
    FUSED_ENTRIES=0;DEPENDENTS={};PLANNED={};DONE=set();FUSED_KEYS=set()
    raw_codes.cache_clear();closed_profiles.cache_clear();PROFILE_SECONDS=FILL_SECONDS=ADD_SECONDS=0.0
    KERNEL_ENTRIES=EXPANDED_ENTRIES=0

@nb.njit(cache=True,parallel=True)
def fused_first_shell(T,UU,X,Y,K,ip,iq,Up,Uq):
    """Three disjoint/overlapping zero-cost rectangles plus the local source.
    X is the predecessor of a target ending ')' on P; Y on Q.
    Every target cell is assigned once. Returns its exact maximum.
    """
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        rowmax=np.uint64(0);ki=ip[i]
        for j in range(T.shape[1]):
            x=np.uint64(K[ki,iq[j]])
            if i<Up and j<Uq:
                if UU.size:x+=UU[i,j]
            else:
                if i>=Up and X.size:x+=X[i-Up,j]
                if j>=Uq and Y.size:x+=Y[i,j-Uq]
            T[i,j]=x
            if x>rowmax:rowmax=x
        maxima[i]=rowmax
    mx=np.uint64(0)
    for r in range(len(maxima)):
        if maxima[r]>mx:mx=maxima[r]
    return mx

FUSED_KEYS=set()
FUSED_ENTRIES=0
TIGHT=False

def prestage(n,t,cur,U,target_for,previous_maxima):
    global FUSED_KEYS,FUSED_ENTRIES
    FUSED_KEYS=set();maxima={}
    for key in sorted(U):
        a,b,h=key;e=(t+1-a-b+h)//2
        if e!=1 or a==h:continue
        Na,Nb=E.mt.B(a,h),E.mt.B(b,h)
        if Na*Nb<4096:continue
        ip,P=E.ms.profiles(a,h);iq,Q=E.ms.profiles(b,h)
        dt=E.dtype_for(U[key],'mixed')
        if dt==object:continue
        K=source_kernel(P,Q,E.old.choose(n),1,np.empty(0,dt))
        empty=np.empty((0,0),dt)
        def source(a,b,h):
            if a>b:
                V=cur.get((b,a,h));return empty if V is None else V.T
            return cur.get((a,b,h),empty)
        UU=source(a-1,b-1,h-1)
        X=source(a-1,b+1,h+1)
        Y=source(a+1,b-1,h+1)
        bound=None
        if TIGHT:
            def mxof(a,b,h):return previous_maxima.get((min(a,b),max(a,b),h),0)
            bound=max(mxof(a-1,b-1,h-1),mxof(a-1,b+1,h+1)+mxof(a+1,b-1,h+1))+int(K.max(initial=0))
        T=target_for(key,False,bound)
        mx=fused_first_shell(T,UU,X,Y,K,ip,iq,E.mt.B(a-1,h-1),E.mt.B(b-1,h-1))
        maxima[key]=int(mx);FUSED_KEYS.add(key);FUSED_ENTRIES+=T.size
    return maxima

_original_inject=inject

def inject_skip_fused(n,t,U,target_for):
    filtered={k:u for k,u in U.items() if k not in FUSED_KEYS}
    return _original_inject(n,t,filtered,target_for)

DEPENDENTS={}
PLANNED={}
DONE=set()

def prepare_release(n,t,cur,U,target_for,previous_maxima):
    """Schedule first-shell fusion just before a required predecessor is freed."""
    global DEPENDENTS,PLANNED,DONE,FUSED_KEYS
    DEPENDENTS={};PLANNED={};DONE=set();FUSED_KEYS=set()
    for key in U:
        a,b,h=key
        if t+1-a-b+h!=2 or a==h or E.mt.B(a,h)*E.mt.B(b,h)<4096:continue
        if E.dtype_for(U[key],'mixed')==object:continue
        parents=[]
        for aa,bb,hh in [(a-1,b-1,h-1),(a-1,b+1,h+1),(a+1,b-1,h+1)]:
            p=(min(aa,bb),max(aa,bb),hh)
            if p in cur:parents.append(p)
        if not parents:continue
        PLANNED[key]=parents;FUSED_KEYS.add(key)
        for p in set(parents):DEPENDENTS.setdefault(p,[]).append(key)
    return {}

def fuse_on_release(n,t,parent,cur,U,target_for,previous_maxima,maxima):
    global FUSED_ENTRIES
    for key in DEPENDENTS.get(parent,[]):
        if key in DONE:continue
        a,b,h=key
        ip,P=E.ms.profiles(a,h);iq,Q=E.ms.profiles(b,h)
        dt=E.dtype_for(U[key],'mixed');K=source_kernel(P,Q,E.old.choose(n),1,np.empty(0,dt))
        empty=np.empty((0,0),dt)
        def source(aa,bb,hh):
            p=(min(aa,bb),max(aa,bb),hh)
            if p not in previous_maxima:return empty
            if p not in cur:raise AssertionError(('fusion parent freed too early',key,p))
            V=cur[p]
            return V.T if aa>bb else V
        UU=source(a-1,b-1,h-1);X=source(a-1,b+1,h+1);Y=source(a+1,b-1,h+1)
        T=target_for(key,False)
        mx=fused_first_shell(T,UU,X,Y,K,ip,iq,E.mt.B(a-1,h-1),E.mt.B(b-1,h-1))
        maxima[key]=int(mx);DONE.add(key);FUSED_ENTRIES+=T.size
