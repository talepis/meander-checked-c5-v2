import math, time, gc, os, sys, ctypes
import numpy as np
from scatter_kernels import scatter_cols_serial, scatter_rows_serial, scatter_cols_parallel, scatter_rows_parallel
from functools import lru_cache

@lru_cache(None)
def B(m,h):
    if m<0 or h<0 or h>m or ((m-h)&1): return 0
    k=(m-h)//2
    return math.comb(m,k)-(math.comb(m,k-1) if k else 0)

def unrank_bits(m,h,r):
    bits=[0]*m
    for pos in range(m-1,-1,-1):
        u=B(m-1,h-1)
        if r<u:
            bits[pos]=1; h-=1
        else:
            bits[pos]=0; r-=u; h+=1
        m-=1
    return bits

def rank_bits(bits):
    h=sum(1 if x else -1 for x in bits)
    r=0; hh=h
    for m in range(len(bits),0,-1):
        if bits[m-1]: hh-=1
        else:
            r += B(m-1,hh-1); hh+=1
    return h,r

_close={}
def close_map(m,H):
    key=(m,H)
    if key in _close: return _close[key]
    N=B(m,H)
    arr=np.empty(N,dtype=np.int64)
    for r in range(N):
        bits=unrank_bits(m,H,r)
        st=[]
        for i,x in enumerate(bits):
            if x: st.append(i)
            else: st.pop()
        bits[st[-1]]=0
        hh,rr=rank_bits(bits)
        assert hh==H-2
        arr[r]=rr
    _close[key]=arr
    return arr

def full_keys(n,t):
    if t==0: return [(0,0,0)]
    L=2*n-t; par=t&1
    out=[]
    for a in range(par,L+1,2):
        for b in range(par,L+1,2):
            if a==b==0: continue
            lo=max(0,a+b-t)
            if (lo-par)&1: lo+=1
            for h in range(lo,min(a,b)+1,2):
                if B(a,h) and B(b,h): out.append((a,b,h))
    return out

def canon_keys(n,t):
    out=set()
    for a,b,h in full_keys(n,t):
        if a<=b: out.add((a,b,h))
        else: out.add((b,a,h))
    return out

def transfer_sym_lazy(n,current,t,mod=None,dtype=np.uint64):
    allowed=canon_keys(n,t+1)
    target={}
    def getT(key):
        a,b,h=key
        # Representative semantics: store only actual states with a<=b.
        # Contributions landing in a>b belong to the reflected state and are discarded;
        # they are recovered by processing the reflected source orientation.
        if a>b: return None
        ck=(a,b,h)
        if ck not in allowed: return None
        T=target.get(ck)
        if T is None:
            T=np.zeros((B(a,h),B(b,h)),dtype=dtype)
            target[ck]=T
        return T
    def add_block(key,rs,cs,V):
        T=getT(key)
        if T is not None: T[rs,cs] += V
    def add_scatter_cols(key,maps,V):
        T=getT(key)
        if T is not None:
            if V.size >= 2000000: scatter_cols_parallel(T,maps,V)
            else: scatter_cols_serial(T,maps,V)
    def add_scatter_rows(key,maps,V):
        T=getT(key)
        if T is not None:
            if V.size >= 2000000: scatter_rows_parallel(T,maps,V)
            else: scatter_rows_serial(T,maps,V)
    def process(a,b,h,A):
        Na,Nb=A.shape
        add_block((a+1,b+1,h+1),slice(0,Na),slice(0,Nb),A)
        if b>0:
            Uq=B(b-1,h-1); Dq=Nb-Uq
            if Uq:
                off=B(a,h-2)
                add_block((a+1,b-1,h-1),slice(off,off+Na),slice(0,Uq),A[:,:Uq])
            if Dq:
                add_block((a+1,b-1,h+1),slice(0,Na),slice(0,Dq),A[:,Uq:])
        if a>0:
            Up=B(a-1,h-1); Dp=Na-Up
            if Up:
                off=B(b,h-2)
                add_block((a-1,b+1,h-1),slice(0,Up),slice(off,off+Nb),A[:Up,:])
            if Dp:
                add_block((a-1,b+1,h+1),slice(0,Dp),slice(0,Nb),A[Up:,:])
        if a>0 and b>0:
            Up=B(a-1,h-1); Uq=B(b-1,h-1); Dp=Na-Up; Dq=Nb-Uq
            if Dp and Dq:
                add_block((a-1,b-1,h+1),slice(0,Dp),slice(0,Dq),A[Up:,Uq:])
            if Up and Dq and h>=1:
                add_scatter_cols((a-1,b-1,h-1),close_map(b-1,h+1),A[:Up,Uq:])
            if Dp and Uq and h>=1:
                add_scatter_rows((a-1,b-1,h-1),close_map(a-1,h+1),A[Up:,:Uq])
    for k in list(current.keys()):
        a,b,h=k
        A=current.pop(k)
        process(a,b,h,A)
        if a<b: process(b,a,h,A.T)
        del A
    maxpre=0
    if mod is not None:
        mm=dtype(mod)
        for A in target.values():
            if A.size:
                maxpre=max(maxpre,int(A.max(initial=0)))
                np.remainder(A,mm,out=A)
    return target,maxpre

def meander_sym(n,mod=None,verbose=False,dtype=np.uint64):
    cur={(0,0,0):np.array([[1]],dtype=dtype)}
    peak=0; peakt=0; maxpre=0
    t0=time.perf_counter()
    for t in range(0,2*n-1):
        cur,mp=transfer_sym_lazy(n,cur,t,mod,dtype)
        maxpre=max(maxpre,mp)
        stored=sum(A.size for A in cur.values())
        if stored>peak: peak,peakt=stored,t+1
        if verbose and ((t+1)%2==0 or t+1>=2*n-4):
            print(f't={t+1} stored={stored} GB={sum(A.nbytes for A in cur.values())/1e9:.3f}',flush=True)
    ans=int(cur[(1,1,1)][0,0])
    return ans,peak,peakt,time.perf_counter()-t0,maxpre

def crt_pairwise(residues, moduli):
    x=0; M=1
    for r,m in zip(residues,moduli):
        k=((r-x)%m)*pow(M,-1,m)%m
        x += M*k
        M *= m
    return x,M

def catalan(n):
    return math.comb(2*n,n)//(n+1)

def _trim_heap():
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass

def exact_meander(n, verbose=True):
    upper=catalan(n)**2
    # Natural uint64 is fastest and memory-safe here through n=23.
    if n<=23 and ((1<<64)*129999997)>upper:
        r64,*stats64=meander_sym(n,None,verbose,np.uint64)
        _trim_heap()
        rp,*statsp=meander_sym(n,129999997,verbose,np.uint32)
        _trim_heap()
        x,M=crt_pairwise([r64,rp],[1<<64,129999997])
        if not (x<upper<M):
            raise RuntimeError('CRT range check failed')
        return x, [(1<<64,r64,stats64),(129999997,rp,statsp)], upper, M
    # uint32 modular path. These ~27-bit primes are safe with layerwise
    # reduction while n+4 times the modulus stays below 2^32.
    candidates=[129999997,129999983,129999973,129999931,129999929,
                129999917,129999913,129999911]
    safe=[p for p in candidates if (n+4)*p < (1<<32)]
    mods=[]; P=1
    for p in safe:
        mods.append(p); P*=p
        if P>upper: break
    if P<=upper:
        raise RuntimeError('Need more safe CRT moduli for this n')
    residues=[]; runs=[]
    for p in mods:
        r,*stats=meander_sym(n,p,verbose,np.uint32)
        residues.append(r); runs.append((p,r,stats))
        _trim_heap()
    x,M=crt_pairwise(residues,mods)
    if not (x<upper<M):
        raise RuntimeError('CRT range check failed')
    return x,runs,upper,M

if __name__=='__main__':
    n=int(sys.argv[1])
    if len(sys.argv)>=3 and sys.argv[2]=='--exact':
        x,runs,upper,M=exact_meander(n,True)
        print({'n':n,'M_n':x,'runs':runs,'upper_bound':upper,'crt_modulus':M},flush=True)
    else:
        mod=None if len(sys.argv)<3 or sys.argv[2]=='none' else int(sys.argv[2])
        dtype=np.uint32 if mod is not None else np.uint64
        print(meander_sym(n,mod,True,dtype),flush=True)
