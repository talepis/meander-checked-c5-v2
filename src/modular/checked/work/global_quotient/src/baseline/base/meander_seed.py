"""Exact frontier enumeration with analytic minimal-cost (excess-zero) layers.
Derived here from the supplied manuscript's four transitions, without OEIS inputs.
Only the nonminimal layers are stored as dense matrices.
"""
import math,time,pickle,sys,os,json
from functools import lru_cache
import numpy as np
import numba as nb
import meander_turbo as mt
import allocation_guard as guard

@nb.njit(cache=True)
def profile_codes(m,H,N,Btab):
    codes=np.empty(N,np.uint64)
    bits=np.empty(m,np.uint8)
    stack=np.empty(m+1,np.int32)
    for r in range(N):
        h=H;rr=r
        for mm in range(m,0,-1):
            u=Btab[mm-1,h-1] if h>=1 else 0
            if rr<u:bits[mm-1]=1;h-=1
            else:bits[mm-1]=0;rr-=u;h+=1
        depth=0;stack[0]=0
        for pos in range(m):
            if bits[pos]:depth+=1;stack[depth]=0
            else:depth-=1;stack[depth]+=1
        code=np.uint64(1)
        for j in range(H+1):
            for k in range(stack[j]):code=(code<<np.uint64(1))|np.uint64(1)
            if j<H:code <<= np.uint64(1)
        codes[r]=code
    return codes

@nb.njit(cache=True)
def decode_profiles(codes,h):
    out=np.zeros((len(codes),h+1),np.uint8)
    for i in range(len(codes)):
        code=codes[i];j=h
        while code>1:
            if code & np.uint64(1):out[i,j]+=1
            else:j-=1
            code >>= np.uint64(1)
    return out

@lru_cache(None)
def profiles(a,h):
    guard.check(mt.B(a,h)*64,('original-prefix',a,h))
    tab=np.zeros((a+2,a+3),np.int64)
    for m in range(a+1):
        for v in range(a+2):tab[m,v]=mt.B(m,v)
    codes=profile_codes(a,h,mt.B(a,h),tab)
    unique,ids=np.unique(codes,return_inverse=True)
    return ids.astype(np.int32),decode_profiles(unique,h)

@nb.njit(cache=True,parallel=True)
def fill_kernel(K,P,Q,choose,mod):
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            v=np.uint64(1)
            for k in range(P.shape[1]):
                u=int(P[i,k]);w=int(Q[j,k]);v*=choose[u+w,u]
            if mod:v %= np.uint64(mod)
            K[i,j]=v

@nb.njit(cache=True,parallel=True)
def gather_add(T,r0,c0,K,ir,ic,rs,re,cs,ce):
    for i in nb.prange(re-rs):
        ki=ir[rs+i]
        for j in range(ce-cs):T[r0+i,c0+j]+=K[ki,ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def gather_cols(T,maps,K,ir,ic,rs,re,cs,ce):
    for i in nb.prange(re-rs):
        ki=ir[rs+i]
        for j in range(ce-cs):T[i,maps[j]]+=K[ki,ic[cs+j]]

@nb.njit(cache=True,parallel=True)
def gather_rows(T,maps,K,ir,ic,rs,re,cs,ce):
    for j in nb.prange(ce-cs):
        kj=ic[cs+j]
        for i in range(re-rs):T[maps[i],j]+=K[ir[rs+i],kj]

@lru_cache(None)
def choose_table(n):
    ar=np.zeros((n+2,n+2),np.uint64)
    for i in range(n+2):
        for j in range(i+1):ar[i,j]=math.comb(i,j)
    return ar

def transfer(n,current,t,mod,dtype):
    allowed=mt.canon_keys(n,t+1)
    target={}
    def getT(key):
        a,b,h=key
        if a>b or key not in allowed or a+b-h==t+1:return None
        T=target.get(key)
        if T is None:
            T=np.zeros((mt.B(a,h),mt.B(b,h)),dtype=dtype);target[key]=T
        return T
    def add(key,rs,cs,V):
        T=getT(key)
        if T is not None:T[rs,cs]+=V
    def sc(key,m,V):
        T=getT(key)
        if T is not None:
            (mt.scatter_cols_parallel if V.size>=2000000 else mt.scatter_cols_serial)(T,m,V)
    def sr(key,m,V):
        T=getT(key)
        if T is not None:
            (mt.scatter_rows_parallel if V.size>=2000000 else mt.scatter_rows_serial)(T,m,V)
    def dense(a,b,h,A):
        Na,Nb=A.shape
        add((a+1,b+1,h+1),slice(0,Na),slice(0,Nb),A)
        Up=mt.B(a-1,h-1);Uq=mt.B(b-1,h-1);Dp=Na-Up;Dq=Nb-Uq
        if b:
            if Uq:
                off=mt.B(a,h-2)
                add((a+1,b-1,h-1),slice(off,off+Na),slice(0,Uq),A[:,:Uq])
            if Dq:add((a+1,b-1,h+1),slice(0,Na),slice(0,Dq),A[:,Uq:])
        if a:
            if Up:
                off=mt.B(b,h-2)
                add((a-1,b+1,h-1),slice(0,Up),slice(off,off+Nb),A[:Up,:])
            if Dp:add((a-1,b+1,h+1),slice(0,Dp),slice(0,Nb),A[Up:,:])
        if a and b:
            if Dp and Dq:add((a-1,b-1,h+1),slice(0,Dp),slice(0,Dq),A[Up:,Uq:])
            if Up and Dq and h>=1:sc((a-1,b-1,h-1),mt.close_map(b-1,h+1),A[:Up,Uq:])
            if Dp and Uq and h>=1:sr((a-1,b-1,h-1),mt.close_map(a-1,h+1),A[Up:,:Uq])
    def seed(a,b,h,K,ir,ic):
        Na,Nb=len(ir),len(ic)
        Up=mt.B(a-1,h-1);Uq=mt.B(b-1,h-1);Dp=Na-Up;Dq=Nb-Uq
        def ga(key,rs,re,cs,ce):
            T=getT(key)
            if T is not None:gather_add(T,0,0,K,ir,ic,rs,re,cs,ce)
        if b and Dq:ga((a+1,b-1,h+1),0,Na,Uq,Nb)
        if a and Dp:ga((a-1,b+1,h+1),Up,Na,0,Nb)
        if a and b:
            if Dp and Dq:ga((a-1,b-1,h+1),Up,Na,Uq,Nb)
            if Up and Dq and h>=1:
                T=getT((a-1,b-1,h-1))
                if T is not None:gather_cols(T,mt.close_map(b-1,h+1),K,ir,ic,0,Up,Uq,Nb)
            if Dp and Uq and h>=1:
                T=getT((a-1,b-1,h-1))
                if T is not None:gather_rows(T,mt.close_map(a-1,h+1),K,ir,ic,Up,Na,0,Uq)
    # Free old dense blocks before creating seed kernels.
    for key in list(current):
        A=current[key]
        if A is None:continue
        current.pop(key);a,b,h=key
        dense(a,b,h,A)
        if a<b:dense(b,a,h,A.T)
        del A
    max_kernel=0
    for key in sorted(current):
        a,b,h=key
        if not a and not b:continue
        ir,P=profiles(a,h);ic,Q=profiles(b,h)
        K=np.empty((len(P),len(Q)),dtype=dtype)
        fill_kernel(K,P,Q,choose_table(n),0 if mod is None else mod)
        max_kernel=max(max_kernel,K.size)
        seed(a,b,h,K,ir,ic)
        if a<b:seed(b,a,h,K.T,ic,ir)
        del K
    maxpre=0
    if mod is not None:
        for A in target.values():
            maxpre=max(maxpre,int(A.max(initial=0)))
            np.remainder(A,dtype(mod),out=A)
    for a,b,h in allowed:
        if a+b-h==t+1:target[(a,b,h)]=None
    return target,maxpre,max_kernel

def meander(n,mod=129999997,verbose=False,dtype=np.uint32):
    if not 1<=n<=25:raise ValueError('This release is tested for 1 <= n <= 25.')
    # Covers conservative pre-reduction reverse indegree for the supplied close maps.
    D=4+2*((n-1)//2)
    if mod is not None and D*(mod-1)>np.iinfo(dtype).max:
        raise ValueError('Modulus too large for provably safe layerwise reduction.')
    cur={(0,0,0):None};peak=pt=mp=pk=0;t0=time.perf_counter()
    for t in range(2*n-1):
        cur,m,k=transfer(n,cur,t,mod,dtype)
        mp=max(mp,m);pk=max(pk,k)
        stored=sum(A.size for A in cur.values() if A is not None)
        if stored>peak:peak,pt=stored,t+1
        if verbose:print(json.dumps({'t':t+1,'dense':stored,'seed_kernel_max':k}),flush=True)
    ans=1 if n==1 else int(cur[(1,1,1)][0,0])
    return ans,peak,pt,time.perf_counter()-t0,mp,pk

if __name__=='__main__':
    n=int(sys.argv[1]);p=int(sys.argv[2]) if len(sys.argv)>2 else 129999997
    cache=sys.argv[3] if len(sys.argv)>3 else os.path.join(os.path.dirname(__file__),'meander_closemaps_n25.pkl')
    with open(cache,'rb') as f:mt._close=pickle.load(f)
    print(json.dumps({'n':n,'mod':p,'result':meander(n,p,False,np.uint32)}),flush=True)
