"""Exact two-shell analytic elimination. Based solely on supplied four-move transfer.
Excess 0 and 1 are computed from proved coefficient identities, never advanced
as dense states. Higher-excess states use the original exact transfer.
"""
import os,sys,time,math,pickle,json
from functools import lru_cache
from pathlib import Path
import numpy as np
import numba as nb
import allocation_guard as guard
from runtime_metrics import peak_rss_kib
sys.path.insert(0,str(Path(__file__).resolve().parent/'base'))
import meander_turbo as mt
import meander_seed as ms

@nb.njit(cache=True)
def feature_arrays(m,H,N,Btab):
    # Each prefix: total squared node-degree moment and sparse root-child moments.
    bits=np.empty(m,np.uint8)
    roots=np.zeros((N,H+1),np.int32)
    sums=np.zeros(N,np.int64)
    deg=np.empty(m+1,np.int32);rd=np.empty(m+1,np.int32)
    for r in range(N):
        hh=H;rr=r
        for mm in range(m,0,-1):
            u=Btab[mm-1,hh-1] if hh>=1 else 0
            if rr<u:bits[mm-1]=1;hh-=1
            else:bits[mm-1]=0;rr-=u;hh+=1
        # rd[d] accumulates root child counts of the active forest at level d.
        depth=0;deg[0]=0;rd[0]=0;s=0
        for pos in range(m):
            if bits[pos]:depth+=1;deg[depth]=0;rd[depth]=0
            else:
                childdeg=deg[depth];s+=(childdeg+1)*(childdeg+1)
                depth-=1;deg[depth]+=1;rd[depth]+=childdeg
        total_roots=0
        for k in range(H+1):
            roots[r,k]=rd[k];total_roots+=deg[k]
        sums[r]=s-total_roots
    nz=np.zeros(N+1,np.int64)
    for r in range(N):
        for k in range(H+1):
            if roots[r,k]:nz[r+1]+=1
    for r in range(N):nz[r+1]+=nz[r]
    lev=np.empty(nz[N],np.int16);val=np.empty(nz[N],np.int16)
    z=0
    for r in range(N):
        for k in range(H+1):
            if roots[r,k]:lev[z]=k;val[z]=roots[r,k];z+=1
    return sums,nz,lev,val

@lru_cache(None)
def features(a,h):
    guard.check(mt.B(a,h)*(8+4*(h+1)+8+4*(h+1)),('original-features',a,h))
    tab=np.zeros((a+2,a+3),np.int64)
    for m in range(a+1):
        for v in range(a+2):tab[m,v]=mt.B(m,v)
    return feature_arrays(a,h,mt.B(a,h),tab)

@lru_cache(None)
def choose(n):
    C=np.zeros((2*n+5,2*n+5),np.uint64)
    # Only degrees <= n+2 are used; avoid coefficients overflowing uint64.
    for m in range(n+3):
        for k in range(m+1):C[m,k]=math.comb(m,k)
    return C

@nb.njit(cache=True,parallel=True)
def profile_kernels(P,Q,C):
    K0=np.empty((len(P),len(Q)),np.uint64)
    K1=np.empty((len(P),len(Q)),np.uint64)
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            Z=np.uint64(0);L=np.uint64(1);A=B=np.uint64(0)
            for k in range(P.shape[1]):
                u=int(P[i,k]);v=int(Q[j,k]);m=u+v;c=C[m,u]
                # Formal forests with all roots leaves: R=u and S=u.
                loc=c*m+C[m,u+1]*u+C[m,v+1]*v+2*((v+1)*C[m+1,v+2]+(u+1)*C[m+1,u+2])
                Z=c*Z+A*C[m+1,u]+B*C[m+1,v]+L*loc
                A=L*C[m+1,v];B=L*C[m+1,u];L*=c
            K0[i,j]=L;K1[i,j]=Z
    return K0,K1

@nb.njit(cache=True,parallel=True)
def expand_e1(K0,K1,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,mod,dtype_array):
    A=np.empty((len(ip),len(iq)),dtype=dtype_array.dtype)
    for i in nb.prange(len(ip)):
        ci=ip[i]
        for j in range(len(iq)):
            cj=iq[j];base=K0[ci,cj]
            x=K1[ci,cj]+base*np.uint64(SP[i]+SQ[j])
            for z in range(op[i],op[i+1]):
                k=lp[z]
                x+=(base*np.uint64(Q[cj,k])//np.uint64(P[ci,k]+1))*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z]
                x+=(base*np.uint64(P[ci,k])//np.uint64(Q[cj,k]+1))*np.uint64(vq[z])
            if mod:x%=np.uint64(mod)
            A[i,j]=x
    return A

@lru_cache(None)
def shell_one_bound(n):
    # For K closed pairs and h unmatched openings: 3K+2h <= 2(n-1).
    return max(4*(K+h+1)**2*(1<<K)
               for K in range(2*(n-1)//3+1)
               for h in range((2*(n-1)-3*K)//2+1))

def analytic_e1(a,b,h,n,mod,dtype):
    ip,P=ms.profiles(a,h);iq,Q=ms.profiles(b,h)
    guard.array((2,len(P),len(Q)),np.uint64,('original-profile-kernels',a,b,h))
    K0,K1=profile_kernels(P,Q,choose(n))
    fp=features(a,h);fq=features(b,h)
    guard.array((len(ip),len(iq)),dtype,('original-e1-expanded',a,b,h))
    # A proved uniform bound allows avoiding a costly remainder at every entry.
    raw_ok=(mod is None or shell_one_bound(n)<mod)
    A=expand_e1(K0,K1,ip,iq,P,Q,*fp,*fq,0 if raw_ok else mod,np.empty(0,dtype))
    return A,2*K0.size

def transfer(n,current,t,mod,dtype,strip=2):
    allowed=mt.canon_keys(n,t+1);target={}
    def getT(key):
        a,b,h=key
        if a>b or key not in allowed or t+1-(a+b-h)<2*strip:return None
        T=target.get(key)
        if T is None:
            guard.array((mt.B(a,h),mt.B(b,h)),dtype,('original-dense',n,t+1,key))
            T=np.zeros((mt.B(a,h),mt.B(b,h)),dtype=dtype);target[key]=T
        return T
    def add(key,rs,cs,V):
        T=getT(key)
        if T is not None:T[rs,cs]+=V
    def sc(key,m,V):
        T=getT(key)
        if T is not None:(mt.scatter_cols_parallel if V.size>=2000000 else mt.scatter_cols_serial)(T,m,V)
    def sr(key,m,V):
        T=getT(key)
        if T is not None:(mt.scatter_rows_parallel if V.size>=2000000 else mt.scatter_rows_serial)(T,m,V)
    def dense(a,b,h,A,defect=False):
        Na,Nb=A.shape
        if not defect:add((a+1,b+1,h+1),slice(0,Na),slice(0,Nb),A)
        Up=mt.B(a-1,h-1);Uq=mt.B(b-1,h-1);Dp=Na-Up;Dq=Nb-Uq
        if b:
            if Uq and not defect:
                off=mt.B(a,h-2);add((a+1,b-1,h-1),slice(off,off+Na),slice(0,Uq),A[:,:Uq])
            if Dq:add((a+1,b-1,h+1),slice(0,Na),slice(0,Dq),A[:,Uq:])
        if a:
            if Up and not defect:
                off=mt.B(b,h-2);add((a-1,b+1,h-1),slice(0,Up),slice(off,off+Nb),A[:Up,:])
            if Dp:add((a-1,b+1,h+1),slice(0,Dp),slice(0,Nb),A[Up:,:])
        if a and b:
            if Dp and Dq:add((a-1,b-1,h+1),slice(0,Dp),slice(0,Dq),A[Up:,Uq:])
            if Up and Dq and h>=1:sc((a-1,b-1,h-1),mt.close_map(b-1,h+1),A[:Up,Uq:])
            if Dp and Uq and h>=1:sr((a-1,b-1,h-1),mt.close_map(a-1,h+1),A[Up:,:Uq])
    # Release dense sources before analytic shell expansion.
    for key in list(current):
        A=current[key]
        if A is None:continue
        current.pop(key);a,b,h=key;dense(a,b,h,A)
        if a<b:dense(b,a,h,A.T)
        del A
    pk=0;analytic_entries=0
    for a,b,h in sorted(current):
        if a==b==0:continue
        e=(t-a-b+h)//2
        if e==0:
            # Only the DD(--), delta-excess=2 move can leave both eliminated shells.
            key=(a-1,b-1,h+1);T=getT(key)
            if T is None:continue
            ip,P=ms.profiles(a,h);iq,Q=ms.profiles(b,h)
            guard.array((len(P),len(Q)),dtype,('original-zero-kernel',a,b,h))
            K=np.empty((len(P),len(Q)),dtype=dtype)
            ms.fill_kernel(K,P,Q,ms.choose_table(n),0 if mod is None else mod)
            Up=mt.B(a-1,h-1);Uq=mt.B(b-1,h-1)
            ms.gather_add(T,0,0,K,ip,iq,Up,len(ip),Uq,len(iq))
            # a>b reflected sources never land in a<=b for DD; diagonal already full.
            pk=max(pk,K.size);del K
        elif e==1:
            A,k=analytic_e1(a,b,h,n,mod,dtype);pk=max(pk,k)
            analytic_entries+=A.size
            dense(a,b,h,A,True)
            if a<b:dense(b,a,h,A.T,True)
            del A
        else:raise AssertionError(('unexpected absent dense block',a,b,h,t))
    maxpre=0
    if mod is not None:
        for A in target.values():
            maxpre=max(maxpre,int(A.max(initial=0)));np.remainder(A,dtype(mod),out=A)
    for a,b,h in allowed:
        if t+1-(a+b-h)<2*strip:target[(a,b,h)]=None
    return target,maxpre,pk,analytic_entries

def meander(n,mod=129999997,dtype=np.uint32,verbose=False):
    if not 1<=n<=25:raise ValueError('Packaged enumerator supports n=1..25; larger orders need an extended map cache and memory planning.')
    if mod is not None and (not isinstance(mod,(int,np.integer)) or mod<2):raise ValueError('Modulus must be an integer >= 2.')
    if np.dtype(dtype) not in (np.dtype(np.uint16),np.dtype(np.uint32),np.dtype(np.uint64)):raise ValueError('Expected an unsigned 16/32/64-bit dtype.')
    if mod is None and dtype!=np.uint64:raise ValueError('The implicit ring modulus 2^64 requires uint64.')
    D=4+2*((n-1)//2)
    if mod is not None and D*(mod-1)>np.iinfo(dtype).max:raise ValueError('Unsafe layerwise modulus.')
    cur={(0,0,0):None};peak=pt=mp=pk=ae=0;t0=time.perf_counter()
    for t in range(2*n-1):
        cur,m,k,aaa=transfer(n,cur,t,mod,dtype);mp=max(mp,m);pk=max(pk,k);ae+=aaa
        count=sum(A.size for A in cur.values() if A is not None)
        if count>peak:peak,pt=count,t+1
        if verbose:print(json.dumps({'t':t+1,'dense':count,'analytic_e1_expanded':aaa}),flush=True)
    if n==1:ans=1
    elif n==2:ans=2
    else:ans=int(cur[(1,1,1)][0,0])
    return {'n':n,'modulus':(1<<64) if mod is None else mod,'residue':ans,'peak_dense':peak,'peak_t':pt,'seconds':time.perf_counter()-t0,'max_pre_reduce':mp,'max_kernel_entries':pk,'analytic_e1_expanded_entries':ae,'rss_kib':peak_rss_kib()}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('n',type=int);ap.add_argument('--mod',default='129999997');ap.add_argument('--verbose',action='store_true');ap.add_argument('--cache',default=str(Path(__file__).resolve().parent/'base'/'meander_closemaps_n25.pkl'));args=ap.parse_args()
    with open(args.cache,'rb') as f:mt._close=pickle.load(f)
    mod=None if args.mod=='u64' else int(args.mod);dtype=np.uint64 if mod is None else (np.uint16 if mod<5000 else np.uint32)
    print(json.dumps(meander(args.n,mod,dtype,args.verbose)),flush=True)
