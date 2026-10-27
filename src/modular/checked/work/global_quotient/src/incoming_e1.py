"""Candidate exact contraction of the eliminated e=1 shell.
Uses ordered terminal root-degree moments; no guessed equality of states.
See PROOF_INCOMING_E1.md and independent forward-source tests.
"""
from functools import lru_cache
import time, math
import numpy as np
import numba as nb
import engine_v2 as E
import allocation_guard as guard
import recursive_profiles as rp

@nb.njit(cache=True)
def degree_step(U,D,dc,di):
    ans=np.empty(len(U)+len(D),np.uint32)
    for i in range(len(U)):ans[i]=U[i]
    for i in range(len(D)):
        c=dc[di[i]];d=0
        while c>1 and c & np.uint64(1):d+=1;c>>=np.uint64(1)
        x=D[i]
        for _ in range(d):
            x>>=np.uint32(1) # terminal zero of a root code
            while x>1 and x & np.uint32(1):x>>=np.uint32(1)
        ans[len(U)+i]=(x<<np.uint32(d+1))|((np.uint32(1)<<np.uint32(d+1))-np.uint32(2))
    return ans

@lru_cache(None)
def degree_codes(m,h):
    if m<0 or h<0 or h>m or (m-h)%2:return np.empty(0,np.uint32)
    if (m-h)//2+1>32:raise ValueError('root-degree unary code exceeds uint32 bit budget')
    if not m:return np.array([1],np.uint32)
    guard.array((E.mt.B(m,h),),np.uint32,('degree-codes',m,h))
    dc,di=rp.compressed_codes(m-1,h+1)
    return degree_step(degree_codes(m-1,h-1),degree_codes(m-1,h+1),dc,di)

@nb.njit(cache=True,parallel=True)
def build_moments(codes,ids,P,C,zmax):
    out=np.zeros((len(codes),zmax+1),np.uint64)
    h=P.shape[1]-1
    for i in nb.prange(len(codes)):
        count=int(P[ids[i],h]);r=codes[i]
        for k in range(count-1,-1,-1):
            r>>=np.uint32(1);d=0
            while r>1 and r & np.uint32(1):d+=1;r>>=np.uint32(1)
            if d:
                for z in range(zmax+1):out[i,z]+=C[z+k,z]*np.uint64(d)
    return out

@nb.njit(cache=True,inline='always')
def step(Z,L,A,B,u,v,C):
    m=u+v;c=C[m,u]
    loc=c*np.uint64(m)+C[m,u+1]*np.uint64(u)+C[m,v+1]*np.uint64(v)+np.uint64(2)*(np.uint64(v+1)*C[m+1,v+2]+np.uint64(u+1)*C[m+1,u+2])
    return (c*Z+A*C[m+1,u]+B*C[m+1,v]+L*loc,L*c,L*C[m+1,v],L*C[m+1,u])

@nb.njit(cache=True,parallel=True)
def profile_kernels_loop(P,Q,C,e):
    """Seven coefficient planes. All integer quotients are binomial identities."""
    K=np.zeros((7,len(P),len(Q)),np.uint64);h=P.shape[1]-1
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            u=int(P[i,h]);v=int(Q[j,h]);Z=np.uint64(0);L=np.uint64(1);A=np.uint64(0);B=np.uint64(0)
            Zp=Z;Lp=L;Ap=A;Bp=B
            for k in range(h):
                if k==h-1:Zp,Lp,Ap,Bp=Z,L,A,B
                Z,L,A,B=step(Z,L,A,B,int(P[i,k]),int(Q[j,k]),C)
            if e==3:
                if h==0:continue
                p=int(P[i,h-1]);q=int(Q[j,h-1])
                V,W,_,_=step(Zp,Lp,Ap,Bp,p+1,q+1,C)
                cp=W*np.uint64(q+1)//np.uint64(p+2);cq=W*np.uint64(p+1)//np.uint64(q+2)
                K[0,i,j]=W
                K[1,i,j]=V+W*np.uint64(u*(u+3)+v*(v+3))+cp*np.uint64(u)+cq*np.uint64(v)
                K[2,i,j]=cp;K[3,i,j]=cq
                continue
            W=L*(C[u+v,u+1]+C[u+v,v+1]);value=np.uint64(0)
            for k in range(v):
                z,l,a,b=step(Z,L,A,B,u,k,C)
                z,l,a,b=step(z,l,a,b,0,1,C)
                w=v-1-k;value+=z+l*np.uint64(w*(w+3))
            for k in range(u):
                z,l,a,b=step(Z,L,B,A,v,k,C)
                z,l,a,b=step(z,l,a,b,0,1,C)
                w=u-1-k;value+=z+l*np.uint64(w*(w+3))
            cp=cq=np.uint64(0)
            if h:
                p=int(P[i,h-1]);q=int(Q[j,h-1])
                cp=W*np.uint64(q)//np.uint64(p+1);cq=W*np.uint64(p)//np.uint64(q+1)
                if u==0:
                    z,w,_,_=step(Zp,Lp,Ap,Bp,p,q+1,C)
                    qp=w*np.uint64(p)//np.uint64(q+2)
                    value+=z+w*np.uint64(v*(v+3))+qp*np.uint64(v)
                    cp+=w*np.uint64(q+1)//np.uint64(p+1);cq+=qp;W+=w
                if v==0:
                    z,w,_,_=step(Zp,Lp,Ap,Bp,p+1,q,C)
                    pp=w*np.uint64(q)//np.uint64(p+2)
                    value+=z+w*np.uint64(u*(u+3))+pp*np.uint64(u)
                    cp+=pp;cq+=w*np.uint64(p+1)//np.uint64(q+1);W+=w
                # Original excess-zero -> excess-two contribution, exactly once.
                value+=Lp*C[p+q+2,p+1]
            K[0,i,j]=W;K[1,i,j]=value;K[2,i,j]=cp;K[3,i,j]=cq
            K[4,i,j]=L*(C[u+v,u+2]+C[u+v,u])
            K[5,i,j]=L*(C[u+v,v+2]+C[u+v,u]);K[6,i,j]=L
    return K

@nb.njit(cache=True,parallel=True)
def profile_kernels_lookup(P,Q,C,H,e,example):
    """Seven coefficient planes. All integer quotients are binomial identities."""
    K=np.zeros((7,len(P),len(Q)),example.dtype);h=P.shape[1]-1
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            u=int(P[i,h]);v=int(Q[j,h]);Z=np.uint64(0);L=np.uint64(1);A=np.uint64(0);B=np.uint64(0)
            Zp=Z;Lp=L;Ap=A;Bp=B
            for k in range(h):
                if k==h-1:Zp,Lp,Ap,Bp=Z,L,A,B
                Z,L,A,B=step(Z,L,A,B,int(P[i,k]),int(Q[j,k]),C)
            if e==3:
                if h==0:continue
                p=int(P[i,h-1]);q=int(Q[j,h-1])
                V,W,_,_=step(Zp,Lp,Ap,Bp,p+1,q+1,C)
                cp=W*np.uint64(q+1)//np.uint64(p+2);cq=W*np.uint64(p+1)//np.uint64(q+2)
                K[0,i,j]=W
                K[1,i,j]=V+W*np.uint64(u*(u+3)+v*(v+3))+cp*np.uint64(u)+cq*np.uint64(v)
                K[2,i,j]=cp;K[3,i,j]=cq
                continue
            W=L*(C[u+v,u+1]+C[u+v,v+1]);value=np.uint64(0)
            value=Z*H[u,v,0]+L*H[u,v,1]+A*H[u,v,2]+B*H[u,v,3]
            cp=cq=np.uint64(0)
            if h:
                p=int(P[i,h-1]);q=int(Q[j,h-1])
                cp=W*np.uint64(q)//np.uint64(p+1);cq=W*np.uint64(p)//np.uint64(q+1)
                if u==0:
                    z,w,_,_=step(Zp,Lp,Ap,Bp,p,q+1,C)
                    qp=w*np.uint64(p)//np.uint64(q+2)
                    value+=z+w*np.uint64(v*(v+3))+qp*np.uint64(v)
                    cp+=w*np.uint64(q+1)//np.uint64(p+1);cq+=qp;W+=w
                if v==0:
                    z,w,_,_=step(Zp,Lp,Ap,Bp,p+1,q,C)
                    pp=w*np.uint64(q)//np.uint64(p+2)
                    value+=z+w*np.uint64(u*(u+3))+pp*np.uint64(u)
                    cp+=pp;cq+=w*np.uint64(p+1)//np.uint64(q+1);W+=w
                # Original excess-zero -> excess-two contribution, exactly once.
                value+=Lp*C[p+q+2,p+1]
            K[0,i,j]=W;K[1,i,j]=value;K[2,i,j]=cp;K[3,i,j]=cq
            K[4,i,j]=L*(C[u+v,u+2]+C[u+v,u])
            K[5,i,j]=L*(C[u+v,v+2]+C[u+v,u]);K[6,i,j]=L
    return K

@nb.njit(cache=True,parallel=True)
def profile_kernels_context(P,Q,C,H,e,example,ctx,prev,ip,iq,ipp,iqq):
    """Seven coefficient planes. All integer quotients are binomial identities."""
    K=np.zeros((7,len(P),len(Q)),example.dtype);h=P.shape[1]-1
    for i in nb.prange(len(P)):
        for j in range(len(Q)):
            u=int(P[i,h]);v=int(Q[j,h])
            Z=np.uint64(ctx[0,ip[i],iq[j]]);L=np.uint64(ctx[1,ip[i],iq[j]])
            A=np.uint64(ctx[2,ip[i],iq[j]]);B=np.uint64(ctx[3,ip[i],iq[j]])
            Zp=np.uint64(prev[0,ipp[i],iqq[j]]);Lp=np.uint64(prev[1,ipp[i],iqq[j]])
            Ap=np.uint64(prev[2,ipp[i],iqq[j]]);Bp=np.uint64(prev[3,ipp[i],iqq[j]])
            if e==3:
                if h==0:continue
                p=int(P[i,h-1]);q=int(Q[j,h-1])
                V,W,_,_=step(Zp,Lp,Ap,Bp,p+1,q+1,C)
                cp=W*np.uint64(q+1)//np.uint64(p+2);cq=W*np.uint64(p+1)//np.uint64(q+2)
                K[0,i,j]=W
                K[1,i,j]=V+W*np.uint64(u*(u+3)+v*(v+3))+cp*np.uint64(u)+cq*np.uint64(v)
                K[2,i,j]=cp;K[3,i,j]=cq
                continue
            W=L*(C[u+v,u+1]+C[u+v,v+1]);value=np.uint64(0)
            value=Z*H[u,v,0]+L*H[u,v,1]+A*H[u,v,2]+B*H[u,v,3]
            cp=cq=np.uint64(0)
            if h:
                p=int(P[i,h-1]);q=int(Q[j,h-1])
                cp=W*np.uint64(q)//np.uint64(p+1);cq=W*np.uint64(p)//np.uint64(q+1)
                if u==0:
                    z,w,_,_=step(Zp,Lp,Ap,Bp,p,q+1,C)
                    qp=w*np.uint64(p)//np.uint64(q+2)
                    value+=z+w*np.uint64(v*(v+3))+qp*np.uint64(v)
                    cp+=w*np.uint64(q+1)//np.uint64(p+1);cq+=qp;W+=w
                if v==0:
                    z,w,_,_=step(Zp,Lp,Ap,Bp,p+1,q,C)
                    pp=w*np.uint64(q)//np.uint64(p+2)
                    value+=z+w*np.uint64(u*(u+3))+pp*np.uint64(u)
                    cp+=pp;cq+=w*np.uint64(p+1)//np.uint64(q+1);W+=w
                # Original excess-zero -> excess-two contribution, exactly once.
                value+=Lp*C[p+q+2,p+1]
            K[0,i,j]=W;K[1,i,j]=value;K[2,i,j]=cp;K[3,i,j]=cq
            K[4,i,j]=L*(C[u+v,u+2]+C[u+v,u])
            K[5,i,j]=L*(C[u+v,v+2]+C[u+v,u]);K[6,i,j]=L
    return K

@nb.njit(cache=True,parallel=True)
def add_incoming(T,K,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,MP,MQ,e):
    h=P.shape[1]-1
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];u=int(P[ci,h]);mx=np.uint64(0)
        for j in range(T.shape[1]):
            cj=iq[j];v=int(Q[cj,h]);base=np.uint64(K[0,ci,cj])
            x=np.uint64(T[i,j])+np.uint64(K[1,ci,cj])+base*(np.uint64(SP[i])+np.uint64(SQ[j]))
            for z in range(op[i],op[i+1]):
                k=lp[z]
                if k==h:c=np.uint64(K[4,ci,cj])
                elif k==h-1:c=np.uint64(K[2,ci,cj])
                else:c=base*np.uint64(Q[cj,k])//np.uint64(int(P[ci,k])+1)
                x+=c*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z]
                if k==h:c=np.uint64(K[5,ci,cj])
                elif k==h-1:c=np.uint64(K[3,ci,cj])
                else:c=base*np.uint64(P[ci,k])//np.uint64(int(Q[cj,k])+1)
                x+=c*np.uint64(vq[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*(MP[i,v]+MQ[j,u])
            T[i,j]=x
            if x>mx:mx=x
        maxima[i]=mx
    m=np.uint64(0)
    for i in range(len(maxima)):
        if maxima[i]>m:m=maxima[i]
    return m

@nb.njit(cache=True,parallel=True)
def add_incoming_symmetric(T,K,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,MP,MQ,e):
    h=P.shape[1]-1
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];u=int(P[ci,h]);mx=np.uint64(0)
        for j in range(i+1):
            cj=iq[j];v=int(Q[cj,h]);base=np.uint64(K[0,ci,cj])
            x=np.uint64(T[i,j])+np.uint64(K[1,ci,cj])+base*np.uint64(SP[i]+SQ[j])
            for z in range(op[i],op[i+1]):
                k=lp[z]
                if k==h:c=np.uint64(K[4,ci,cj])
                elif k==h-1:c=np.uint64(K[2,ci,cj])
                else:c=base*np.uint64(Q[cj,k])//np.uint64(int(P[ci,k])+1)
                x+=c*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z]
                if k==h:c=np.uint64(K[5,ci,cj])
                elif k==h-1:c=np.uint64(K[3,ci,cj])
                else:c=base*np.uint64(P[ci,k])//np.uint64(int(Q[cj,k])+1)
                x+=c*np.uint64(vq[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*(MP[i,v]+MQ[j,u])
            T[i,j]=x;T[j,i]=x
            if x>mx:mx=x
        maxima[i]=mx
    m=np.uint64(0)
    for i in range(len(maxima)):
        if maxima[i]>m:m=maxima[i]
    return m

@nb.njit(cache=True,parallel=True)
def add_incoming_recip(T,K,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,MP,MQ,R,e):
    h=P.shape[1]-1
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];u=int(P[ci,h]);mx=np.uint64(0)
        for j in range(T.shape[1]):
            cj=iq[j];v=int(Q[cj,h]);base=np.uint64(K[0,ci,cj])
            x=np.uint64(T[i,j])+np.uint64(K[1,ci,cj])+base*np.uint64(SP[i]+SQ[j])
            for z in range(op[i],op[i+1]):
                k=lp[z]
                if k==h:c=np.uint64(K[4,ci,cj])
                elif k==h-1:c=np.uint64(K[2,ci,cj])
                else:c=(base*np.uint64(Q[cj,k])*R[int(P[ci,k])+1])>>np.uint64(32)
                x+=c*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z]
                if k==h:c=np.uint64(K[5,ci,cj])
                elif k==h-1:c=np.uint64(K[3,ci,cj])
                else:c=(base*np.uint64(P[ci,k])*R[int(Q[cj,k])+1])>>np.uint64(32)
                x+=c*np.uint64(vq[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*(MP[i,v]+MQ[j,u])
            T[i,j]=x
            if x>mx:mx=x
        maxima[i]=mx
    m=np.uint64(0)
    for i in range(len(maxima)):
        if maxima[i]>m:m=maxima[i]
    return m

@nb.njit(cache=True,parallel=True)
def add_incoming_symmetric_recip(T,K,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,MP,MQ,R,e):
    h=P.shape[1]-1
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];u=int(P[ci,h]);mx=np.uint64(0)
        for j in range(i+1):
            cj=iq[j];v=int(Q[cj,h]);base=np.uint64(K[0,ci,cj])
            x=np.uint64(T[i,j])+np.uint64(K[1,ci,cj])+base*np.uint64(SP[i]+SQ[j])
            for z in range(op[i],op[i+1]):
                k=lp[z]
                if k==h:c=np.uint64(K[4,ci,cj])
                elif k==h-1:c=np.uint64(K[2,ci,cj])
                else:c=(base*np.uint64(Q[cj,k])*R[int(P[ci,k])+1])>>np.uint64(32)
                x+=c*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z]
                if k==h:c=np.uint64(K[5,ci,cj])
                elif k==h-1:c=np.uint64(K[3,ci,cj])
                else:c=(base*np.uint64(P[ci,k])*R[int(Q[cj,k])+1])>>np.uint64(32)
                x+=c*np.uint64(vq[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*(MP[i,v]+MQ[j,u])
            T[i,j]=x;T[j,i]=x
            if x>mx:mx=x
        maxima[i]=mx
    m=np.uint64(0)
    for i in range(len(maxima)):
        if maxima[i]>m:m=maxima[i]
    return m

STATS={}
MAXIMA={}
USE_MAXIMA=True
PACK_KERNELS=True
LOOKUP=True
SYMMETRIC=False
DIRECT32=False
CONTEXT=False
RECIP=False

def clear():
    degree_codes.cache_clear();MAXIMA.clear();STATS.clear()
    hockey.cache_clear()
    import sys
    if 'profile_contexts' in sys.modules:sys.modules['profile_contexts'].family.cache_clear()
    STATS.update(kernel_seconds=0.,feature_seconds=0.,add_seconds=0.,entries=0,kernel_entries=0,blocks=0)

def inject(n,t,U,target_for):
    if n>28:raise ValueError('Incoming e=1 contraction certified for n<=28 in this implementation')
    for key in sorted(U):
        a,b,h=key;e=(t-a-b+h)//2
        if e not in (2,3) or (e==3 and h==0):continue
        ts=time.perf_counter()
        ip,P=E.ms.profiles(a,h);iq,Q=E.ms.profiles(b,h)
        if DIRECT32:
            bound=source_certificate(key,e)
            if bound>=1<<32:raise ArithmeticError(('No uint32 source certificate',key,e,bound))
        kd=np.uint32 if DIRECT32 else np.uint64
        if CONTEXT:
            from profile_contexts import build
            cx=build((a-h)//2,(b-h)//2,h,rp.compressed_codes(a,h)[0],rp.compressed_codes(b,h)[0],E.old.choose(n))
            K=profile_kernels_context(P,Q,E.old.choose(n),hockey(n),e,np.empty(0,kd),*cx)
            del cx
        else:
            K=(profile_kernels_lookup(P,Q,E.old.choose(n),hockey(n),e,np.empty(0,kd)) if LOOKUP else profile_kernels_loop(P,Q,E.old.choose(n),e))
        if PACK_KERNELS and not DIRECT32:
            m=int(K.max(initial=0))
            if m<1<<16:K=K.astype(np.uint16)
            elif m<1<<32:K=K.astype(np.uint32)
        STATS['kernel_seconds']+=time.perf_counter()-ts;ts=time.perf_counter()
        fp,fq=rp.features(a,h),rp.features(b,h)
        if e==2:
            MP=build_moments(degree_codes(a,h),ip,P,E.old.choose(n),(b-h)//2)
            MQ=build_moments(degree_codes(b,h),iq,Q,E.old.choose(n),(a-h)//2)
        else:MP=MQ=np.empty((0,0),np.uint64)
        STATS['feature_seconds']+=time.perf_counter()-ts;ts=time.perf_counter()
        T=target_for(key)
        if T.dtype==object:raise ValueError('Object destination not expected in eliminated-shell source under n<=28')
        if RECIP:
            from seed_fast import reciprocals
            k=(a+b-2*h)//2
            if 6*k*(1<<k)>=1<<32:raise ArithmeticError('Reciprocal numerator certificate unavailable')
            fn=add_incoming_symmetric_recip if SYMMETRIC and a==b else add_incoming_recip
            mx=fn(T,K,ip,iq,P,Q,*fp,*fq,MP,MQ,reciprocals(n),e)
        else:
            fn=add_incoming_symmetric if SYMMETRIC and a==b else add_incoming
            mx=fn(T,K,ip,iq,P,Q,*fp,*fq,MP,MQ,e)
        if USE_MAXIMA:MAXIMA[id(T)]=(T,int(mx))
        STATS['add_seconds']+=time.perf_counter()-ts
        STATS['entries']+=T.size;STATS['kernel_entries']+=K.size;STATS['blocks']+=1

BASE_MAXIMUM=None

def maximum(A):
    q=MAXIMA.pop(id(A),None)
    if q is not None:
        assert q[0] is A
        return q[1]
    return BASE_MAXIMUM(A) if BASE_MAXIMUM is not None else A.max(initial=0)

def configure():
    import configure3 as cfg
    global BASE_MAXIMUM
    strip=cfg.configure('low-memory')
    E.SEED_INJECTOR=lambda *args:0
    E.LOCAL_FORCING=inject
    BASE_MAXIMUM=E.MAXIMUM;E.MAXIMUM=maximum
    clear()
    return strip

@lru_cache(None)
def hockey(n):
    """Exact contraction coefficients; prepared once per n using Python integers."""
    def C(m,k):return math.comb(m,k) if 0<=k<=m else 0
    def one(u,v):
        z=l=a=b=0
        for k in range(v):
            m=u+k;c=C(m,u);ca=C(m+1,u);cb=C(m+1,k)
            loc=c*m+C(m,u+1)*u+C(m,k+1)*k+2*((k+1)*C(m+1,k+2)+(u+1)*C(m+1,u+2))
            w=v-1-k
            z+=c;l+=loc+cb+2*ca+c*(3+w*(w+3));a+=ca;b+=cb
        return z,l,a,b
    T=np.zeros((n+4,n+4,4),np.uint64)
    for u in range(n+3):
        for v in range(n+3-u):
            z,l,a,b=one(u,v);zz,ll,aa,bb=one(v,u)
            values=(z+zz,l+ll,a+bb,b+aa)
            if max(values)>=1<<64:raise ArithmeticError('Contraction table exceeds uint64')
            T[u,v]=values
    return T

def source_certificate(key,e):
    a,b,h=key;k=(a+b-2*h)//2
    if e==2:
        out=k*4*(k+h+2)**2*(1<<k)
        if h:out+=8*(k+h+1)**2*(1<<(k+1))+(1<<(k+2))
    elif e==3:out=4*(k+h+2)**2*(1<<(k+2)) if h else 0
    else:raise ValueError(e)
    # Also bounds each derivative coefficient plane, separately from its value.
    return max(out,6*(k+2)*(1<<(k+2)))
