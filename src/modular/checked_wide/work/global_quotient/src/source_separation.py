"""Exact one-sided separation of the contracted sources.

F(P,Q) = R(P, profile(Q)) + C(profile(P), Q).
No quotient of completed coefficients is used. Non-negative integer terms
are partitioned, so each side is bounded by the existing source certificate.
"""
import numpy as np
import numba as nb
import time
import incoming_e1 as I
import engine_v2 as E
import allocation_guard as guard
import recursive_profiles as rp

@nb.njit(cache=True, parallel=True)
def row_source(K,ip,P,Q,S,off,lev,val,M,e):
    out=np.empty((len(ip),len(Q)),K.dtype)
    h=P.shape[1]-1
    for i in nb.prange(len(ip)):
        ci=ip[i]
        for cj in range(len(Q)):
            base=np.uint64(K[0,ci,cj])
            x=np.uint64(K[1,ci,cj])+base*np.uint64(S[i])
            for z in range(off[i],off[i+1]):
                k=lev[z]
                if k==h:c=np.uint64(K[4,ci,cj])
                elif k==h-1:c=np.uint64(K[2,ci,cj])
                else:c=base*np.uint64(Q[cj,k])//np.uint64(int(P[ci,k])+1)
                x+=c*np.uint64(val[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*M[i,int(Q[cj,h])]
            out[i,cj]=x
    return out

@nb.njit(cache=True, parallel=True)
def col_source(K,iq,P,Q,S,off,lev,val,M,e):
    out=np.empty((len(P),len(iq)),K.dtype)
    h=P.shape[1]-1
    for ci in nb.prange(len(P)):
        for j in range(len(iq)):
            cj=iq[j];base=np.uint64(K[0,ci,cj])
            x=base*np.uint64(S[j])
            for z in range(off[j],off[j+1]):
                k=lev[z]
                if k==h:c=np.uint64(K[5,ci,cj])
                elif k==h-1:c=np.uint64(K[3,ci,cj])
                else:c=base*np.uint64(P[ci,k])//np.uint64(int(Q[cj,k])+1)
                x+=c*np.uint64(val[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*M[j,int(P[ci,h])]
            out[ci,j]=x
    return out

@nb.njit(cache=True, parallel=True)
def add_separated(T,R,C,ip,iq):
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];mx=np.uint64(0)
        for j in range(T.shape[1]):
            x=np.uint64(T[i,j])+np.uint64(R[i,iq[j]])+np.uint64(C[ci,j])
            T[i,j]=x
            if x>mx:mx=x
        maxima[i]=mx
    result=np.uint64(0)
    for k in range(len(maxima)):
        if maxima[k]>result:result=maxima[k]
    return result

@nb.njit(cache=True, parallel=True)
def fuse_separated(T,R,C,ip,iq,UU,X,Y,Up,Uq):
    maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];mx=np.uint64(0)
        for j in range(T.shape[1]):
            x=np.uint64(R[i,iq[j]])+np.uint64(C[ci,j])
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

# Same as the old source evaluator, but initializes each destination from
# its dense zero-excess predecessors rather than from a pre-updated T.
@nb.njit(cache=True, parallel=True)
def fuse_direct(T,K,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,MP,MQ,e,UU,X,Y,Up,Uq):
    h=P.shape[1]-1;maxima=np.zeros(T.shape[0],np.uint64)
    for i in nb.prange(T.shape[0]):
        ci=ip[i];u=int(P[ci,h]);mx=np.uint64(0)
        for j in range(T.shape[1]):
            cj=iq[j];v=int(Q[cj,h]);base=np.uint64(K[0,ci,cj])
            x=np.uint64(K[1,ci,cj])+base*(np.uint64(SP[i])+np.uint64(SQ[j]))
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
            if e==2:x+=np.uint64(K[6,ci,cj])*(np.uint64(MP[i,v])+np.uint64(MQ[j,u]))
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

STATS={}
FUSION=False
SEPARATE=False
MIN_SIZE=4096
RATIO=1.5
GROUPED=False
DEPENDENTS={}
DONE=set()
FUSED=set()

def clear():
    STATS.clear();STATS.update(prepare_seconds=0.,separate_seconds=0.,apply_seconds=0.,separated_entries=0,full_entries=0,fused_entries=0,peak_temporary_bytes=0,direct_evaluations=0,one_sided_evaluations=0,reflected_copy_entries=0)
    DEPENDENTS.clear();DONE.clear();FUSED.clear()
    import sys
    if 'source_signatures' in sys.modules:sys.modules['source_signatures'].groups.cache_clear()

def data(n,key,e):
    ts=time.perf_counter();a,b,h=key
    ip,P=E.ms.profiles(a,h);iq,Q=E.ms.profiles(b,h)
    cap=I.source_certificate(key,e)
    if cap>=1<<64:raise ArithmeticError('Missing BEFORE-cast uint64 source certificate')
    kd=np.uint32 if cap<1<<32 else np.uint64
    guard.array((7,len(P),len(Q)),kd,('source-kernel',n,key,e))
    if I.CONTEXT:
        from profile_contexts import build
        cx=build((a-h)//2,(b-h)//2,h,rp.compressed_codes(a,h)[0],rp.compressed_codes(b,h)[0],E.old.choose(n))
        # Context construction allocates its own matrices after the earlier
        # check. Recheck the kernel against CURRENT commit before Numba allocates.
        guard.array((7,len(P),len(Q)),kd,('source-kernel-after-context',n,key,e))
        K=I.profile_kernels_context(P,Q,E.old.choose(n),I.hockey(n),e,np.empty(0,kd),*cx)
    else:K=I.profile_kernels_lookup(P,Q,E.old.choose(n),I.hockey(n),e,np.empty(0,kd))
    fp,fq=rp.features(a,h),rp.features(b,h)
    if e==2:
        guard.array((len(ip),(b-h)//2+1),np.uint64,('moments-P',key))
        MP=I.build_moments(I.degree_codes(a,h),ip,P,E.old.choose(n),(b-h)//2)
        guard.array((len(iq),(a-h)//2+1),np.uint64,('moments-Q',key))
        MQ=I.build_moments(I.degree_codes(b,h),iq,Q,E.old.choose(n),(a-h)//2)
    else:MP=MQ=np.empty((0,0),np.uint64)
    STATS['prepare_seconds']+=time.perf_counter()-ts
    return K,ip,iq,P,Q,fp,fq,MP,MQ

def apply(n,key,e,T,parents=None):
    a,b,h=key
    K,ip,iq,P,Q,fp,fq,MP,MQ=data(n,key,e)
    size=len(ip)*len(iq);small=len(ip)*len(Q)+len(P)*len(iq)
    selected=SEPARATE and size>=MIN_SIZE and small<size*RATIO
    if selected:
        guard.check((len(ip)*len(Q)+len(P)*len(iq))*K.dtype.itemsize,('separated-source',key,e))
        ts=time.perf_counter()
        if GROUPED:
            import source_signatures as G
            rp0,gp,fr=G.groups(a,h,e);rq0,gq,fc=G.groups(b,h,e)
            R=row_source(K,ip[rp0],P,Q,*fr,MP[rp0] if e==2 else MP,e)
            C=(reflected_column(R,K[1],ip[rp0]) if MIRROR_REUSE and a==b else col_source(K,iq[rq0],P,Q,*fc,MQ[rq0] if e==2 else MQ,e))
            small=R.size+C.size
        else:
            R=row_source(K,ip,P,Q,*fp,MP,e)
            C=(reflected_column(R,K[1],ip) if MIRROR_REUSE and a==b else col_source(K,iq,P,Q,*fq,MQ,e))
        STATS['one_sided_evaluations']+=R.size+(0 if MIRROR_REUSE and a==b else C.size)
        if MIRROR_REUSE and a==b:STATS['reflected_copy_entries']+=C.size
        STATS['separate_seconds']+=time.perf_counter()-ts
        STATS['separated_entries']+=small
        STATS['peak_temporary_bytes']=max(STATS['peak_temporary_bytes'],R.nbytes+C.nbytes+K.nbytes)
        ts=time.perf_counter()
        if GROUPED:
            if parents is None:mx=G.add_grouped(T,R,C,ip,iq,gp,gq)
            else:mx=G.fuse_grouped(T,R,C,ip,iq,gp,gq,*parents,E.mt.B(a-1,h-1),E.mt.B(b-1,h-1))
        elif parents is None:mx=add_separated(T,R,C,ip,iq)
        else:mx=fuse_separated(T,R,C,ip,iq,*parents,E.mt.B(a-1,h-1),E.mt.B(b-1,h-1))
    else:
        STATS['direct_evaluations']+=size
        ts=time.perf_counter()
        if parents is None:mx=I.add_incoming(T,K,ip,iq,P,Q,*fp,*fq,MP,MQ,e)
        else:mx=fuse_direct(T,K,ip,iq,P,Q,*fp,*fq,MP,MQ,e,*parents,E.mt.B(a-1,h-1),E.mt.B(b-1,h-1))
    STATS['apply_seconds']+=time.perf_counter()-ts;STATS['full_entries']+=size
    if parents is not None:STATS['fused_entries']+=size
    return int(mx)

def inject(n,t,U,target_for):
    for key in sorted(U):
        a,b,h=key;e=(t-a-b+h)//2
        if e not in (2,3) or (e==3 and not h) or key in DONE:continue
        T=target_for(key)
        if T.dtype==object:raise ArithmeticError('Unexpected bigint analytic destination')
        mx=apply(n,key,e,T)
        I.MAXIMA[id(T)]=(T,mx)

def prepare_release(n,t,cur,U,target_for,previous_maxima):
    DEPENDENTS.clear();DONE.clear();FUSED.clear()
    for key in U:
        a,b,h=key
        if t+1-a-b+h!=4 or E.mt.B(a,h)*E.mt.B(b,h)<MIN_SIZE:continue
        if E.dtype_for(U[key],'mixed')==object:continue
        ps=[]
        for aa,bb,hh in [(a-1,b-1,h-1),(a-1,b+1,h+1),(a+1,b-1,h+1)]:
            p=(min(aa,bb),max(aa,bb),hh)
            if p in cur:ps.append(p)
        if not ps:continue
        FUSED.add(key)
        for p in set(ps):DEPENDENTS.setdefault(p,[]).append(key)
    return {}

def on_release(n,t,parent,cur,U,target_for,previous_maxima,premax):
    for key in DEPENDENTS.get(parent,[]):
        if key in DONE:continue
        a,b,h=key;dt=E.dtype_for(U[key],'mixed');empty=np.empty((0,0),dt)
        def source(aa,bb,hh):
            p=(min(aa,bb),max(aa,bb),hh)
            if p not in previous_maxima:return empty
            if p not in cur:raise ArithmeticError(('Predecessor released too soon',key,p))
            V=cur[p];return V.T if aa>bb else V
        parents=(source(a-1,b-1,h-1),source(a-1,b+1,h+1),source(a+1,b-1,h+1))
        # All source and coefficient operands are nonnegative. U already includes
        # all of them, so the old arithmetic certificate also covers fused order.
        T=target_for(key,False)
        premax[key]=apply(n,key,2,T,parents);DONE.add(key)

def configure(variant):
    import configure_bugfirst as base
    global FUSION,SEPARATE,RATIO,GROUPED,MIRROR_REUSE
    strip=base.configure('contracted')
    if variant=='contracted':return strip
    MIRROR_REUSE='mirror' in variant
    variant=variant.replace('-mirror','')
    RATIO=1.5
    for tag,value in [('ratiohalf',0.5),('ratioone',1.0),('ratioall',100.0)]:
        if tag in variant:RATIO=value;variant=variant.replace('-'+tag,'')
    GROUPED=('grouped' in variant or 'packed' in variant)
    import source_signatures as G
    G.PACKED='packed' in variant
    variant=variant.replace('-grouped','').replace('-packed','')
    if variant not in ('separated','fused','fused-separated'):raise ValueError(variant)
    FUSION='fused' in variant;SEPARATE='separated' in variant
    E.LOCAL_FORCING=inject
    if FUSION:E.PRESTAGE=prepare_release;E.ON_RELEASE=on_release
    clear();return strip

@nb.njit(cache=True,parallel=True)
def reflected_column(R,K1,profile_ids):
    """Exact reflection identity, tiled for contiguous short inner accesses."""
    out=np.empty((R.shape[1],R.shape[0]),R.dtype)
    nt=(R.shape[0]+31)//32
    for tile in nb.prange(nt):
        j0=tile*32;j1=min(j0+32,R.shape[0])
        for ci0 in range(0,R.shape[1],32):
            ci1=min(ci0+32,R.shape[1])
            for j in range(j0,j1):
                cj=profile_ids[j]
                for ci in range(ci0,ci1):out[ci,j]=R[j,ci]-K1[ci,cj]
    return out

MIRROR_REUSE=False
