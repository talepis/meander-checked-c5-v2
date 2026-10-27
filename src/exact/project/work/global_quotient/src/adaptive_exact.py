"""One exact run, with certified per-block widths. No CRT or guess enters arithmetic.
Safe block bounds sum (maximum predecessor coefficient)*(maximum reverse multiplicity).
Each target is typed BEFORE any addition, and every predecessor maximum is exact.
"""
import os,sys,time,math,json,pickle,gc
from pathlib import Path
from functools import lru_cache
import numpy as np
from runtime_metrics import peak_rss_kib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'baseline'))
import seed_two as old
mt=old.mt;ms=old.ms
SEED_INJECTOR=None
COMPACT_ZERO=False

@lru_cache(None)
def multiplicity(m,h):
    v=mt.close_map(m,h)
    return int(np.bincount(v).max(initial=0))

def seed_bound(key,e):
    a,b,h=key;K=(a+b-2*h)//2
    if e==0:return math.comb(K,(a-h)//2)
    return 4*(K+h+1)**2*(1<<K)

def dtype_for(u,policy):
    if policy=='u64':return np.uint64 if u<(1<<64) else object
    if policy=='u32':return np.uint32 if u<(1<<32) else (np.uint64 if u<(1<<64) else object)
    if u<(1<<16):return np.uint16
    if u<(1<<32):return np.uint32
    if u<(1<<64):return np.uint64
    return object

# op=(destination, kind, flip, source rectangle, destination offset, optional map key)
def operations(n,t,keys,strip=2):
    allowed={k for k in mt.canon_keys(n,t+1) if t+1-sum(k[:2])+k[2]>=2*strip}
    plan={}
    for key in keys:
        a0,b0,h=key
        e=(t-a0-b0+h)//2
        out=[]
        for flip in range(1+(a0<b0)):
            a,b=(b0,a0) if flip else (a0,b0)
            Na,Nb=mt.B(a,h),mt.B(b,h)
            Up,Uq=mt.B(a-1,h-1),mt.B(b-1,h-1)
            Dp,Dq=Na-Up,Nb-Uq
            def push(dest,kind,rect,off=(0,0),mp=None):
                if dest in allowed and rect[0]<rect[1] and rect[2]<rect[3]:
                    out.append((dest,kind,flip,rect,off,mp))
            push((a+1,b+1,h+1),'add',(0,Na,0,Nb))
            if True:
                if b:
                    push((a+1,b-1,h-1),'add',(0,Na,0,Uq),(mt.B(a,h-2),0))
                    push((a+1,b-1,h+1),'add',(0,Na,Uq,Nb))
                if a:
                    push((a-1,b+1,h-1),'add',(0,Up,0,Nb),(0,mt.B(b,h-2)))
                    push((a-1,b+1,h+1),'add',(Up,Na,0,Nb))
                if a and b and h>=1:
                    push((a-1,b-1,h-1),'cols',(0,Up,Uq,Nb),mp=(b-1,h+1))
                    push((a-1,b-1,h-1),'rows',(Up,Na,0,Uq),mp=(a-1,h+1))
            if a and b:push((a-1,b-1,h+1),'add',(Up,Na,Uq,Nb))
        if out:plan[key]=out
    return plan,allowed

def _add_scatter(T,V,kind,maps,threshold=2000000):
    if V.dtype==object and T.dtype!=object:V=V.astype(T.dtype)
    if T.dtype==object or V.dtype==object:
        if kind=='cols':
            for j,k in enumerate(maps):T[:,k]+=V[:,j]
        else:
            for i,k in enumerate(maps):T[k,:]+=V[i,:]
        return
    if kind=='cols':fn=mt.scatter_cols_parallel if V.size>=threshold else mt.scatter_cols_serial
    else:fn=mt.scatter_rows_parallel if V.size>=threshold else mt.scatter_rows_serial
    fn(T,maps,V)

def meander(n,policy='mixed',verbose=False,trace=False,assert_bounds=True,strip=2):
    if not 1<=n<=28:raise ValueError('This research release limits n to 28; memory grows exponentially')
    if policy not in ('mixed','u32','u64'):raise ValueError('policy must be mixed,u32,u64')
    if strip not in (0,1,2):raise ValueError('strip must be 0,1,2')
    cur={(0,0,0):np.array([[1]],dtype=np.uint16)} if strip==0 else {}
    maxima={(0,0,0):1} if strip==0 else {}
    history=[];seconds={x:0.0 for x in ('bounds','dense','analytic','maxima')}
    peak_entries=peak_bytes=peak_live=peak_t=peak_object=0;analytic_entries=0
    t0=time.perf_counter()
    for t in range(2*n-1):
        ts=time.perf_counter()
        keys=mt.canon_keys(n,t)
        plan,allowed=operations(n,t,keys,strip)
        U={}
        for key,ops in plan.items():
            e=(t-sum(key[:2])+key[2])//2
            mx=maxima[key] if e>=strip else seed_bound(key,e)
            for dest,kind,flip,rect,off,mp in ops:
                mult=1 if kind=='add' else multiplicity(*mp)
                U[dest]=U.get(dest,0)+mx*mult
        for key in U:
            e=(t+1-sum(key[:2])+key[2])//2
            if e==0:U[key]=min(U[key],seed_bound(key,0))
        target={};target_dtypes={key:dtype_for(u,policy) for key,u in U.items()}
        seconds['bounds']+=time.perf_counter()-ts
        def target_for(key):
            T=target.get(key)
            if T is None:
                a,b,h=key;T=np.zeros((mt.B(a,h),mt.B(b,h)),dtype=target_dtypes[key]);target[key]=T
            return T
        # Dense sources are consumed first to reduce the two-layer peak.
        for key in list(cur):
            ts=time.perf_counter();A=cur.pop(key)
            for dest,kind,flip,rect,off,mp in plan.get(key,[]):
                r0,r1,c0,c1=rect
                V=(A.T if flip else A)[r0:r1,c0:c1]
                T=target_for(dest)
                if kind=='add':
                    tr,tc=off
                    if V.dtype==object and T.dtype!=object:V=V.astype(T.dtype)
                    T[tr:tr+r1-r0,tc:tc+c1-c0]+=V
                else:_add_scatter(T,V,kind,mt.close_map(*mp))
            del A
            peak_live=max(peak_live,sum(A.nbytes for A in cur.values())+sum(A.nbytes for A in target.values()))
            seconds['dense']+=time.perf_counter()-ts
        for key,ops in sorted(plan.items()):
            a,b,h=key;e=(t-a-b+h)//2
            if e>=strip:continue
            ts=time.perf_counter()
            if e==0:
                ip,P=ms.profiles(a,h);iq,Q=ms.profiles(b,h)
                kdt=dtype_for(seed_bound(key,0),'mixed') if COMPACT_ZERO else np.uint64
                K=np.empty((len(P),len(Q)),dtype=kdt)
                ms.fill_kernel(K,P,Q,ms.choose_table(n),0)
                for dest,kind,flip,rect,off,mp in ops:
                    T=target_for(dest);r0,r1,c0,c1=rect
                    KK,ir,ic=(K.T,iq,ip) if flip else (K,ip,iq)
                    if T.dtype==object:
                        V=KK[ir[r0:r1,None],ic[None,c0:c1]].astype(object)
                        if kind=='add':
                            tr,tc=off;T[tr:tr+r1-r0,tc:tc+c1-c0]+=V
                        else:_add_scatter(T,V,kind,mt.close_map(*mp))
                    elif kind=='add':ms.gather_add(T,*off,KK,ir,ic,r0,r1,c0,c1)
                    elif kind=='cols':ms.gather_cols(T,mt.close_map(*mp),KK,ir,ic,r0,r1,c0,c1)
                    else:ms.gather_rows(T,mt.close_map(*mp),KK,ir,ic,r0,r1,c0,c1)
                del K
            elif SEED_INJECTOR is not None:
                analytic_entries+=SEED_INJECTOR(key,ops,target_for,n)
            else:
                dt=np.uint32 if seed_bound(key,1)<(1<<32) else np.uint64
                A,_=old.analytic_e1(a,b,h,n,None,dt);analytic_entries+=A.size
                for dest,kind,flip,rect,off,mp in ops:
                    r0,r1,c0,c1=rect;V=(A.T if flip else A)[r0:r1,c0:c1];T=target_for(dest)
                    if kind=='add':
                        tr,tc=off;T[tr:tr+r1-r0,tc:tc+c1-c0]+=V
                    else:_add_scatter(T,V,kind,mt.close_map(*mp))
                del A
            seconds['analytic']+=time.perf_counter()-ts
        ts=time.perf_counter()
        maxima={key:int(A.max(initial=0)) for key,A in target.items()}
        if assert_bounds:
            for key,mx in maxima.items():
                if mx>U[key]:raise ArithmeticError(('invalid block bound',t,key,mx,U[key]))
        count=sum(A.size for A in target.values());nbytes=sum(A.nbytes for A in target.values())
        objs=sum(A.size for A in target.values() if A.dtype==object)
        if count>peak_entries:peak_entries,peak_t=count,t+1
        peak_bytes=max(peak_bytes,nbytes);peak_object=max(peak_object,objs)
        widths={str(dt):sum(A.size for A in target.values() if A.dtype==dt) for dt in (np.dtype('uint16'),np.dtype('uint32'),np.dtype('uint64'),np.dtype('object'))}
        step={'t':t+1,'entries':count,'array_bytes':nbytes,'widths':widths,'max_value':max(maxima.values(),default=0)}
        if trace:step['blocks']=[{'key':key,'dtype':str(A.dtype),'size':A.size,'max':maxima[key],'bound':U[key]} for key,A in target.items()]
        history.append(step)
        if verbose:print(json.dumps(step),flush=True)
        cur=target
        seconds['maxima']+=time.perf_counter()-ts
    ans=1 if n==1 else (2 if n==2 else int(cur[(1,1,1)][0,0]))
    return {'n':n,'answer':ans,'policy':policy,'strip':strip,'seconds':time.perf_counter()-t0,'phase_seconds':seconds,'peak_dense':peak_entries,'peak_t':peak_t,'peak_array_bytes':peak_bytes,'peak_live_dense_bytes':peak_live,'peak_object_entries':peak_object,'analytic_entries':analytic_entries,'rss_kib':peak_rss_kib(),'layers':history}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('n',type=int);ap.add_argument('--policy',default='mixed');ap.add_argument('--tag',default='adaptive');ap.add_argument('--verbose',action='store_true');ap.add_argument('--trace',action='store_true');ap.add_argument('--warm',type=int,default=10)
    ar=ap.parse_args()
    with open(ROOT/'baseline/base/meander_closemaps_n25.pkl','rb') as f:mt._close=pickle.load(f)
    if ar.warm:meander(ar.warm,ar.policy)
    ms.profiles.cache_clear();old.features.cache_clear();gc.collect()
    out=meander(ar.n,ar.policy,ar.verbose,ar.trace)
    (ROOT/'results'/f'{ar.tag}.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='layers'}),flush=True)
