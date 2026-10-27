"""Accepted projected-family-sym exact arithmetic with explicit bounded storage.
The operations, target certificates and analytic identities are inherited.
Only the coefficient storage and rectangle scheduling differ.
"""
import time
import numpy as np
import engine_v2 as E
import incoming_e1 as I
import source_separation as S
import projected_source as P3
import allocation_guard as guard
from bounded_storage import Store,add,scatter,features_slice
import compact_v2 as CV
import extremal_v2
from packed_boundary import PackedBoundary,should_pack,packed_add as add,packed_scatter as scatter
import best_engine as RETAINED
import terminal_sink as TS

def inject(n,key,e,T):
    if isinstance(T,TS.Sink):return TS.inject(n,key,e,T)
    if T.dtype==object:raise ArithmeticError('unexpected bigint analytic source')
    if e==3:
        K,ip,iq,P,Q,fp,fq=P3.data(n,key)
        MP=MQ=None
    else:K,ip,iq,P,Q,fp,fq,MP,MQ=S.data(n,key,e)
    size=len(ip)*len(iq);small=len(ip)*len(Q)+len(P)*len(iq)
    selected=size>=S.MIN_SIZE and small<size*S.RATIO
    if selected:
        guard.check(small*K.dtype.itemsize,('bounded source factors',key,e))
        if e==3:R=P3.row(K,ip,P,Q,*fp)
        else:R=S.row_source(K,ip,P,Q,*fp,MP,e)
        if key[0]==key[1]:C=S.reflected_column(R,K[1],ip)
        elif e==3:C=P3.col(K,iq,P,Q,*fq)
        else:C=S.col_source(K,iq,P,Q,*fq,MQ,e)
    mx=0
    for r0,r1,c0,c1 in T.rectangles():
        with T.store.borrowed(T):
            A=T.read(r0,r1,c0,c1)
            if selected:m=S.add_separated(A,R[r0:r1],C[:,c0:c1],ip[r0:r1],iq[c0:c1])
            elif e==3:m=P3.direct(A,K,ip[r0:r1],iq[c0:c1],P,Q,*features_slice(fp,r0,r1),*features_slice(fq,c0,c1))
            else:m=I.add_incoming(A,K,ip[r0:r1],iq[c0:c1],P,Q,*features_slice(fp,r0,r1),*features_slice(fq,c0,c1),MP[r0:r1],MQ[c0:c1],e)
            mx=max(mx,int(m));T.write(r0,r1,c0,c1,A);del A
    return mx

def meander(n,root,ram_limit=4<<30,small_limit=64<<20,chunk=64<<20,verbose=True,layer_check=None,force_files=False):
    global CV
    CV=RETAINED.CV
    accumulator=TS.Accumulator()
    if not 1<=n<=33:raise ValueError('audited domain 1..33')
    if E.LOWER_ONLY or E.PRESTAGE or E.ON_RELEASE or E.DOWNSIZE:raise RuntimeError('unsupported configured hooks')
    store=Store(root,ram_limit,small_limit,chunk)
    import variants_next
    store.cache_evictor=variants_next.clear
    store.add_kernel=E.ADD_KERNEL
    guard.PRESSURE_HANDLER=store.make_space
    class ScatterBackend:
        # Reuse the accepted CPU gather kernel when RAM windows are contiguous.
        # Each target row belongs to one thread; sums retain their typed bounds.
        @staticmethod
        def scatter_rows_parallel(T,m,V):E._add_scatter(T,V,'rows',m)
        @staticmethod
        def scatter_cols_parallel(T,m,V):E._add_scatter(T,V,'cols',m)
    scatter_backend=ScatterBackend()
    cur={};maxima={};history=[];started=time.perf_counter()
    phases=dict(bounds=0.,dense=0.,analytic=0.,maxima=0.)
    for t in range(2*n-1):
        ts=time.perf_counter();keys=E.mt.canon_keys(n,t);plan,allowed=E.operations(n,t,keys,2);U={}
        # Every boundary history is contracted at its FIRST entry. Its DD
        # continuation is already included by the exact terminal functional.
        plan={key:ops for key,ops in plan.items() if not TS.boundary(n,t,key)}
        for key,ops in plan.items():
            e=(t-sum(key[:2])+key[2])//2
            if e==0 and E.LOCAL_BOUNDS is not None:continue
            mx=maxima[key] if e>=2 else E.seed_bound(key,e)
            for dest,kind,flip,rect,off,mp in ops:
                mult=1 if kind=='add' else E.multiplicity(*mp)
                U[dest]=U.get(dest,0)+mx*mult
        if E.LOCAL_BOUNDS is not None:E.LOCAL_BOUNDS(n,t+1,allowed,U)
        if E.BOUND_REFINER is not None:E.BOUND_REFINER(n,t+1,U,maxima)
        target={};source_max={}
        compact_keys={k for k in U if t+1-k[0]-k[1]+k[2]==4 and not TS.boundary(n,t+1,k) and CV.selected(*k)}
        # A second, explicitly experimental hypothesis supplies tight V2 maxima.
        # Sample accumulation and every decoded coefficient are checked against it.
        for ckey in compact_keys:U[ckey]=min(U[ckey],extremal_v2.maximum(*ckey))
        dependents={}
        for ckey in compact_keys:
            a,b,h=ckey
            for aa,bb,hh in ((a-1,b-1,h-1),(a-1,b+1,h+1),(a+1,b-1,h+1)):
                ck=(min(aa,bb),max(aa,bb),hh)
                if ck in cur:dependents.setdefault(ck,set()).add(ckey)
        def build_compact(ckey):
            if ckey not in target:
                target[ckey]=CV.Compact(n,ckey,U[ckey],cur,maxima,store)
                source_max[ckey]=target[ckey].bound
        def target_for(key):
            if key not in target:
                a,b,h=key;shape=(E.mt.B(a,h),E.mt.B(b,h));dt=E.dtype_for(U[key],'mixed')
                if TS.boundary(n,t+1,key):target[key]=TS.Sink(n,t+1,key,U[key],store,accumulator)
                elif should_pack(n,t+1,key):target[key]=PackedBoundary(store,shape,dt,E.mt.B(a-1,h-1))
                else:target[key]=store.create(shape,dt)
            T=target[key]
            if force_files and not isinstance(T,TS.Sink):
                if isinstance(T,PackedBoundary):T.spill()
                elif T.array is not None and T.dtype.kind=='u':T.spill()
            return T
        phases['bounds']+=time.perf_counter()-ts;ts=time.perf_counter()
        for key in sorted(cur):
            for ck in sorted(dependents.get(key,())):build_compact(ck)
            A=cur[key]
            for dest,kind,flip,rect,off,mp in plan.get(key,[]):
                if dest in compact_keys:continue
                T=target_for(dest);V=A.view(rect,flip)
                with store.borrowed(T,A):
                    if isinstance(T,TS.Sink):
                        # Reflection gives equal final projections of the two
                        # orientations of an asymmetric canonical predecessor.
                        if key[0]<key[1] and flip:continue
                        scale=2 if key[0]<key[1] else 1
                        if kind=='add':T.add(V,off,scale=scale)
                        else:T.scatter(V,kind,E.mt.close_map(*mp),scatter_backend,scale=scale)
                    elif kind=='add':RETAINED.add(T,V,off)
                    else:RETAINED.scatter(T,V,kind,E.mt.close_map(*mp),scatter_backend)
            A.close()
        cur.clear();phases['dense']+=time.perf_counter()-ts;ts=time.perf_counter()
        for key in sorted(U):
            if key in compact_keys:
                build_compact(key);continue
            a,b,h=key;e=(t+1-a-b+h)//2
            if e in (2,3) and (e!=3 or h):source_max[key]=inject(n,key,e,target_for(key))
        phases['analytic']+=time.perf_counter()-ts;ts=time.perf_counter()
        maxima={key:source_max[key] if key in source_max else A.maximum_reduce() for key,A in target.items()}
        for key,mx in maxima.items():
            if mx>U[key]:raise ArithmeticError(('invalid target bound',t+1,key,mx,U[key]))
        if layer_check:layer_check(t+1,target,maxima,U)
        step=dict(t=t+1,seconds=time.perf_counter()-started,entries=sum(A.size for A in target.values()),array_bytes=sum(A.charge if isinstance(A,CV.Compact) else A.nbytes for A in target.values()),logical_array_bytes=sum(A.logical_nbytes if isinstance(A,PackedBoundary) else A.nbytes for A in target.values()),active_entries=sum(A.active_size if isinstance(A,PackedBoundary) else A.size for A in target.values()),pruned_native_bytes=sum(A.logical_nbytes-A.nbytes for A in target.values() if isinstance(A,PackedBoundary)),packed_boundary_blocks=sum(isinstance(A,PackedBoundary) for A in target.values()),ram_blocks_bytes=store.ram,disk_bytes=store.disk,max_value_decimal=str(max(maxima.values(),default=0)),blocks=[dict(key=key,dtype=str(A.dtype),size=A.size,max_decimal=str(maxima[key]),bound_decimal=str(U[key]),storage='packed' if isinstance(A,PackedBoundary) else ('compact' if isinstance(A,CV.Compact) else ('ram' if A.array is not None else 'file'))) for key,A in sorted(target.items())])
        history.append(step)
        step['storage_stats_cumulative']=store.stats.copy()
        step['terminal_stats_cumulative']=accumulator.stats.copy()
        step['terminal_sinks']=sum(isinstance(A,TS.Sink) for A in target.values())
        step['terminal_geometry_bytes']=accumulator.stats['geometry_live_bytes']
        step['terminal_contribution_decimal']=str(accumulator.total)
        for block in step['blocks']:
            if isinstance(target[block['key']],TS.Sink):
                block['storage']='terminal-functional';block['max_is_unused_upper_bound']=True
                block['stored_coefficient_bytes']=0
        if verbose:
            import json
            print(json.dumps({k:v for k,v in step.items() if k!='blocks'}),flush=True)
        phases['maxima']+=time.perf_counter()-ts;cur=target
    answer=1 if n==1 else (2 if n==2 else accumulator.total)
    for A in cur.values():A.close()
    store.finish()
    assert accumulator.stats['geometry_live_bytes']==0
    guard.PRESSURE_HANDLER=None
    return dict(n=n,answer_decimal=str(answer),seconds=time.perf_counter()-started,phase_seconds=phases,layers=history,storage_stats=store.stats,variant='exact-terminal-boundary-functional',strip=2,conjectural_V2_used=True,conjectural_extremal_bounds_used=True,compact_statistics=CV.STATS.copy(),terminal_functional=accumulator.stats.copy(),terminal_maxima_are_unused_upper_bounds=True)
