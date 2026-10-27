"""Reuse immutable compressed source windows across their RAM consumers."""
import extended_engine as BASE
import extended_v2 as CV
import best_engine as R
import bounded_storage as BS
import ram_transfer as X

def meander(*args,**kwargs):
    saved=R.add,R.scatter,CV.Compact.close
    old_add,old_scatter,old_close=saved
    pending={}
    X.clear()
    def enqueue(T,V,kind,off=None,maps=None,backend=None):
        if isinstance(V,BS.View) and isinstance(V.base,CV.Compact) and X.resident(T):
            pending.setdefault(id(V.base),[]).append((T,V,kind,off,maps,backend));return True
        return False
    def add(T,V,off=(0,0)):
        if not enqueue(T,V,'add',off):return old_add(T,V,off)
    def scatter(T,V,kind,maps,backend):
        if not enqueue(T,V,kind,maps=maps,backend=backend):return old_scatter(T,V,kind,maps,backend)
    def close(A):
        reqs=pending.pop(id(A),[])
        if reqs and not X.apply(A,reqs,old_add):
            X.STATS['fallback_sources']+=1
            for T,V,kind,off,maps,backend in reqs:
                with A.store.borrowed(T,A):
                    if kind=='add':old_add(T,V,off)
                    else:old_scatter(T,V,kind,maps,backend)
        old_close(A)
    R.add=add;R.scatter=scatter;CV.Compact.close=close
    try:
        result=BASE.meander(*args,**kwargs)
        assert not pending
        result['ram_fanout']=X.STATS.copy()
        return result
    finally:R.add,R.scatter,CV.Compact.close=saved;pending.clear()
