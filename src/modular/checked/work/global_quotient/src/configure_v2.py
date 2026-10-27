"""Configuration of the selected exact kernels, kept separate from ablation scripts."""
import sys,pickle
import numpy as np
import engine_v2 as engine
import kernels_v2 as kernels
import recursive_profiles as profiles
import allocation_guard as guard

ORIGINAL_SCATTER=engine._add_scatter
ORIGINAL_PROFILES=engine.ms.profiles
ORIGINAL_FEATURES=engine.old.features
_inverse_cache={}

def inverse(m, n):
    # Retain the map reference: temporary NumPy view IDs can be recycled.
    key=(int(m.ctypes.data),m.shape,m.strides,int(n))
    if key not in _inverse_cache:
        if m.ndim!=1 or (m.size and (int(m.min())<0 or int(m.max())>=n)):
            raise ValueError('Close-map image is outside the target axis')
        guard.check((int(n)+1)*16+len(m)*4,'inverse-map')
        ptr,ix=kernels.inverse_map(m,n)
        _inverse_cache[key]=(m,ptr,ix)
    return _inverse_cache[key][1:]

def scatter(T,V,kind,maps,threshold=100000):
    axis=0 if kind=='rows' else 1
    if V.shape[axis]!=len(maps):raise ValueError('Close-map/source shape mismatch')
    if T.dtype!=object and V.dtype!=object and V.size>=threshold:
        if kind=='rows' and V.strides[1]<=V.strides[0]:
            kernels.scatter_rows_inverse(T,*inverse(maps,T.shape[0]),V);return
        if kind=='cols' and V.strides[0]<V.strides[1]:
            kernels.scatter_rows_inverse(T.T,*inverse(maps,T.shape[1]),V.T);return
    ORIGINAL_SCATTER(T,V,kind,maps)

def configure(mode='speed'):
    if mode not in ('speed','low-memory'):raise ValueError('Unknown exact engine')
    engine.COMPACT_ZERO=True
    engine.ADD_KERNEL=kernels.add_parallel
    engine._add_scatter=scatter
    engine.ms.profiles=profiles.profiles
    engine.CONSTANT_INJECTOR=profiles.constant_inject
    engine.ZERO_INJECTOR=None
    engine.SOURCE_ORDER=None
    engine._UINT8=False
    engine.TRACE_ZERO=False
    engine.MEMORY_GUARD=False
    # Compatibility for retained seed, map-building and optional verification modules.
    sys.modules['adaptive_exact']=engine
    engine.old.features=profiles.features if mode=='low-memory' else ORIGINAL_FEATURES
    if mode=='low-memory':
        from seed_ranges import RangeInjector
        engine.SEED_INJECTOR=RangeInjector(True)
    else:engine.SEED_INJECTOR=None
    return 1 if mode=='speed' else 2

def load_maps(n):
    path=engine.ROOT/'baseline/base/meander_closemaps_n25.pkl'
    with path.open('rb') as f:engine.mt._close=pickle.load(f)
    if n>25:
        from extend_maps import extend
        extend(n)

def clear_work_caches():
    profiles.clear_all();ORIGINAL_PROFILES.cache_clear();ORIGINAL_FEATURES.cache_clear()
    _inverse_cache.clear();engine.ZERO_STATS.clear()
