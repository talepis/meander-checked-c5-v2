import best_engine as R
import allocation_guard as G
import fast_memory as F
import tile_transfer as X
import bounded_storage as BS
import packed_boundary as PB

def meander(*args,**kwargs):
    X.clear()
    saved=(G.check,PB.regular_add,PB.regular_scatter,BS.Block.rectangles,PB.PackedBoundary.rectangles)
    G.check=F.check;PB.regular_add=X.add;PB.regular_scatter=X.scatter
    BS.Block.rectangles=X.rectangles;PB.PackedBoundary.rectangles=X.packed_rectangles
    try:r=R.meander(*args,**kwargs)
    finally:
        G.check,PB.regular_add,PB.regular_scatter,BS.Block.rectangles,PB.PackedBoundary.rectangles=saved
    r['tile_transfer']=X.STATS.copy();r['additional_shared_context_cache_bytes']=0
    return r
