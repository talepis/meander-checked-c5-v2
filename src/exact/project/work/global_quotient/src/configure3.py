"""Production configurations; no conjectural pruning or numeric assumptions."""
import configure_v2 as base
import variants3 as variants
import local_forcing as f
engine=base.engine

def configure(mode='balanced'):
    if mode not in ('fast','balanced','low-memory'):raise ValueError('Unknown engine mode')
    strip=base.configure('low-memory' if mode=='low-memory' else 'speed')
    engine.MAXIMUM=None;engine.DOWNSIZE=False;engine.LOWER_ONLY=False
    engine.LOWER_ADD=None;engine.MIRROR=None;engine.LOCAL_FORCING=None
    engine.PRESTAGE=None;engine.ON_RELEASE=None;engine.BOUND_REFINER=None;engine.LOCAL_BOUNDS=None
    f.TIGHT=False;f.CLOSED_TWO=False
    engine.ms.gather_add=variants.oldga;engine.ms.gather_cols=variants.oldgc;engine.ms.gather_rows=variants.oldgr
    if mode=='fast':variant='tight+directbounds'
    elif mode=='balanced':variant='release+bound+directbounds'
    else:variant='forcing+directbounds'
    variants.configure(variant,base)
    return strip

def clear_work_caches():base.clear_work_caches();f.clear()
load_maps=base.load_maps
