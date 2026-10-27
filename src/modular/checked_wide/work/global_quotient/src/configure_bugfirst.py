"""Select the audited contracted-source engine, retaining the former engines."""
import configure3 as previous
import incoming_e1 as incoming
engine=previous.engine

def configure(mode='contracted'):
    if mode=='contracted':
        strip=incoming.configure()
        incoming.LOOKUP=True;incoming.DIRECT32=True;incoming.CONTEXT=True
        incoming.SYMMETRIC=False;incoming.RECIP=False
        incoming.USE_MAXIMA=True
        return strip
    if mode not in ('low-memory','balanced','fast'):raise ValueError('Unknown engine')
    return previous.configure(mode)

def clear_work_caches():
    previous.clear_work_caches();incoming.clear()

load_maps=previous.load_maps
