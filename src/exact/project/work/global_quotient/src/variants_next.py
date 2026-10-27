"""New experiments layered over the unchanged separated-source worker."""
import source_separation as S
import source_signatures as G
import recursive_signatures as R
import configure_bugfirst as B

def configure(name):
    if name=='baseline':return S.configure('separated-mirror')
    if name=='packed':return S.configure('separated-packed-mirror')
    if name=='recursive':
        strip=S.configure('separated-grouped-mirror')
        G.groups=R.groups
        return strip
    if name in ('recip','tile','recip-tile'):
        strip=S.configure('separated-mirror')
        import fast_factors as F
        F.configure('recip' in name,'tile' in name)
        return strip
    if name.startswith('cache'):
        strip=S.configure('separated-mirror')
        import context_reuse as C
        C.configure(int(name[5:]))
        return strip
    if name=='family':
        strip=S.configure('separated-mirror')
        import context_reuse as C
        C.C.family=C.family
        return strip
    if name.startswith('projected'):
        strip=S.configure('separated-mirror')
        if 'family' in name:
            import context_reuse as C
            C.C.family=C.family
        if 'sym' in name:
            import symmetric_context as SC
            SC.configure()
        import projected_source as P
        P.configure()
        return strip
    if name in ('sym','sym-family'):
        strip=S.configure('separated-mirror')
        if 'family' in name:
            import context_reuse as C
            C.C.family=C.family
        import symmetric_context as SC
        SC.configure()
        return strip
    raise ValueError(name)

def clear():
    B.clear_work_caches();S.clear();R.clear()
    import sys
    if 'context_reuse' in sys.modules:sys.modules['context_reuse'].clear()
    if 'projected_source' in sys.modules:sys.modules['projected_source'].clear()
