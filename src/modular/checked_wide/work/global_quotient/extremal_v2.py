"""Experimental V2 maximum conjecture. No stored Mn or measured maxima are inputs.

Concentrating all leaves at a common optimal forest level matched 949 complete
V2 block maxima in the retained M31 run. This is empirical evidence, not proof.
"""
import math
from functools import lru_cache
from probe_mixed import newsource,feat

@lru_cache(None)
def leaf_value(p,q,k,h):
 W='('*k+'()'*p+'('*(h-k);Z='('*k+'()'*q+'('*(h-k)
 result=newsource(W,Z,2)
 if h:
  u=feat(W+')')[0];v=feat(Z+')')[0]
  result+=math.prod(math.comb(x+y,x) for x,y in zip(u,v))
 if not p and not q:return result+(leaf_value(0,0,min(k,h-1),h-1) if h else 0)
 if h>k:return result+leaf_value(p,q,k,h-1)
 if p:result+=leaf_value(p-1,q,k,h+1)
 if q:result+=leaf_value(p,q-1,k,h+1)
 return result

@lru_cache(None)
def maximum(a,b,h):
 p,q=(a-h)//2,(b-h)//2
 return max(leaf_value(p,q,k,h) for k in range(h+1))
