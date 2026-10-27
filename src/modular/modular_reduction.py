import os
"""Hybrid invariant: e=2 exact, e>=3 residues; no modular division.
Excess increments by 0,1,2, so residue states never feed exact V2 states.
Original exact analytic sources and V2 divisions are preserved. Reduction
follows certified nonnegative accumulation. Even/composite moduli work.
"""
def validate_modulus(n,p):
 if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=34:raise ValueError('Pilot audited domain is n=1..34')
 if isinstance(p,bool) or not isinstance(p,int) or not 2<=p<=(1<<57)-1:raise ValueError('Pilot modulus must be an integer in [2,2^57-1]')
 if (4+2*((n-1)//2))*(p-1)>=(1<<63):raise ValueError('Modulus accumulation certificate rejected')

def reduce_layer(t,target,maxima,p,sink_type,compact_type):
 import numpy as np
 result={}
 for key,A in target.items():
  e=(t-key[0]-key[1]+key[2])//2
  if isinstance(A,sink_type):result[key]=maxima[key];continue
  if e==2:result[key]=maxima[key];continue
  if e<3 or isinstance(A,compact_type):raise ArithmeticError('Unexpected modular/exact shell transition')
  if A.dtype.kind!='u':raise ArithmeticError('Hybrid CRT requires certified native unsigned storage')
  if maxima[key]<p:result[key]=maxima[key];continue
  if p>np.iinfo(A.dtype).max:raise ArithmeticError('Modulus does not fit the certified destination dtype')
  mx=0
  with A.store.borrowed(A):
   for rect in A.rectangles():
    values=A.read(*rect)
    if p&(p-1)==0 and os.environ.get('V2_CRT_USE_NUMPY')!='1':np.bitwise_and(values,A.dtype.type(p-1),out=values)
    else:np.remainder(values,A.dtype.type(p),out=values)
    mx=max(mx,int(values.max(initial=0)));A.write(*rect,values)
  result[key]=mx
 return result
