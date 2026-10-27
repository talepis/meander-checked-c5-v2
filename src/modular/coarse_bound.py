"""Proved root-excess Green-tree upper bound. No extrema assumption."""
def upper(a,b,h):
 if min(a,b,h)<0 or h>min(a,b) or (a-h)%2 or (b-h)%2:raise ValueError('Invalid equal-height prefix dimensions')
 k=(a+b-2*h)//2
 return (k+1)*(k+h+1)*(8*(k+2)*(k+h+3)**2+4)*(1<<k)
