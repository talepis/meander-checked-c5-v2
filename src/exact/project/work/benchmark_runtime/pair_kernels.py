"""Values at requested pairs, using the proved integer mixed contraction."""
import numpy as np
import numba as nb
from analytic_mixed_kernels import divide

@nb.njit(cache=True,parallel=True)
def decode(R,C,anchors,ip,iq,DP,offq,levq,valq,P,Q,choose,rows,cols,cap,template):
    out=np.empty(len(rows),template.dtype);errors=np.zeros(len(rows),np.uint8)
    d=P.shape[1]
    for z in nb.prange(len(rows)):
        i=rows[z];j=cols[z];gp=ip[i];gq=iq[j]
        value=np.int64(R[i,gq])+np.int64(C[gp,j])-np.int64(R[anchors[gp],gq])
        if offq[j]!=offq[j+1]:
            nonzero=False
            for l in range(d+1):
                if DP[i,l]:nonzero=True
            if nonzero:
                K=np.int64(1)
                for l in range(d):
                    if P[gp,l] and Q[gq,l]:K*=np.int64(choose[int(P[gp,l])+int(Q[gq,l]),int(P[gp,l])])
                A=K*np.int64(DP[i,0])
                for l in range(d):
                    if Q[gq,l] and DP[i,l+1]:
                        kl,rem=divide(K*np.int64(Q[gq,l]),np.int64(P[gp,l])+1)
                        if rem:errors[z]|=1
                        A+=kl*np.int64(DP[i,l+1])
                for t in range(offq[j],offq[j+1]):
                    coordinate=levq[t]
                    if coordinate==0:coefficient=A
                    else:
                        l=coordinate-1;delta=np.int64(DP[i,l+1])
                        if not P[gp,l]:coefficient=K*delta
                        else:
                            kl=np.int64(0)
                            if delta and Q[gq,l]:
                                kl,rem=divide(K*np.int64(Q[gq,l]),np.int64(P[gp,l])+1)
                                if rem:errors[z]|=1
                            coefficient,rem=divide((A-kl*delta)*np.int64(P[gp,l]),np.int64(Q[gq,l])+1)
                            if rem:errors[z]|=1
                            coefficient+=K*delta
                    value+=coefficient*np.int64(valq[t])
        if value<0:errors[z]|=2;out[z]=0
        elif value>cap:errors[z]|=4;out[z]=0
        else:out[z]=value
    return out,errors

@nb.njit(cache=True)
def sum128(values):
    lo=np.uint64(0);hi=np.uint64(0)
    for value in values:
        previous=lo;lo+=np.uint64(value)
        if lo<previous:hi+=np.uint64(1)
    return lo,hi

@nb.njit(cache=True)
def sum128_folded(values,off_diagonal):
    lo=np.uint64(0);hi=np.uint64(0)
    for z in range(len(values)):
        value=np.uint64(values[z]);previous=lo;lo+=value
        if lo<previous:hi+=np.uint64(1)
        if off_diagonal[z]:
            previous=lo;lo+=value
            if lo<previous:hi+=np.uint64(1)
    return lo,hi
