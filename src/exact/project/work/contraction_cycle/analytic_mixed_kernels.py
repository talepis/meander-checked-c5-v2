"""Exact diagonal-plus-rank-one mixed V2 contraction; no sampled H plane."""
import numpy as np
import numba as nb


@nb.njit(inline='always')
def divide(value,den):
    if den==1:return value,0
    if den==2:return value>>1,value&1
    if den==4:return value>>2,value&3
    if den==8:return value>>3,value&7
    if den==16:return value>>4,value&15
    return value//den,value%den


@nb.njit(cache=True)
def deltas(X,ids,anchors):
    D=np.empty(X.shape,np.int32)
    off=np.zeros(len(X)+1,np.int64)
    for i in range(len(X)):
        a=anchors[ids[i]]
        for j in range(X.shape[1]):
            d=X[i,j]-X[a,j]
            D[i,j]=d
            if d:off[i+1]+=1
    for i in range(len(X)):off[i+1]+=off[i]
    lev=np.empty(off[-1],np.int16);val=np.empty(off[-1],np.int32)
    for i in range(len(X)):
        z=off[i]
        for j in range(X.shape[1]):
            if D[i,j]:lev[z]=j;val[z]=D[i,j];z+=1
    return D,off,lev,val


@nb.njit(cache=True,parallel=True)
def decode(R,C,anchorsp,anchorsq,ip,iq,DP,offq,levq,valq,P,Q,choose,rows,cols,order,ends,cap,template):
    out=np.empty((len(rows),len(cols)),template.dtype)
    errors=np.zeros(len(rows),np.int32)
    d=P.shape[1]
    for ii in nb.prange(len(rows)):
        i=rows[ii];gp=ip[i];ap=anchorsp[gp]
        zero=True
        for l in range(d+1):
            if DP[i,l]:zero=False
        if zero:
            for jj in range(len(cols)):
                j=cols[jj];gq=iq[j]
                value=np.int64(R[i,gq])+np.int64(C[gp,j])-np.int64(R[ap,gq])
                if value<0:errors[ii]|=2;out[ii,jj]=0
                elif value>cap:errors[ii]|=4;out[ii,jj]=0
                else:out[ii,jj]=value
            continue
        temp=np.empty(d+1,np.int64);kl=np.empty(d,np.int64)
        for group in range(len(ends)-1):
            begin=ends[group];end=ends[group+1];gq=iq[cols[order[begin]]]
            active=False
            for z in range(begin,end):
                j=cols[order[z]]
                if offq[j]!=offq[j+1]:active=True
            if active:
                K=np.int64(1)
                for l in range(d):
                    if P[gp,l] and Q[gq,l]:K*=np.int64(choose[int(P[gp,l])+int(Q[gq,l]),int(P[gp,l])])
                A=K*np.int64(DP[i,0])
                for l in range(d):
                    kl[l]=0
                    if Q[gq,l] and DP[i,l+1]:
                        numerator=K*np.int64(Q[gq,l]);den=np.int64(P[gp,l])+1
                        quotient,remainder=divide(numerator,den)
                        if remainder:errors[ii]|=1
                        kl[l]=quotient;A+=quotient*np.int64(DP[i,l+1])
                temp[0]=A
                for l in range(d):
                    if not P[gp,l]:temp[l+1]=K*np.int64(DP[i,l+1])
                    else:
                        numerator=(A-kl[l]*np.int64(DP[i,l+1]))*np.int64(P[gp,l])
                        den=np.int64(Q[gq,l])+1
                        quotient,remainder=divide(numerator,den)
                        if remainder:errors[ii]|=1
                        temp[l+1]=quotient+K*np.int64(DP[i,l+1])
            base=np.int64(R[i,gq])-np.int64(R[ap,gq])
            for z in range(begin,end):
                jj=order[z];j=cols[jj];value=base+np.int64(C[gp,j])
                for t in range(offq[j],offq[j+1]):value+=temp[levq[t]]*np.int64(valq[t])
                if value<0:errors[ii]|=2;out[ii,jj]=0
                elif value>cap:errors[ii]|=4;out[ii,jj]=0
                else:out[ii,jj]=value
    return out,errors
