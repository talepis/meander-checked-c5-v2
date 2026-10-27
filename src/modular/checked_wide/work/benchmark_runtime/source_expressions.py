"""Finite integer expressions for analytic sources, no K/R/C coefficient planes."""
import numpy as np
import numba as nb
import incoming_e1 as I

@nb.njit(inline='always')
def coefficients2(P,Q,C,H):
    h=len(P)-1;u=int(P[h]);v=int(Q[h])
    Z=np.uint64(0);L=np.uint64(1);A=np.uint64(0);B=np.uint64(0)
    Zp=Z;Lp=L;Ap=A;Bp=B
    for k in range(h):
        if k==h-1:Zp,Lp,Ap,Bp=Z,L,A,B
        Z,L,A,B=I.step(Z,L,A,B,int(P[k]),int(Q[k]),C)
    W=L*(C[u+v,u+1]+C[u+v,v+1])
    value=Z*H[u,v,0]+L*H[u,v,1]+A*H[u,v,2]+B*H[u,v,3]
    cp=np.uint64(0);cq=np.uint64(0)
    if h:
        p=int(P[h-1]);q=int(Q[h-1])
        cp=W*np.uint64(q)//np.uint64(p+1);cq=W*np.uint64(p)//np.uint64(q+1)
        if u==0:
            z,w,_,_=I.step(Zp,Lp,Ap,Bp,p,q+1,C)
            qp=w*np.uint64(p)//np.uint64(q+2)
            value+=z+w*np.uint64(v*(v+3))+qp*np.uint64(v)
            cp+=w*np.uint64(q+1)//np.uint64(p+1);cq+=qp;W+=w
        if v==0:
            z,w,_,_=I.step(Zp,Lp,Ap,Bp,p+1,q,C)
            pp=w*np.uint64(q)//np.uint64(p+2)
            value+=z+w*np.uint64(u*(u+3))+pp*np.uint64(u)
            cp+=pp;cq+=w*np.uint64(p+1)//np.uint64(q+1);W+=w
        value+=Lp*C[p+q+2,p+1]
    chp=L*(C[u+v,u+2]+C[u+v,u]);chq=L*(C[u+v,v+2]+C[u+v,u])
    return W,value,cp,cq,chp,chq,L

@nb.njit(inline='always')
def moment(code,count,z,C):
    out=np.uint64(0)
    for k in range(count-1,-1,-1):
        code>>=np.uint32(1);degree=0
        while code>1 and code&np.uint32(1):degree+=1;code>>=np.uint32(1)
        if degree:out+=C[z+k,z]*np.uint64(degree)
    return out

@nb.njit(cache=True,parallel=True)
def values2(rows,cols,ip,iq,P,Q,sp,op,lp,vp,sq,oq,lq,vq,dp,dq,C,H,cap):
    out=np.empty(len(rows),np.uint64);errors=np.zeros(len(rows),np.uint8)
    h=P.shape[1]-1
    for z in nb.prange(len(rows)):
        i=rows[z];j=cols[z];pi=ip[i];qj=iq[j]
        base,value,cp,cq,chp,chq,L=coefficients2(P[pi],Q[qj],C,H)
        x=value+base*(np.uint64(sp[i])+np.uint64(sq[j]))
        for t in range(op[i],op[i+1]):
            k=lp[t]
            coefficient=chp if k==h else (cp if k==h-1 else base*np.uint64(Q[qj,k])//np.uint64(int(P[pi,k])+1))
            x+=coefficient*np.uint64(vp[t])
        for t in range(oq[j],oq[j+1]):
            k=lq[t]
            coefficient=chq if k==h else (cq if k==h-1 else base*np.uint64(P[pi,k])//np.uint64(int(Q[qj,k])+1))
            x+=coefficient*np.uint64(vq[t])
        u=int(P[pi,h]);v=int(Q[qj,h])
        x+=L*(moment(dp[i],u,v,C)+moment(dq[j],v,u,C))
        if x>cap:errors[z]=1
        out[z]=x
    return out,errors

@nb.njit(cache=True,parallel=True)
def values3(rows,cols,ip,iq,P,Q,sp,op,lp,vp,sq,oq,lq,vq,C,cap):
    out=np.empty(len(rows),np.uint64);errors=np.zeros(len(rows),np.uint8)
    for z in nb.prange(len(rows)):
        i=rows[z];j=cols[z];pi=ip[i];qj=iq[j]
        Z=np.uint64(0);L=np.uint64(1);A=np.uint64(0);B=np.uint64(0)
        for k in range(P.shape[1]):Z,L,A,B=I.step(Z,L,A,B,int(P[pi,k]),int(Q[qj,k]),C)
        x=Z+L*(np.uint64(sp[i])+np.uint64(sq[j]))
        for t in range(op[i],op[i+1]):
            k=lp[t];x+=(L*np.uint64(Q[qj,k])//np.uint64(int(P[pi,k])+1))*np.uint64(vp[t])
        for t in range(oq[j],oq[j+1]):
            k=lq[t];x+=(L*np.uint64(P[pi,k])//np.uint64(int(Q[qj,k])+1))*np.uint64(vq[t])
        if x>cap:errors[z]=1
        out[z]=x
    return out,errors
