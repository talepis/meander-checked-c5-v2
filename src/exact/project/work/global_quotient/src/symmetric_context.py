"""Triangular evaluation of exact coefficient planes, NOT of frontier states."""
import numpy as np
import numba as nb
import incoming_e1 as I
OLD=I.profile_kernels_context

@nb.njit(cache=True,inline='always')
def entry(P,Q,C,H,e,ctx,prev,ip,iq,ipp,iqq,i,j):
    h=P.shape[1]-1;u=int(P[i,h]);v=int(Q[j,h])
    Z=np.uint64(ctx[0,ip[i],iq[j]]);L=np.uint64(ctx[1,ip[i],iq[j]])
    A=np.uint64(ctx[2,ip[i],iq[j]]);B=np.uint64(ctx[3,ip[i],iq[j]])
    Zp=np.uint64(prev[0,ipp[i],iqq[j]]);Lp=np.uint64(prev[1,ipp[i],iqq[j]])
    Ap=np.uint64(prev[2,ipp[i],iqq[j]]);Bp=np.uint64(prev[3,ipp[i],iqq[j]])
    if e==3:
        if h==0:
            z=np.uint64(0);return z,z,z,z,z,z,z
        p=int(P[i,h-1]);q=int(Q[j,h-1])
        V,W,_,_=I.step(Zp,Lp,Ap,Bp,p+1,q+1,C)
        cp=W*np.uint64(q+1)//np.uint64(p+2);cq=W*np.uint64(p+1)//np.uint64(q+2)
        val=V+W*np.uint64(u*(u+3)+v*(v+3))+cp*np.uint64(u)+cq*np.uint64(v)
        return W,val,cp,cq,np.uint64(0),np.uint64(0),np.uint64(0)
    W=L*(C[u+v,u+1]+C[u+v,v+1])
    value=Z*H[u,v,0]+L*H[u,v,1]+A*H[u,v,2]+B*H[u,v,3]
    cp=cq=np.uint64(0)
    if h:
        p=int(P[i,h-1]);q=int(Q[j,h-1])
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
    return W,value,cp,cq,L*(C[u+v,u+2]+C[u+v,u]),L*(C[u+v,v+2]+C[u+v,u]),L

@nb.njit(cache=True,parallel=True)
def triangular(P,Q,C,H,e,example,ctx,prev,ip,iq,ipp,iqq):
    N=len(P);out=np.empty((7,N,N),example.dtype);nt=(N+31)//32
    for z in nb.prange(nt*(nt+1)//2):
        lo=0;hi=nt
        while lo+1<hi:
            mid=(lo+hi)//2
            if mid*(mid+1)//2<=z:lo=mid
            else:hi=mid
        ti=lo;tj=z-ti*(ti+1)//2
        for i in range(ti*32,min((ti+1)*32,N)):
            for j in range(tj*32,min((tj+1)*32,N)):
                if j>i:continue
                w,v,cp,cq,rp,rq,l=entry(P,Q,C,H,e,ctx,prev,ip,iq,ipp,iqq,i,j)
                out[0,i,j]=w;out[0,j,i]=w
                out[1,i,j]=v;out[1,j,i]=v
                out[2,i,j]=cp;out[3,j,i]=cp
                out[3,i,j]=cq;out[2,j,i]=cq
                out[4,i,j]=rp;out[5,j,i]=rp
                out[5,i,j]=rq;out[4,j,i]=rq
                out[6,i,j]=l;out[6,j,i]=l
    return out

def kernel(P,Q,*args):
    return triangular(P,Q,*args) if P is Q and len(P)>=32 else OLD(P,Q,*args)

def configure():I.profile_kernels_context=kernel
