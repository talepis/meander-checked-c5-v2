"""Evaluate only analytic source cells with retained outgoing moves; exact symmetry reuse."""
import math
import numpy as np
import numba as nb
import adaptive_exact as ae
import seed_fast as sf
old=ae.old;ms=ae.ms

@nb.njit(cache=True,parallel=True)
def compact_symmetric_kernels(P,C,k0example):
    N=len(P);K0=np.empty((N,N),k0example.dtype);K1=np.empty((N,N),np.uint32)
    for i in nb.prange(N):
        for j in range(i+1):
            Z=np.uint64(0);L=np.uint64(1);A=B=np.uint64(0)
            for k in range(P.shape[1]):
                u=int(P[i,k]);v=int(P[j,k]);m=u+v;c=C[m,u]
                loc=c*m+C[m,u+1]*u+C[m,v+1]*v+2*((v+1)*C[m+1,v+2]+(u+1)*C[m+1,u+2])
                Z=c*Z+A*C[m+1,u]+B*C[m+1,v]+L*loc
                A=L*C[m+1,v];B=L*C[m+1,u];L*=c
            K0[i,j]=L;K0[j,i]=L;K1[i,j]=Z;K1[j,i]=Z
    return K0,K1

@nb.njit(cache=True,parallel=True)
def expand_range(K0,K1,ip,iq,P,Q,SP,op,lp,vp,SQ,oq,lq,vq,R,ra,rb,ca,cb,symmetric):
    A=np.empty((rb-ra,cb-ca),np.uint32)
    for ii in nb.prange(rb-ra):
        i=ra+ii;ci=ip[i]
        last=ca+ii+1 if symmetric else cb
        for j in range(ca,last):
            cj=iq[j];base=np.uint64(K0[ci,cj])
            x=np.uint64(K1[ci,cj])+base*np.uint64(SP[i]+SQ[j])
            for z in range(op[i],op[i+1]):
                k=lp[z];numer=base*np.uint64(Q[cj,k])
                x+=((numer*R[int(P[ci,k])+1])>>np.uint64(32))*np.uint64(vp[z])
            for z in range(oq[j],oq[j+1]):
                k=lq[z];numer=base*np.uint64(P[ci,k])
                x+=((numer*R[int(Q[cj,k])+1])>>np.uint64(32))*np.uint64(vq[z])
            A[ii,j-ca]=x
            if symmetric:A[j-ca,ii]=x
    return A

class RangeInjector:
    def __init__(self,symmetry=True):self.symmetry=symmetry
    def __call__(self,key,ops,target_for,n):
        a,b,h=key
        if old.shell_one_bound(n)>=(1<<32):raise ValueError('analytic uint32 bound unavailable')
        ip,P=ms.profiles(a,h);iq,Q=ms.profiles(b,h)
        k=(a+b-2*h)//2;kdt=np.uint16 if math.comb(k,k//2)<(1<<16) else np.uint32
        sym=self.symmetry and a==b
        if sym:K0,K1=compact_symmetric_kernels(P,old.choose(n),np.empty(0,kdt))
        else:K0,K1=sf.compact_profile_kernels(P,Q,old.choose(n),np.empty(0,kdt))
        fp,fq=old.features(a,h),old.features(b,h)
        Up,Uq=ae.mt.B(a-1,h-1),ae.mt.B(b-1,h-1)
        Na,Nb=len(ip),len(iq)
        R=sf.reciprocals(n)
        def consume(quad,A):
            qr0,qr1,qc0,qc1=quad
            for dest,kind,flip,rect,off,mp in ops:
                r0,r1,c0,c1=rect
                # Intersect in the original (unreflected) source coordinates.
                sr0,sr1,sc0,sc1=(c0,c1,r0,r1) if flip else rect
                l=max(qr0,sr0);r=min(qr1,sr1);c=max(qc0,sc0);d=min(qc1,sc1)
                if l>=r or c>=d:continue
                V=A[l-qr0:r-qr0,c-qc0:d-qc0]
                if flip:V=V.T;l,r,c,d=c,d,l,r
                T=target_for(dest)
                if kind=='add':
                    tr,tc=off;tr+=l-r0;tc+=c-c0
                    T[tr:tr+r-l,tc:tc+d-c]+=V
                elif kind=='cols':
                    M=ae.mt.close_map(*mp)[c-c0:d-c0]
                    ae._add_scatter(T[l-r0:r-r0,:],V,kind,M)
                else:
                    M=ae.mt.close_map(*mp)[l-r0:r-r0]
                    ae._add_scatter(T[:,c-c0:d-c0],V,kind,M)
        entries=0
        if Up and Uq<Nb:
            q=(0,Up,Uq,Nb)
            A=expand_range(K0,K1,ip,iq,P,Q,*fp,*fq,R,*q,False);entries+=A.size
            consume(q,A)
            if sym:consume((Up,Na,0,Uq),A.T)
            del A
        if not sym and Up<Na and Uq:
            q=(Up,Na,0,Uq)
            A=expand_range(K0,K1,ip,iq,P,Q,*fp,*fq,R,*q,False);entries+=A.size;consume(q,A);del A
        if Up<Na and Uq<Nb:
            q=(Up,Na,Uq,Nb)
            A=expand_range(K0,K1,ip,iq,P,Q,*fp,*fq,R,*q,sym)
            entries+=A.size if not sym else len(A)*(len(A)+1)//2
            consume(q,A);del A
        return entries
