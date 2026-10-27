"""Alternative EXACT kernels: reciprocal division and cache-friendly columns."""
import numpy as np
import numba as nb
import source_separation as S
import seed_fast

@nb.njit(cache=True,parallel=True)
def row_recip(K,ip,P,Q,SS,off,lev,val,M,e,recip):
    out=np.empty((len(ip),len(Q)),K.dtype);h=P.shape[1]-1
    for i in nb.prange(len(ip)):
        ci=ip[i]
        for cj in range(len(Q)):
            base=np.uint64(K[0,ci,cj]);x=np.uint64(K[1,ci,cj])+base*np.uint64(SS[i])
            for z in range(off[i],off[i+1]):
                k=lev[z]
                if k==h:c=np.uint64(K[4,ci,cj])
                elif k==h-1:c=np.uint64(K[2,ci,cj])
                else:c=(base*np.uint64(Q[cj,k])*recip[int(P[ci,k])+1])>>np.uint64(32)
                x+=c*np.uint64(val[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*M[i,int(Q[cj,h])]
            out[i,cj]=x
    return out

@nb.njit(cache=True,parallel=True)
def col_recip(K,iq,P,Q,SS,off,lev,val,M,e,recip):
    out=np.empty((len(P),len(iq)),K.dtype);h=P.shape[1]-1
    for ci in nb.prange(len(P)):
        for j in range(len(iq)):
            cj=iq[j];base=np.uint64(K[0,ci,cj]);x=base*np.uint64(SS[j])
            for z in range(off[j],off[j+1]):
                k=lev[z]
                if k==h:c=np.uint64(K[5,ci,cj])
                elif k==h-1:c=np.uint64(K[3,ci,cj])
                else:c=(base*np.uint64(P[ci,k])*recip[int(Q[cj,k])+1])>>np.uint64(32)
                x+=c*np.uint64(val[z])
            if e==2:x+=np.uint64(K[6,ci,cj])*M[j,int(P[ci,h])]
            out[ci,j]=x
    return out

@nb.njit(cache=True,parallel=True)
def col_tiled(K,iq,P,Q,SS,off,lev,val,M,e,recip,use_recip):
    out=np.empty((len(P),len(iq)),K.dtype);h=P.shape[1]-1
    nt=(len(iq)+31)//32
    for tile in nb.prange(nt):
        j0=tile*32;j1=min(j0+32,len(iq))
        for ci0 in range(0,len(P),32):
            ci1=min(ci0+32,len(P))
            for j in range(j0,j1):
                cj=iq[j]
                for ci in range(ci0,ci1):
                    base=np.uint64(K[0,ci,cj]);x=base*np.uint64(SS[j])
                    for z in range(off[j],off[j+1]):
                        k=lev[z]
                        if k==h:c=np.uint64(K[5,ci,cj])
                        elif k==h-1:c=np.uint64(K[3,ci,cj])
                        elif use_recip:c=(base*np.uint64(P[ci,k])*recip[int(Q[cj,k])+1])>>np.uint64(32)
                        else:c=base*np.uint64(P[ci,k])//np.uint64(int(Q[cj,k])+1)
                        x+=c*np.uint64(val[z])
                    if e==2:x+=np.uint64(K[6,ci,cj])*M[j,int(P[ci,h])]
                    out[ci,j]=x
    return out

ORIGINAL_ROW=S.row_source;ORIGINAL_COL=S.col_source

def certificate(P,Q):
    # k is the total number of closed pairs, not just visible roots. Thus its
    # runtime full-geometry check is installed at S.data below.
    return seed_fast.reciprocals(30)

def row(*args):return row_recip(*args,certificate(args[2],args[3]))
def col(*args):return col_recip(*args,certificate(args[2],args[3]))
def tiled(*args):return col_tiled(*args,certificate(args[2],args[3]),False)
def tiled_recip(*args):return col_tiled(*args,certificate(args[2],args[3]),True)
ORIGINAL_DATA=S.data

def checked_data(n,key,e):
    a,b,h=key;k=(a+b-2*h)//2
    if 6*k*(1<<k)>=1<<32:raise ArithmeticError('Reciprocal numerator certificate unavailable')
    return ORIGINAL_DATA(n,key,e)

def configure(recip=False,tile=False):
    if recip:S.data=checked_data;S.row_source=row
    if tile:S.col_source=tiled_recip if recip else tiled
    elif recip:S.col_source=col
