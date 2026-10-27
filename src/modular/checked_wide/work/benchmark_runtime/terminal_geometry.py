"""Exact DD continuation as a 0/1 functional, without coefficient storage."""
from math import comb
import os
import numpy as np
import numba as nb
import allocation_guard as G

@nb.njit(cache=True,parallel=True)
def encode(L,h,table):
    N=table[L,h];p=(L-h)//2
    partners=np.empty((N,L),np.int8)
    unmatched=np.empty((N,h),np.int8)
    signatures=np.zeros((N,2),np.uint64)
    areas=np.empty(N,np.int16)
    for rank in nb.prange(N):
        bits=np.empty(L,np.int8);stack=np.empty(L,np.int8)
        cursor=np.empty(1,np.int64);cursor[0]=rank;hh=h
        for m in range(L,0,-1):
            up=table[m-1,hh-1] if hh>0 else 0
            if cursor[0]<up:bits[m-1]=1;hh-=1
            else:bits[m-1]=0;cursor[0]-=up;hh+=1
        top=0;area=0
        for k in range(L):
            if bits[k]:area+=top;stack[top]=k;top+=1
            else:
                top-=1;j=stack[top];partners[rank,k]=j;partners[rank,j]=k
        for k in range(h):
            position=stack[k];unmatched[rank,k]=position
            partners[rank,position]=-k-1
            cumulative=(position-k)//2
            bit=k*(p+1)+cumulative
            signatures[rank,bit//64]|=np.uint64(1)<<np.uint64(bit%64)
        areas[rank]=area
    return partners,unmatched,signatures,areas

@nb.njit(inline='always')
def connected(i,j,partners,unmatched,signatures):
    L=partners.shape[1]
    if L==1:return True
    if (signatures[i,0]&signatures[j,0]) or (signatures[i,1]&signatures[j,1]):return False
    side=0;position=0
    for count in range(1,L+1):
        other=partners[i,position] if side==0 else partners[j,position]
        if other>=0:position=other;side=1-side
        elif side==0:position=unmatched[j,-other-1]
        else:position=unmatched[i,-other-1]
        if side==0 and position==0:return count==L
    return False

@nb.njit(cache=True,parallel=True)
def mask(rows,cols,partners,unmatched,signatures,upper_only):
    out=np.empty((len(rows),len(cols)),np.uint8)
    candidates=np.zeros(len(rows),np.int64)
    for r in nb.prange(len(rows)):
        i=rows[r]
        for c in range(len(cols)):
            j=cols[c]
            if upper_only and j<i:
                out[r,c]=0;continue
            possible=partners.shape[1]==1 or not ((signatures[i,0]&signatures[j,0]) or (signatures[i,1]&signatures[j,1]))
            candidates[r]+=possible
            out[r,c]=connected(i,j,partners,unmatched,signatures) if possible else 0
    return out,candidates

@nb.njit(cache=True,parallel=True)
def build_packed(offsets,partners,unmatched,signatures,areas,area_target):
    N=len(offsets)-1;packed=np.empty(offsets[-1],np.uint64)
    candidates=np.zeros(N,np.int64)
    # Pair a long row with a short row for balanced parallel work.
    for pair in nb.prange((N+1)//2):
        first_row=np.int64(pair)
        for which in range(2):
            i=first_row if which==0 else N-1-first_row
            if which==1 and i==first_row:continue
            for word in range(offsets[i+1]-offsets[i]):
                bits=np.uint64(0);first=i+64*word
                for j in range(first,min(first+64,N)):
                    possible=((areas[i]+areas[j])&1)==area_target and (partners.shape[1]==1 or not ((signatures[i,0]&signatures[j,0]) or (signatures[i,1]&signatures[j,1])))
                    candidates[i]+=possible
                    if possible and connected(i,j,partners,unmatched,signatures):
                        bits|=np.uint64(1)<<np.uint64(j-first)
                packed[offsets[i]+word]=bits
    return packed,candidates

@nb.njit(cache=True,parallel=True)
def mask_packed(rows,cols,offsets,packed,partners,signatures,upper_only):
    out=np.empty((len(rows),len(cols)),np.uint8);candidates=np.zeros(len(rows),np.int64)
    for r in nb.prange(len(rows)):
        i=rows[r]
        for c in range(len(cols)):
            j=cols[c]
            if upper_only and j<i:out[r,c]=0;continue
            possible=partners.shape[1]==1 or not ((signatures[i,0]&signatures[j,0]) or (signatures[i,1]&signatures[j,1]))
            candidates[r]+=possible
            if not possible:out[r,c]=0;continue
            a=i if i<j else j;b=j if i<j else i;bit=b-a
            out[r,c]=(packed[offsets[a]+bit//64]>>np.uint64(bit%64))&np.uint64(1)
    return out,candidates

class Geometry:
    def __init__(self,L,h,cache=True):
        if not 1<=L<=32 or not 0<=h<=L or (L-h)%2:raise ValueError((L,h))
        if h*((L-h)//2+1)>128:raise ValueError('through signature exceeds audited 128-bit representation')
        table=np.zeros((L+1,L+2),np.int64)
        for m in range(L+1):
            for height in range(m%2,m+1,2):
                p=(m-height)//2;table[m,height]=comb(m,p)-(comb(m,p-1) if p else 0)
        self.L=L;self.h=h;self.N=int(table[L,h])
        G.check(self.N*(L+h+18),'terminal structural word metadata')
        self.partners,self.unmatched,self.signatures,self.areas=encode(L,h,table)
        self.area_target=(comb(L-1,2)+comb(h,2))&1
        self.nbytes=self.partners.nbytes+self.unmatched.nbytes+self.signatures.nbytes+self.areas.nbytes
        self.backend=os.environ.get('MEANDER_TERMINAL_BACKEND','cpu');self.device=None
        self.cache_enabled=cache and self.backend=='cpu';self.packed=None;self.offsets=None
        self.unique_pairs_tested=0;self.unique_candidates=0
    def mask(self,rows,cols,upper_only=False):
        G.check(len(rows)*len(cols)+len(rows)*8,'terminal connectivity window')
        if self.backend!='cpu' and len(rows)*len(cols)>=65536:
            if self.device is None:
                from opencl_geometry import DeviceGeometry
                self.device=DeviceGeometry(self.L,self.h,self.partners,self.unmatched,self.signatures,self.backend)
            return self.device.mask(rows,cols,upper_only)
        if self.cache_enabled:
            if self.packed is None:
                words=(self.N-np.arange(self.N,dtype=np.int64)+63)//64
                offsets=np.empty(self.N+1,np.int64);offsets[0]=0;np.cumsum(words,out=offsets[1:])
                G.check(int(offsets[-1])*8+offsets.nbytes+self.N*8,'one bit per triangular terminal state')
                self.packed,counts=build_packed(offsets,self.partners,self.unmatched,self.signatures,self.areas,self.area_target)
                self.offsets=offsets;self.unique_candidates=int(counts.sum())
                self.unique_pairs_tested=self.N*(self.N+1)//2
                self.nbytes+=self.packed.nbytes+offsets.nbytes
            return mask_packed(rows,cols,self.offsets,self.packed,self.partners,self.signatures,upper_only)
        return mask(rows,cols,self.partners,self.unmatched,self.signatures,upper_only)
    def close(self):
        if self.device is not None:self.device.close();self.device=None
