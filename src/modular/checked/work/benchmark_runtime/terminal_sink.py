"""Immediate final contribution of a boundary input; zero stored coefficients."""
import time
import math
import numpy as np
import allocation_guard as G
import engine_v2 as E
import incoming_e1 as I
import recursive_profiles as RP
import projected_source as P3
import pair_reader as PR
import pair_kernels as PK
import source_expressions as SE
from terminal_geometry import Geometry

def boundary(n,t,key):return key[0]==key[1]==2*n-t

class Accumulator:
    def __init__(self):
        self.total=0
        self.stats=dict(blocks=0,geometry_seconds=0.,connectivity_seconds=0.,pair_value_seconds=0.,reduction_seconds=0.,geometric_pairs_tested=0,passed_through_criterion=0,productive_pair_evaluations=0,source_pair_evaluations=0,input_pair_evaluations=0,largest_mask_cells=0,stored_boundary_coefficient_bytes=0,geometry_live_bytes=0,geometry_peak_bytes=0,structural_cache_live_bytes=0,structural_cache_peak_bytes=0,unique_cached_pairs_tested=0,unique_cached_candidates=0)

class Sink:
    is_terminal_sink=True
    def __init__(self,n,t,key,cap,store,acc):
        self.n=n;self.t=t;self.key=key;self.cap=int(cap);self.store=store;self.acc=acc
        L=key[0];h=key[2];N=E.mt.B(L,h)
        self.shape=(N,N);self.size=N*N;self.dtype=np.dtype(E.dtype_for(cap,'mixed'))
        self.nbytes=0;self.charge=0;self.array=None;self.closed=False;self.pins=0
        self.logical_nbytes=self.size*self.dtype.itemsize;self.active_size=0
        e=(t-2*L+h)//2
        start=time.perf_counter();self.geometry=Geometry(L,h,cache=(e==2 or (e==3 and h>0)))
        self.cache_charge=0
        acc.stats['geometry_seconds']+=time.perf_counter()-start;acc.stats['blocks']+=1
        acc.stats['geometry_live_bytes']+=self.geometry.nbytes
        acc.stats['geometry_peak_bytes']=max(acc.stats['geometry_peak_bytes'],acc.stats['geometry_live_bytes'])
    def project(self,dest_rows,dest_cols,value_function,analytic=False,scale=1):
        # The proved area parity partitions the Cartesian domain into two
        # allowed rectangles; incompatible rectangles are never visited.
        areas=self.geometry.areas;wanted=self.geometry.area_target
        for parity in (0,1):
            ri=np.flatnonzero((areas[dest_rows]&1)==parity)
            ci=np.flatnonzero((areas[dest_cols]&1)==(parity^wanted))
            if not len(ri) or not len(ci):continue
            self.project_group(dest_rows[ri],dest_cols[ci],lambda rr,cc:value_function(ri[rr],ci[cc]),analytic,scale)
    def project_group(self,dest_rows,dest_cols,value_function,analytic=False,scale=1):
        if not len(dest_rows) or not len(dest_cols):return
        if min(int(dest_rows.min()),int(dest_cols.min()))<0 or max(int(dest_rows.max()),int(dest_cols.max()))>=self.shape[0]:raise IndexError('terminal projection')
        maxcells=max(1,self.store.chunk//64)
        # Respect the existing 512x512 file-tile geometry. A very wide mask
        # with only a few source rows would reread each file tile many times.
        width=min(len(dest_cols),512,maxcells)
        for c0 in range(0,len(dest_cols),width):
            cc=dest_cols[c0:c0+width];step=max(1,maxcells//len(cc))
            for r0 in range(0,len(dest_rows),step):
                rr=dest_rows[r0:r0+step]
                G.check(2*self.store.chunk,'terminal mask, pair indices and exact reduction')
                start=time.perf_counter();previous_bytes=self.geometry.nbytes
                mask,candidates=self.geometry.mask(rr,cc,upper_only=analytic)
                extra=self.geometry.nbytes-previous_bytes
                if extra:
                    self.cache_charge+=extra
                    self.acc.stats['geometry_live_bytes']+=extra
                    self.acc.stats['geometry_peak_bytes']=max(self.acc.stats['geometry_peak_bytes'],self.acc.stats['geometry_live_bytes'])
                    self.acc.stats['structural_cache_live_bytes']+=extra
                    self.acc.stats['structural_cache_peak_bytes']=max(self.acc.stats['structural_cache_peak_bytes'],self.acc.stats['structural_cache_live_bytes'])
                    self.acc.stats['unique_cached_pairs_tested']+=self.geometry.unique_pairs_tested
                    self.acc.stats['unique_cached_candidates']+=self.geometry.unique_candidates
                ir,jc=np.nonzero(mask)
                self.acc.stats['connectivity_seconds']+=time.perf_counter()-start
                self.acc.stats['geometric_pairs_tested']+=mask.size
                self.acc.stats['passed_through_criterion']+=int(candidates.sum())
                self.acc.stats['largest_mask_cells']=max(self.acc.stats['largest_mask_cells'],mask.size)
                del mask,candidates
                if not len(ir):continue
                start=time.perf_counter();values=value_function(ir+r0,jc+c0)
                if len(values)!=len(ir):raise ValueError('terminal value vector')
                if values.dtype==object:
                    if any(int(x)<0 or int(x)>self.cap for x in values):raise ArithmeticError('terminal bigint contribution bound')
                elif int(values.max(initial=0))>self.cap:raise ArithmeticError('terminal contribution exceeds existing U')
                self.acc.stats['pair_value_seconds']+=time.perf_counter()-start
                start=time.perf_counter()
                if analytic:
                    off_diagonal=dest_rows[ir+r0]!=dest_cols[jc+c0]
                    if values.dtype==object:amount=sum(int(x)*(2 if off else 1) for x,off in zip(values,off_diagonal))
                    else:
                        lo,hi=PK.sum128_folded(values,off_diagonal);amount=int(lo)+(int(hi)<<64)
                elif values.dtype==object:amount=sum(int(x) for x in values)
                else:
                    lo,hi=PK.sum128(values);amount=int(lo)+(int(hi)<<64)
                self.acc.total+=scale*amount
                self.acc.stats['reduction_seconds']+=time.perf_counter()-start
                self.acc.stats['productive_pair_evaluations']+=len(values)
                counter='source_pair_evaluations' if analytic else 'input_pair_evaluations'
                self.acc.stats[counter]+=len(values)
                del values,ir,jc
    def add(self,V,off=(0,0),scale=1):
        nr,nc=V.shape;tr,tc=off
        U=E.mt.B(self.key[0]-1,self.key[2]-1)
        if self.key[0]>1 and tr+nr<=U and tc+nc<=U:return
        rows=np.arange(nr,dtype=np.int64)+tr;cols=np.arange(nc,dtype=np.int64)+tc
        self.project(rows,cols,lambda rr,cc:PR.read(V,rr,cc),scale=scale)
    def scatter(self,V,kind,maps,backend,scale=1):
        nr,nc=V.shape
        rows=maps if kind=='rows' else np.arange(nr,dtype=np.int64)
        cols=maps if kind=='cols' else np.arange(nc,dtype=np.int64)
        if len(rows)!=nr or len(cols)!=nc:raise ValueError('terminal closing map shape')
        self.project(rows,cols,lambda rr,cc:PR.read(V,rr,cc),scale=scale)
    def maximum_reduce(self,mod=None):
        if mod is not None:raise ValueError('CRT is not part of the terminal experiment')
        # An upper bound only. Boundary maxima never feed an interior transition.
        return self.cap
    def close(self):
        if self.closed or self.pins:raise RuntimeError('invalid sink close')
        self.acc.stats['geometry_live_bytes']-=self.geometry.nbytes
        self.acc.stats['structural_cache_live_bytes']-=self.cache_charge
        self.geometry.close()
        self.geometry=None;self.closed=True
    def read(self,*args):raise RuntimeError('a terminal functional has no stored coefficient grid')
    def view(self,*args):raise RuntimeError('terminal boundary must have no outgoing transfer')

def inject(n,key,e,T):
    a,b,h=key;C=E.old.choose(n);certificate=I.source_certificate(key,e)
    if certificate>=1<<64:raise ArithmeticError('analytic source uint64 certificate unavailable')
    if e==3:
        ip,P,fp,_=P3.closed(a,h);iq,Q,fq,_=P3.closed(b,h)
        def values(rows,cols):return SE.values3(rows,cols,ip,iq,P,Q,*fp,*fq,C,certificate)
    else:
        ip,P=E.ms.profiles(a,h);iq,Q=E.ms.profiles(b,h)
        fp=RP.features(a,h);fq=RP.features(b,h)
        dp=I.degree_codes(a,h);dq=I.degree_codes(b,h);H=I.hockey(n)
        def values(rows,cols):return SE.values2(rows,cols,ip,iq,P,Q,*fp,*fq,dp,dq,C,H,certificate)
    def checked(rows,cols):
        result,errors=values(rows,cols)
        if np.any(errors):raise ArithmeticError('matrix-free analytic source certificate')
        return result
    T.project(np.arange(T.shape[0],dtype=np.int64),np.arange(T.shape[1],dtype=np.int64),checked,analytic=True)
    return T.cap
