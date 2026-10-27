"""Physically omit only a proved unobservable boundary rectangle.

Three ordinary blocks retain the live regions. There is no full square
allocation, no sparse lookup per scalar, and no alteration of arithmetic.
"""
import math
import numpy as np
import allocation_guard as guard
from bounded_storage import View,add as regular_add,scatter as regular_scatter

class SubView:
    def __init__(self,base,r0,r1,c0,c1):
        self.base=base;self.r0=r0;self.c0=c0
        self.shape=(r1-r0,c1-c0);self.dtype=base.dtype
    def read(self,r0,r1,c0,c1):
        return self.base.read(self.r0+r0,self.r0+r1,self.c0+c0,self.c0+c1)

class PackedBoundary:
    @property
    def pins(self):return self._pins
    @pins.setter
    def pins(self,value):
        delta=value-getattr(self,'_pins',0)
        assert value>=0
        for *_,part in getattr(self,'regions',[]):part.pins+=delta
        self._pins=value
    def __init__(self,store,shape,dtype,u):
        self.store=store;self.shape=tuple(shape);self.dtype=np.dtype(dtype)
        assert len(shape)==2 and shape[0]==shape[1] and 0<=u<=shape[0]
        self.u=int(u);self.size=math.prod(shape);self.logical_nbytes=self.size*self.dtype.itemsize
        self.closed=False;self.pins=0;self.array=None;self.is_boundary_packed=True
        self.regions=[];self.blocks={}
        N=shape[0]
        for name,r0,r1,c0,c1 in (('TR',0,u,u,N),('BL',u,N,0,u),('BR',u,N,u,N)):
            if r0==r1 or c0==c1:continue
            part=store.create((r1-r0,c1-c0),dtype)
            self.regions.append((r0,r1,c0,c1,part));self.blocks[name]=part
        self.active_size=self.size-u*u
        self.nbytes=sum(p.nbytes for *_,p in self.regions)
        self.charge=sum(p.charge for *_,p in self.regions)
        assert self.nbytes==self.active_size*self.dtype.itemsize
    def view(self,rect=None,flip=False):return View(self,rect,flip)
    def _validate(self,r0,r1,c0,c1):
        assert not self.closed and 0<=r0<=r1<=self.shape[0] and 0<=c0<=c1<=self.shape[1]
    def intersections(self,r0,r1,c0,c1):
        self._validate(r0,r1,c0,c1)
        for br0,br1,bc0,bc1,part in self.regions:
            a,b=max(r0,br0),min(r1,br1);c,d=max(c0,bc0),min(c1,bc1)
            if a<b and c<d:yield a,b,c,d,br0,bc0,part
    def read(self,r0,r1,c0,c1):
        self._validate(r0,r1,c0,c1)
        if r0<r1 and c0<c1 and r0<self.u and c0<self.u:
            raise ArithmeticError('attempt to read a proved unobservable boundary region')
        for br0,br1,bc0,bc1,part in self.regions:
            if br0<=r0<=r1<=br1 and bc0<=c0<=c1<=bc1:
                # The owner's borrow pins all its children, so a RAM view is
                # safe until the retained transfer finishes using it.
                return part.read(r0-br0,r1-br0,c0-bc0,c1-bc0)
        guard.array((r1-r0,c1-c0),self.dtype,'packed boundary read buffer')
        # Zero denotes an omitted, unobservable value, not its historical
        # forward coefficient. Valid outgoing rectangles never intersect it.
        out=np.empty((r1-r0,c1-c0),self.dtype);covered=0
        for a,b,c,d,br,bc,part in self.intersections(r0,r1,c0,c1):
            out[a-r0:b-r0,c-c0:d-c0]=part.read(a-br,b-br,c-bc,d-bc)
            covered+=(b-a)*(d-c)
        assert covered==out.size
        return out
    def write(self,r0,r1,c0,c1,value):
        self._validate(r0,r1,c0,c1)
        assert value.shape==(r1-r0,c1-c0) and value.dtype==self.dtype
        for a,b,c,d,br,bc,part in self.intersections(r0,r1,c0,c1):
            part.write(a-br,b-br,c-bc,d-bc,value[a-r0:b-r0,c-c0:d-c0])
    def rectangles(self):
        width=64 if self.dtype==object else self.dtype.itemsize
        for r0,r1,c0,c1,_ in self.regions:
            step=max(1,self.store.chunk//max(1,(c1-c0)*width))
            for a in range(r0,r1,step):yield a,min(r1,a+step),c0,c1
    def maximum_reduce(self,mod=None):
        assert mod is None
        return max((p.maximum_reduce() for *_,p in self.regions),default=0)
    def close(self):
        assert not self.closed and self.pins==0
        for *_,p in self.regions:p.close()
        self.closed=True
    def spill(self):
        for *_,p in self.regions:
            if p.array is not None and p.dtype.kind=='u':p.spill()

def packed_add(T,V,off=(0,0)):
    if not isinstance(T,PackedBoundary):return regular_add(T,V,off)
    tr,tc=off;nr,nc=V.shape
    for a,b,c,d,br,bc,part in T.intersections(tr,tr+nr,tc,tc+nc):
        source=SubView(V,a-tr,b-tr,c-tc,d-tc)
        with T.store.borrowed(part):regular_add(part,source,(a-br,c-bc))

def packed_scatter(T,V,kind,maps,backend):
    if not isinstance(T,PackedBoundary):return regular_scatter(T,V,kind,maps,backend)
    # Closing the top unmatched opening makes the destination word end
    # in a closing symbol. Its rank is therefore >= the opening prefix u.
    # This exact invariant permits contiguous destinations and no hole buffer.
    if len(maps):assert int(maps.min())>=T.u,'cap unexpectedly enters an omitted region'
    guard.check(maps.nbytes,'packed shifted closing map')
    shifted=maps-T.u
    N=T.shape[0];u=T.u
    if kind=='cols':
        assert V.shape[0]==N
        for name,a,b in (('TR',0,u),('BR',u,N)):
            if a==b:continue
            part=T.blocks[name];source=SubView(V,a,b,0,V.shape[1])
            with T.store.borrowed(part):regular_scatter(part,source,kind,shifted,backend)
    elif kind=='rows':
        assert V.shape[1]==N
        for name,c,d in (('BL',0,u),('BR',u,N)):
            if c==d:continue
            part=T.blocks[name];source=SubView(V,0,V.shape[0],c,d)
            with T.store.borrowed(part):regular_scatter(part,source,kind,shifted,backend)
    else:raise ValueError(kind)

def should_pack(n,t,key):
    a,b,h=key
    return t<2*n-1 and a==b==2*n-t and h>=1
