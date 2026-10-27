"""Explicit ordinary-file coefficient storage. No mmap or virtual-memory backing.

Only scheduling/storage changes: unsigned blocks keep their certified dtype;
Python-integer tail blocks stay in bounded RAM. Every read owns its buffer.
"""
import itertools, math, shutil
import gc,os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import allocation_guard as guard

class Store:
    def __init__(self, root, ram_limit=4<<30, small_limit=64<<20, chunk=64<<20, disk_reserve=32<<30):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.ram_limit=int(ram_limit);self.small_limit=int(small_limit);self.chunk=int(chunk)
        if min(self.ram_limit,self.small_limit,self.chunk)<1:raise ValueError('positive storage budgets required')
        self.disk_reserve=int(disk_reserve);self.ram=0;self.disk=0;self.counter=itertools.count()
        self.live={};self.cache_evictor=None;self.borrow_depth=0;self.add_kernel=None
        self.stats=dict(read_bytes=0,write_bytes=0,peak_ram_blocks_bytes=0,peak_disk_bytes=0,disk_blocks=0,buffer_limit_bytes=self.chunk,uses_mmap=False,pressure_spills=0,pressure_spill_bytes=0,pressure_cache_clears=0,allocation_file_fallbacks=0,ram_borrowed_read_bytes=0)
    @contextmanager
    def borrowed(self,*blocks):
        blocks=list({id(b):b for b in blocks}.values())
        for b in blocks:
            assert b.store is self and not b.closed
            b.pins+=1
        self.borrow_depth+=1
        try:yield
        finally:
            self.borrow_depth-=1
            for b in blocks:b.pins-=1
    def create(self,shape,dtype):
        b=Block(self,shape,dtype);self.live[id(b)]=b;return b
    def make_space(self,needed,overhead):
        import psutil
        budget=int(os.environ.get('MEANDER_BUDGET_BYTES','0'))
        if not budget:return
        reserve=int(os.environ['MEANDER_RESERVE_BYTES'])
        def fits():
            mem=psutil.Process().memory_info()
            return max(mem.rss,mem.vms)+needed+overhead<=budget and psutil.virtual_memory().available>=reserve+needed+overhead
        if fits():return
        if self.cache_evictor:
            self.cache_evictor();gc.collect();self.stats['pressure_cache_clears']+=1
        while not fits():
            candidates=[b for b in self.live.values() if b.array is not None and b.dtype.kind=='u' and b.pins==0]
            if not candidates:return
            b=max(candidates,key=lambda x:x.nbytes)
            b.spill();gc.collect()
    def finish(self):
        if self.disk or self.ram or self.live:raise RuntimeError('live blocks at storage close')
        self.root.rmdir()

class View:
    def __init__(self,base,rect=None,flip=False):
        self.base=base;self.flip=bool(flip)
        shape=base.shape[::-1] if flip else base.shape
        self.rect=rect or (0,shape[0],0,shape[1])
        r0,r1,c0,c1=self.rect
        if not (0<=r0<=r1<=shape[0] and 0<=c0<=c1<=shape[1]):raise IndexError(self.rect)
        self.shape=(r1-r0,c1-c0);self.dtype=base.dtype
    def read(self,r0,r1,c0,c1):
        if not (0<=r0<=r1<=self.shape[0] and 0<=c0<=c1<=self.shape[1]):raise IndexError('view rectangle')
        a,b,c,d=self.rect;r0+=a;r1+=a;c0+=c;c1+=c
        return self.base.read(c0,c1,r0,r1).T if self.flip else self.base.read(r0,r1,c0,c1)

class Block:
    def __init__(self,store,shape,dtype):
        self.store=store;self.shape=tuple(map(int,shape));self.dtype=np.dtype(dtype)
        if len(self.shape)!=2 or min(self.shape)<0:raise ValueError('matrix shape')
        self.size=math.prod(self.shape);self.nbytes=self.size*self.dtype.itemsize
        self.charge=self.size*(64 if self.dtype==object else self.dtype.itemsize)
        self.closed=False;self.array=None;self.file=None;self.path=None;self.pins=0
        want_ram=(self.charge<=store.small_limit and store.ram+self.charge<=store.ram_limit) or self.dtype==object
        if want_ram:
            if store.ram+self.charge>store.ram_limit:raise MemoryError('bigint tail exceeds explicit RAM pool')
            try:
                guard.check(self.charge,'RAM coefficient block')
                self.array=np.zeros(self.shape,self.dtype)
            except MemoryError:
                if self.dtype==object:raise
                store.stats['allocation_file_fallbacks']+=1
            if self.array is not None:
                store.ram+=self.charge
                store.stats['peak_ram_blocks_bytes']=max(store.stats['peak_ram_blocks_bytes'],store.ram)
        if self.array is None:
            self._open_file()
    def _open_file(self):
            store=self.store
            if self.dtype.kind!='u' or self.nbytes>=1<<63:raise ValueError('unsupported file dtype or offset')
            self.tile_rows=min(512,self.shape[0]);self.tile_cols=min(512,self.shape[1])
            self.grid_rows=(self.shape[0]+self.tile_rows-1)//self.tile_rows
            self.grid_cols=(self.shape[1]+self.tile_cols-1)//self.tile_cols
            self.tile_bytes=self.tile_rows*self.tile_cols*self.dtype.itemsize
            self.file_bytes=self.grid_rows*self.grid_cols*self.tile_bytes
            if self.file_bytes>=1<<63:raise ValueError('file offset exceeds int64')
            if shutil.disk_usage(store.root).free<self.file_bytes+store.disk_reserve:raise OSError('disk reserve gate')
            self.path=store.root/f'{next(store.counter):08d}.bin'
            self.file=self.path.open('x+b',buffering=0);self.file.truncate(self.file_bytes)
            store.disk+=self.file_bytes;store.stats['disk_blocks']+=1
            store.stats['peak_disk_bytes']=max(store.stats['peak_disk_bytes'],store.disk)
    def spill(self):
        if self.closed or self.array is None or self.dtype.kind!='u' or self.pins:raise RuntimeError('invalid RAM spill')
        self._open_file()
        # The allocation guard always leaves 128 MiB of working headroom.
        # At most one 2 MiB tile is allocated here. No whole-array copy.
        tile=np.zeros((self.tile_rows,self.tile_cols),self.dtype)
        for ri in range(self.grid_rows):
            for ci in range(self.grid_cols):
                br=ri*self.tile_rows;bc=ci*self.tile_cols
                nr=min(self.tile_rows,self.shape[0]-br);nc=min(self.tile_cols,self.shape[1]-bc)
                tile.fill(0);tile[:nr,:nc]=self.array[br:br+nr,bc:bc+nc]
                self._write_exact(memoryview(tile).cast('B'))
                self.store.stats['write_bytes']+=self.tile_bytes
        self.array=None;self.store.ram-=self.charge
        self.store.stats['pressure_spills']+=1;self.store.stats['pressure_spill_bytes']+=self.nbytes
    def view(self,rect=None,flip=False):return View(self,rect,flip)
    def _validate(self,r0,r1,c0,c1):
        if self.closed or not (0<=r0<=r1<=self.shape[0] and 0<=c0<=c1<=self.shape[1]):raise IndexError('closed or invalid block rectangle')
    def _read_exact(self,buf):
        offset=0
        while offset<len(buf):
            got=self.file.readinto(buf[offset:])
            if not got:raise EOFError(('truncated coefficient block',str(self.path)))
            offset+=got
    def _write_exact(self,buf):
        offset=0
        while offset<len(buf):
            got=self.file.write(buf[offset:])
            if not got:raise OSError('short coefficient write')
            offset+=got
    def read(self,r0,r1,c0,c1):
        self._validate(r0,r1,c0,c1)
        if self.array is not None and self.pins and self.store.borrow_depth:
            view=self.array[r0:r1,c0:c1]
            self.store.stats['ram_borrowed_read_bytes']+=view.nbytes
            return view
        shape=(r1-r0,c1-c0);guard.array(shape,self.dtype,'coefficient read buffer')
        if self.array is not None:return self.array[r0:r1,c0:c1].copy()
        out=np.empty(shape,self.dtype);guard.check(self.tile_bytes,'tile read buffer')
        tile=np.empty((self.tile_rows,self.tile_cols),self.dtype)
        for offset,rr0,rr1,cc0,cc1,br,bc in self._tiles(r0,r1,c0,c1):
            self.file.seek(offset);self._read_exact(memoryview(tile).cast('B'))
            out[rr0-r0:rr1-r0,cc0-c0:cc1-c0]=tile[rr0-br:rr1-br,cc0-bc:cc1-bc]
            self.store.stats['read_bytes']+=self.tile_bytes
        return out
    def write(self,r0,r1,c0,c1,value):
        self._validate(r0,r1,c0,c1)
        if value.shape!=(r1-r0,c1-c0) or value.dtype!=self.dtype:raise ValueError('typed rectangle write mismatch')
        if self.array is not None:
            target=self.array[r0:r1,c0:c1]
            if target.ctypes.data!=value.ctypes.data or target.strides!=value.strides:target[:]=value
            return
        guard.check(self.tile_bytes,'tile write buffer');tile=np.empty((self.tile_rows,self.tile_cols),self.dtype)
        for offset,rr0,rr1,cc0,cc1,br,bc in self._tiles(r0,r1,c0,c1):
            if rr0==br and rr1==br+self.tile_rows and cc0==bc and cc1==bc+self.tile_cols:
                tile[:]=value[rr0-r0:rr1-r0,cc0-c0:cc1-c0]
            else:
                self.file.seek(offset);self._read_exact(memoryview(tile).cast('B'))
                self.store.stats['read_bytes']+=self.tile_bytes
                tile[rr0-br:rr1-br,cc0-bc:cc1-bc]=value[rr0-r0:rr1-r0,cc0-c0:cc1-c0]
            self.file.seek(offset);self._write_exact(memoryview(tile).cast('B'))
            self.store.stats['write_bytes']+=self.tile_bytes
    def _tiles(self,r0,r1,c0,c1):
        for ri in range(r0//self.tile_rows,(r1+self.tile_rows-1)//self.tile_rows):
            br=ri*self.tile_rows
            for ci in range(c0//self.tile_cols,(c1+self.tile_cols-1)//self.tile_cols):
                bc=ci*self.tile_cols
                yield (ri*self.grid_cols+ci)*self.tile_bytes,max(r0,br),min(r1,br+self.tile_rows),max(c0,bc),min(c1,bc+self.tile_cols),br,bc
    def rectangles(self):
        width=64 if self.dtype==object else self.dtype.itemsize
        rows=max(1,self.store.chunk//max(1,self.shape[1]*width))
        for r0 in range(0,self.shape[0],rows):yield r0,min(r0+rows,self.shape[0]),0,self.shape[1]
    def maximum_reduce(self,mod=None):
        mx=0
        for rect in self.rectangles():
            A=self.read(*rect);mx=max(mx,int(A.max(initial=0)))
            if mod is not None:np.remainder(A,self.dtype.type(mod),out=A);self.write(*rect,A)
            del A
        return mx
    def close(self):
        if self.closed:raise RuntimeError('double coefficient close')
        if self.pins:raise RuntimeError('closing a borrowed RAM block')
        if self.file is not None:
            self.file.close()
            if self.path.resolve().parent!=self.store.root:raise RuntimeError('unsafe cleanup path')
            self.path.unlink();self.store.disk-=self.file_bytes
        else:self.array=None;self.store.ram-=self.charge
        self.closed=True
        self.store.live.pop(id(self),None)

def add(T,V,off=(0,0)):
    nr,nc=V.shape;tr,tc=off
    width=max(T.dtype.itemsize,V.dtype.itemsize,64 if T.dtype==object or V.dtype==object else 0)
    rows=max(1,T.store.chunk//max(1,nc*width))
    for r0 in range(0,nr,rows):
        r1=min(r0+rows,nr);A=T.read(tr+r0,tr+r1,tc,tc+nc);B=V.read(r0,r1,0,nc)
        # T's BEFORE-addition certificate bounds every nonnegative partial sum.
        if T.store.add_kernel is not None and A.dtype!=object and B.dtype!=object and A.size>=100000:
            T.store.add_kernel(A,B)
        else:np.add(A,B,out=A,casting='unsafe')
        T.write(tr+r0,tr+r1,tc,tc+nc,A)
        del A,B

def scatter(T,V,kind,maps,mt):
    axis=0 if kind=='rows' else 1
    if len(maps)!=V.shape[axis] or (len(maps) and (int(maps.min())<0 or int(maps.max())>=T.shape[axis])):raise ValueError('invalid closing map')
    width=max(T.dtype.itemsize,V.dtype.itemsize,64 if T.dtype==object or V.dtype==object else 0)
    if kind=='cols':
        step=max(1,T.store.chunk//max(1,max(T.shape[1],V.shape[1])*width))
        for r0 in range(0,V.shape[0],step):
            r1=min(r0+step,V.shape[0]);A=T.read(r0,r1,0,T.shape[1]);B=V.read(r0,r1,0,V.shape[1])
            if B.dtype==object and A.dtype!=object:
                if int(B.min(initial=0))<0 or int(B.max(initial=0))>np.iinfo(A.dtype).max:raise ArithmeticError('bigint source exceeds certified native destination')
                guard.array(B.shape,A.dtype,'checked bigint scatter cast');B=B.astype(A.dtype)
            if A.dtype==object:
                for j,k in enumerate(maps):A[:,k]+=B[:,j]
            else:mt.scatter_cols_parallel(A,maps,B)
            T.write(r0,r1,0,T.shape[1],A);del A,B
    elif kind=='rows':
        step=max(1,T.store.chunk//max(1,max(T.shape[0],V.shape[0])*width))
        for c0 in range(0,V.shape[1],step):
            c1=min(c0+step,V.shape[1]);A=T.read(0,T.shape[0],c0,c1);B=V.read(0,V.shape[0],c0,c1)
            if B.dtype==object and A.dtype!=object:
                if int(B.min(initial=0))<0 or int(B.max(initial=0))>np.iinfo(A.dtype).max:raise ArithmeticError('bigint source exceeds certified native destination')
                guard.array(B.shape,A.dtype,'checked bigint scatter cast');B=B.astype(A.dtype)
            if A.dtype==object:
                for i,k in enumerate(maps):A[k,:]+=B[i,:]
            else:mt.scatter_rows_parallel(A,maps,B)
            T.write(0,T.shape[0],c0,c1,A);del A,B
    else:raise ValueError(kind)

def features_slice(f,start,stop):
    # Offsets remain absolute indexes into the retained full sparse arrays.
    sums,off,levels,values=f
    return sums[start:stop],off[start:stop+1],levels,values
