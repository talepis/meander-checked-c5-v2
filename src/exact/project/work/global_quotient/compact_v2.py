"""Experimental exact-integer V2 interpolation. The bilinear closure is conjectural.

Only anchor cross sections and a feature-basis cross section are propagated.
Bounds use the explicitly conjectural extremal recurrence, never known Mn.
This first prototype intentionally stops if a compressed section spills to file.
"""
import gc,hashlib,inspect,json,math,time
from pathlib import Path
from functools import lru_cache
import numpy as np
import numba as nb
import engine_v2 as E
import allocation_guard as guard
import recursive_profiles as rp
import source_separation as S
from bounded_storage import View,Block
from experiments import rational_basis,basis_pivots
BASE=Path(__file__).resolve().parents[2];CACHE=BASE/'work/global_quotient/basis_cache'
CACHE_SIGNATURE=hashlib.sha256((inspect.getsource(rational_basis)+inspect.getsource(basis_pivots.py_func)+'e8083a7282e73c119d6cdd1bfdc890c2e2baf61186d35c570196de3771ea41d5').encode()).hexdigest()

STATS={}
GROUP_CACHE_BYTES=0
GROUP_CACHE_LIMIT=32<<20
def clear():
 global GROUP_CACHE_BYTES
 assert GROUP_CACHE_BYTES==0,'live grouped-read caches at study boundary'
 prefix.cache_clear();STATS.clear();STATS.update(blocks=0,logical_bytes=0,sample_bytes=0,prepare_seconds=0.,sample_seconds=0.,read_seconds=0.,decoded_cells=0,basis_cache_builds=0,basis_cache_loads=0,grouped_reads=0,column_group_seconds=0.,group_cache_hits=0)

@nb.njit(cache=True)
def make_features(SS,off,lev,val,h):
 X=np.zeros((len(SS),h+2),np.int64)
 for i in range(len(SS)):
  X[i,0]=SS[i]
  for z in range(off[i],off[i+1]):X[i,lev[z]+1]=val[z]
 return X

@nb.njit(cache=True)
def verify_coordinates(X,ids,W,ranks,den,starts,reps,anchors):
 for i in range(len(X)):
  g=ids[i]
  for k in range(X.shape[1]):
   value=np.int64(0)
   for u in range(ranks[g]):value+=W[i,u]*(X[reps[starts[g]+u],k]-X[anchors[g],k])
   if value!=(X[i,k]-X[anchors[g],k])*den[g]:return False
 return True

def profile_counts(a,h):
 # Constructive feature-rank lemma, recorded in FEATURE_RANK_PROOF_GR.txt.
 p=(a-h)//2;d=h+1
 if p==0:return 1,0
 groups=math.comb(p+d,d)-1;total_rank=0
 for r in range(1,p+1):
  k=p-r;count=math.comb(r+d-1,d-1)
  occupied=d*math.comb(r+d-2,d-1)
  if k==0:continue
  if k==1:total_rank+=occupied-count
  elif r==1:total_rank+=d*(1 if k==2 else 2)
  else:total_rank+=occupied+count
 return groups,total_rank

def check_profile_ranks(P,ranks,a,h):
 roots=np.sum(P.astype(np.int64),axis=1);k=(a-h)//2-roots
 occupied=np.count_nonzero(P,axis=1)
 wanted=np.where(k==0,0,np.where(k==1,occupied-1,np.where(roots==1,np.where(k==2,1,2),occupied+1)))
 assert np.array_equal(ranks,wanted),'constructive feature-rank formula disagrees with certified basis'
 assert (len(P),int(ranks.sum()))==profile_counts(a,h)

@lru_cache(None)
def prefix(a,h):
 start=time.perf_counter();ids,P=E.ms.profiles(a,h);SS,off,lev,val=rp.features(a,h)
 N=len(ids);G=len(P);guard.array((N,h+2),np.int64,'V2 feature basis')
 X=make_features(SS,off,lev,val,h)
 CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f'p{a}_h{h}_{CACHE_SIGNATURE[:12]}.npz';meta=path.with_suffix('.json')
 if path.exists() and meta.exists():
  m=json.loads(meta.read_text());assert m['a']==a and m['h']==h and m['signature']==CACHE_SIGNATURE and m['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
  guard.check(m['array_bytes'],'cached V2 basis')
  with np.load(path,allow_pickle=False) as data:result=tuple(data[f'x{k}'] for k in range(7))
  rid,W,ranks,den,starts,reps,anchors=result
  assert np.array_equal(rid,ids) and W.shape==(N,h+2) and ranks.shape==den.shape==starts.shape==anchors.shape==(G,)
  assert np.all(den>0) and np.all(ranks>=0) and np.all(ranks<=h+2) and len(reps)==int(ranks.sum())
  assert np.all((reps>=0)&(reps<N)) and np.all((anchors>=0)&(anchors<N)) and np.array_equal(starts,np.concatenate(([0],np.cumsum(ranks)[:-1])))
  assert verify_coordinates(X,*result),'cached rational basis does not reconstruct the complete feature matrix'
  check_profile_ranks(P,ranks,a,h)
  STATS['basis_cache_loads']+=1;STATS['prepare_seconds']+=time.perf_counter()-start;return result
 order=np.argsort(ids,kind='stable');count=np.bincount(ids,minlength=G);ends=np.concatenate(([0],np.cumsum(count)))
 guard.array((N,h+2),np.int64,'V2 basis coordinates')
 W=np.zeros((N,h+2),np.int64);ranks=np.zeros(G,np.int32);den=np.ones(G,np.int64);starts=np.zeros(G,np.int64)
 anchors=np.empty(G,np.int64);reps=[]
 for g in range(G):
  ii=order[ends[g]:ends[g+1]];anchors[g]=ii[0]
  Wi,d,rr=rational_basis(X[ii]-X[ii[0]])
  ranks[g]=len(rr);den[g]=d;starts[g]=len(reps);W[ii,:len(rr)]=Wi;reps.extend(ii[rr])
 result=(ids.astype(np.int32),W,ranks,den,starts,np.array(reps,np.int64),anchors)
 assert verify_coordinates(X,*result)
 check_profile_ranks(P,ranks,a,h)
 temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**{f'x{k}':v for k,v in enumerate(result)});temporary.replace(path)
 meta.write_text(json.dumps(dict(a=a,h=h,signature=CACHE_SIGNATURE,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_bytes=sum(v.nbytes for v in result),all_feature_coordinates_checked=True),indent=2),encoding='utf-8')
 STATS['basis_cache_builds']+=1
 STATS['prepare_seconds']+=time.perf_counter()-start
 return result

def selected(a,b,h):
 N,M=E.mt.B(a,h),E.mt.B(b,h)
 if N*M<100000:return False
 gp,rp=profile_counts(a,h);gq,rq=profile_counts(b,h)
 cells=N*gq+gp*M+rp*rq+gp*gq
 return cells<.65*N*M

@nb.njit(cache=True,parallel=True)
def add_sample(values,parent,rows,cols,cap):
 # NumPy int64 + uint64 promotes to float64. Keep this operation wholly integer.
 errors=np.zeros(len(rows),np.int32);limit=np.uint64(cap)
 for ii in nb.prange(len(rows)):
  i=rows[ii]
  for jj in range(len(cols)):
   j=cols[jj];v=np.uint64(parent[ii,jj]);old=values[i,j]
   if old<0 or v>limit or np.uint64(old)>limit-v:errors[ii]=1
   else:values[i,j]=old+np.int64(v)
 return errors

@nb.njit(cache=True,parallel=True)
def source_values(R,C,ip,iq,rows,cols):
 out=np.empty((len(rows),len(cols)),np.int64)
 for ii in nb.prange(len(rows)):
  i=rows[ii]
  for jj in range(len(cols)):
   j=cols[jj];out[ii,jj]=np.int64(R[i,iq[j]])+np.int64(C[ip[i],j])
 return out

@nb.njit(cache=True,parallel=True)
def encode_cross(H,R,C,anchorsp,ip,iq,repsp,repsq,bias):
 errors=np.zeros(len(repsp),np.int32)
 for i in nb.prange(len(repsp)):
  pi=repsp[i];gp=ip[pi]
  for j in range(len(repsq)):
   qj=repsq[j];gq=iq[qj]
   v=np.int64(H[i,j])-np.int64(R[pi,gq])-np.int64(C[gp,qj])+np.int64(R[anchorsp[gp],gq])+bias
   if v<0:errors[i]=1;H[i,j]=0
   else:H[i,j]=v
 return errors

@nb.njit(inline='always')
def integer_division(value,denominator):
 # For signed integers, arithmetic right shift is floor division by 2^k.
 # The bit mask is the nonnegative remainder, including negative values.
 if denominator==1:return value,0
 if denominator==2:return value>>1,value&1
 if denominator==4:return value>>2,value&3
 if denominator==8:return value>>3,value&7
 if denominator==16:return value>>4,value&15
 return value//denominator,value%denominator

@nb.njit(cache=True,parallel=True)
def decode(R,C,H,anchorsp,ip,wp,rp_,dp,op,iq,wq,rq,dq,oq,rows,cols,bias,cap,dtype_template):
 out=np.empty((len(rows),len(cols)),dtype_template.dtype)
 errors=np.zeros(len(rows),np.int32)
 for ii in nb.prange(len(rows)):
  i=rows[ii];gp=ip[i]
  for jj in range(len(cols)):
   j=cols[jj];gq=iq[j];cross=np.int64(0)
   for u in range(rp_[gp]):
    wu=wp[i,u]
    if wu:
     for v in range(rq[gq]):
      wv=wq[j,v]
      if wv:cross+=wu*wv*(np.int64(H[op[gp]+u,oq[gq]+v])-bias)
   denominator=dp[gp]*dq[gq]
   quotient,remainder=integer_division(cross,denominator)
   value=np.int64(R[i,gq])+np.int64(C[gp,j])-np.int64(R[anchorsp[gp],gq])+quotient
   if remainder or value<0 or value>cap:errors[ii]=1;out[ii,jj]=0
   else:out[ii,jj]=value
 return out,errors

@nb.njit(cache=True,parallel=True)
def decode_grouped(R,C,H,anchorsp,ip,wp,rp_,dp,op,iq,wq,rq,dq,oq,rows,cols,order,ends,bias,cap,dtype_template):
 # Associate Wp*H once per row and Q profile, then reuse it for all Q columns.
 # All products retain the same signed integer overflow certificate as decode.
 out=np.empty((len(rows),len(cols)),dtype_template.dtype)
 errors=np.zeros(len(rows),np.int32)
 for ii in nb.prange(len(rows)):
  i=rows[ii];gp=ip[i];temp=np.empty(wq.shape[1],np.int64)
  for group in range(len(ends)-1):
   gq=iq[cols[order[ends[group]]]]
   for v in range(rq[gq]):
    total=np.int64(0)
    for u in range(rp_[gp]):
     wu=wp[i,u]
     if wu:total+=wu*(np.int64(H[op[gp]+u,oq[gq]+v])-bias)
    temp[v]=total
   denominator=dp[gp]*dq[gq]
   base0=np.int64(R[i,gq])-np.int64(R[anchorsp[gp],gq])
   for z in range(ends[group],ends[group+1]):
    jj=order[z];j=cols[jj];cross=np.int64(0)
    for v in range(rq[gq]):cross+=temp[v]*wq[j,v]
    quotient,remainder=integer_division(cross,denominator)
    value=base0+np.int64(C[gp,j])+quotient
    if remainder or value<0 or value>cap:errors[ii]=1;out[ii,jj]=0
    else:out[ii,jj]=value
 return out,errors

@nb.njit(cache=True)
def interval_bound(R,C,H,anchorsp,ip,wp,rp_,dp,op,iq,wq,rq,dq,oq,bias):
 # Historical coarse bound, retained for investigation but no longer invoked.
 ng,mg=len(anchorsp),R.shape[1];rm=np.zeros((ng,mg),np.int64);cm=np.zeros((ng,mg),np.int64)
 awp=np.zeros((ng,wp.shape[1]),np.int64);awq=np.zeros((mg,wq.shape[1]),np.int64)
 for i in range(len(ip)):
  g=ip[i]
  for q in range(mg):rm[g,q]=max(rm[g,q],np.int64(R[i,q]))
  for u in range(rp_[g]):awp[g,u]=max(awp[g,u],abs(wp[i,u]))
 for j in range(len(iq)):
  g=iq[j]
  for p in range(ng):cm[p,g]=max(cm[p,g],np.int64(C[p,j]))
  for v in range(rq[g]):awq[g,v]=max(awq[g,v],abs(wq[j,v]))
 maximum=np.int64(0)
 for p in range(ng):
  for q in range(mg):
   cross=np.int64(0)
   for u in range(rp_[p]):
    for v in range(rq[q]):cross+=awp[p,u]*awq[q,v]*abs(np.int64(H[op[p]+u,oq[q]+v])-bias)
   d=dp[p]*dq[q]
   maximum=max(maximum,rm[p,q]+cm[p,q]-np.int64(R[anchorsp[p],q])+(cross+d-1)//d)
 return maximum

def indexed(A,rows,cols):
 guard.array((len(rows),len(cols)),A.dtype,'V2 sampled predecessor read')
 if isinstance(A,Compact):return A.values(rows,cols)
 if A.array is None:
  out=np.empty((len(rows),len(cols)),A.dtype)
  for tr in np.unique(rows//A.tile_rows):
   ii=np.nonzero(rows//A.tile_rows==tr)[0];br=int(tr)*A.tile_rows;nr=min(A.tile_rows,A.shape[0]-br)
   for tc in np.unique(cols//A.tile_cols):
    jj=np.nonzero(cols//A.tile_cols==tc)[0];bc=int(tc)*A.tile_cols;nc=min(A.tile_cols,A.shape[1]-bc)
    tile=A.read(br,br+nr,bc,bc+nc);out[np.ix_(ii,jj)]=tile[np.ix_(rows[ii]-br,cols[jj]-bc)]
  return out
 with A.store.borrowed(A):return A.array[np.ix_(rows,cols)]

class Compact:
 def __init__(self,n,key,U,cur,maxima,store):
  start=time.perf_counter();self.store=store;self.closed=False;self.pins=0;self.group_cache=None
  a,b,h=key;self.shape=(E.mt.B(a,h),E.mt.B(b,h));self.size=math.prod(self.shape);self.dtype=np.dtype(E.dtype_for(U,'mixed'));self.nbytes=self.size*self.dtype.itemsize
  self.p=prefix(a,h);self.q=prefix(b,h);ip,wp,rp_,dp,op,repsp,anchorsp=self.p;iq,wq,rq,dq,oq,repsq,anchorsq=self.q
  assert int(dp.max(initial=1))*int(dq.max(initial=1))<2**62,'V2 denominator product overflow'
  maxproduct=int(np.abs(wp).max(initial=0))*int(np.abs(wq).max(initial=0))*(h+2)**2*4*int(U)
  assert maxproduct<2**62 and 4*U<2**62,'V2 signed reconstruction overflow certificate'
  self.bias=2*int(U);self.cap=int(U);self.parts=[]
  def make_pool_space(needed):
   while store.ram+needed>store.ram_limit:
    choices=[x for x in store.live.values() if x.array is not None and x.dtype.kind=='u' and not x.pins]
    if not choices:raise MemoryError('Protected compressed sections exceed the guarded RAM pool')
    max(choices,key=lambda x:x.nbytes).spill()
  def alloc(shape,dt,data=None):
   charge=math.prod(shape)*np.dtype(dt).itemsize;make_pool_space(charge)
   if data is None:part=store.create(shape,dt)
   else:
    assert data.shape==shape
    if data.dtype!=dt:
     assert int(data.max(initial=0))<=np.iinfo(dt).max
     guard.array(shape,dt,'V2 source adoption cast');data=data.astype(dt)
    part=object.__new__(Block)
    part.store=store;part.shape=shape;part.dtype=np.dtype(dt);part.size=math.prod(shape);part.nbytes=charge;part.charge=charge
    part.closed=False;part.array=data;part.file=None;part.path=None;part.pins=0
    store.live[id(part)]=part;store.ram+=charge;store.stats['peak_ram_blocks_bytes']=max(store.stats['peak_ram_blocks_bytes'],store.ram)
   self.parts.append(part)
   if part.array is None:raise MemoryError('Protected compressed sections must fit in guarded RAM')
   part.pins+=1 # Immutable sections are protected until the owning V2 block closes.
   return part
  K,sip,siq,P,Q,fp,fq,MP,MQ=S.data(n,key,2)
  guard.check((len(sip)*len(Q)+len(P)*len(siq))*K.dtype.itemsize,'V2 sampled analytic source')
  SR=S.row_source(K,sip,P,Q,*fp,MP,2)
  SC=S.reflected_column(SR,K[1],sip) if a==b else S.col_source(K,siq,P,Q,*fq,MQ,2)
  assert np.array_equal(sip[anchorsp],np.arange(len(anchorsp))) and np.array_equal(siq[anchorsq],np.arange(len(anchorsq)))
  guard.check((len(anchorsp)*len(Q)+len(repsp)*len(Q)+len(P)*len(repsq))*K.dtype.itemsize,'V2 retained small source cross sections')
  # Integer advanced indexing already returns independent data. A second
  # .copy() doubles the last cross-section allocation beyond the checked sum.
  SRap=SR[anchorsp,:];SRbp=SR[repsp,:];SCbq=SC[:,repsq]
  assert not np.shares_memory(SRap,SR) and not np.shares_memory(SRbp,SR) and not np.shares_memory(SCbq,SC)
  self.R=alloc((self.shape[0],len(anchorsq)),self.dtype,SR);SR=self.R.array
  self.C=alloc((len(anchorsp),self.shape[1]),self.dtype,SC);SC=self.C.array
  self.H=alloc((len(repsp),len(repsq)),E.dtype_for(4*U,'mixed'))
  Up,Uq=E.mt.B(a-1,h-1),E.mt.B(b-1,h-1)
  def parent(aa,bb,hh,rows,cols):
   ck=(min(aa,bb),max(aa,bb),hh)
   if ck not in maxima:return None
   assert ck in cur,('V2 predecessor released',key,ck)
   A=cur[ck]
   return indexed(A,cols,rows).T if aa>bb else indexed(A,rows,cols)
  def fill(part,rows,cols,kind):
   if not len(rows) or not len(cols):return
   step=max(1,store.chunk//max(1,len(cols)*16))
   for z in range(0,len(rows),step):
    ir=rows[z:z+step];guard.array((len(ir),len(cols)),np.int64,'V2 sample accumulation')
    if kind=='R':values=source_values(SR,SC,sip,siq,ir,cols)
    elif kind=='C':values=source_values(SRap,SC,np.arange(len(anchorsp),dtype=np.int32),siq,sip[ir].astype(np.int64),cols)
    else:values=source_values(SRbp,SCbq,sip[repsp],siq[repsq],np.arange(z,z+len(ir),dtype=np.int64),np.arange(len(cols),dtype=np.int64))
    ui=np.nonzero(ir<Up)[0];uj=np.nonzero(cols<Uq)[0]
    if len(ui) and len(uj):
     v=parent(a-1,b-1,h-1,ir[ui],cols[uj])
     if v is not None:assert not np.any(add_sample(values,v,ui,uj,self.cap)),'sample sum exceeds certified cap'
    di=np.nonzero(ir>=Up)[0];dj=np.nonzero(cols>=Uq)[0]
    if len(di):
     v=parent(a-1,b+1,h+1,ir[di]-Up,cols)
     if v is not None:assert not np.any(add_sample(values,v,di,np.arange(len(cols),dtype=np.int64),self.cap)),'sample sum exceeds certified cap'
    if len(dj):
     v=parent(a+1,b-1,h+1,ir,cols[dj]-Uq)
     if v is not None:assert not np.any(add_sample(values,v,np.arange(len(ir),dtype=np.int64),dj,self.cap)),'sample sum exceeds certified cap'
    assert int(values.min(initial=0))>=0 and int(values.max(initial=0))<=U
    part.array[z:z+len(ir),:]=values
  with store.borrowed(*self.parts):
   fill(self.R,np.arange(self.shape[0],dtype=np.int64),anchorsq,'R')
   fill(self.C,anchorsp,np.arange(self.shape[1],dtype=np.int64),'C')
   fill(self.H,repsp,repsq,'H')
   # A00[g,q] is already R[anchorsp[g],q]; no duplicate grid is needed.
   errors=encode_cross(self.H.array,self.R.array,self.C.array,anchorsp,ip,iq,repsp,repsq,self.bias)
   assert not np.any(errors),'negative interaction encoding'
   # U already uses the conjectural extremal maximum before allocation.
   # The former min(U, coarse_interval_bound) did not certify that conjecture.
   # Keep the same conditional cap without rescanning large sample matrices.
   self.bound=int(U)
  self.charge=sum(p.charge for p in self.parts);self.array=None;self.file=None;self.file_bytes=0
  STATS['blocks']+=1;STATS['logical_bytes']+=self.nbytes;STATS['sample_bytes']+=self.charge;STATS['sample_seconds']+=time.perf_counter()-start
 def values(self,rows,cols,cache_key=None):
  global GROUP_CACHE_BYTES
  start=time.perf_counter();guard.array((len(rows),len(cols)),self.dtype,'V2 decoded coefficient window')
  if any(p.array is None for p in self.parts):raise MemoryError('Experimental compressed sections spilled; file decoder pending')
  ip,wp,rp_,dp,op,repsp,anchorsp=self.p;iq,wq,rq,dq,oq,repsq,anchorsq=self.q
  grouping=time.perf_counter();guard.check(len(cols)*32+len(rows)*(self.q[1].shape[1]*8+4),'V2 grouped decoder workspace')
  if cache_key is not None and self.group_cache is not None and self.group_cache[0]==cache_key:order,ends=self.group_cache[1:];STATS['group_cache_hits']+=1
  else:
   order=np.argsort(iq[cols],kind='stable');groups=iq[cols[order]]
   ends=np.concatenate((np.array([0],np.int64),np.flatnonzero(groups[1:]!=groups[:-1])+1,np.array([len(cols)],np.int64))) if len(cols) else np.array([0],np.int64)
   if cache_key is not None:
    if self.group_cache is not None:GROUP_CACHE_BYTES-=self.group_cache[1].nbytes+self.group_cache[2].nbytes;self.group_cache=None
    amount=order.nbytes+ends.nbytes
    if GROUP_CACHE_BYTES+amount<=GROUP_CACHE_LIMIT:self.group_cache=(cache_key,order,ends);GROUP_CACHE_BYTES+=amount
  grouped=len(cols)>=64 and len(rows)*len(cols)>=10000 and len(ends)-1<.7*len(cols)
  STATS['column_group_seconds']+=time.perf_counter()-grouping
  with self.store.borrowed(*self.parts):
   if grouped:
    result,errors=decode_grouped(self.R.array,self.C.array,self.H.array,anchorsp,ip,wp,rp_,dp,op,iq,wq,rq,dq,oq,rows,cols,order,ends,self.bias,self.cap,np.empty(0,self.dtype));STATS['grouped_reads']+=1
   else:result,errors=decode(self.R.array,self.C.array,self.H.array,anchorsp,ip,wp,rp_,dp,op,iq,wq,rq,dq,oq,rows,cols,self.bias,self.cap,np.empty(0,self.dtype))
  assert not np.any(errors),'noninteger, negative or out-of-bound V2 reconstruction'
  STATS['read_seconds']+=time.perf_counter()-start;STATS['decoded_cells']+=result.size;return result
 def read(self,r0,r1,c0,c1):
  assert not self.closed and 0<=r0<=r1<=self.shape[0] and 0<=c0<=c1<=self.shape[1]
  return self.values(np.arange(r0,r1,dtype=np.int64),np.arange(c0,c1,dtype=np.int64),(c0,c1))
 def view(self,rect=None,flip=False):return View(self,rect,flip)
 def maximum_reduce(self,mod=None):
  assert mod is None
  return self.bound
 def close(self):
  global GROUP_CACHE_BYTES
  assert not self.closed and not self.pins
  if self.group_cache is not None:GROUP_CACHE_BYTES-=self.group_cache[1].nbytes+self.group_cache[2].nbytes;self.group_cache=None
  for p in self.parts:
   assert p.pins==1;p.pins-=1;p.close()
  self.closed=True

clear()
