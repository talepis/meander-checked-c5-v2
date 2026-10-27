"""Experimental exact-integer V2 interpolation. Bilinear reconstruction uses the proved E2 identity.

Only anchor cross sections and a feature-basis cross section are propagated.
Storage predictions use the unproved extremal recurrence. Rejection precedes unsafe use; the driver can retry with the proved coarse cap.
This first prototype intentionally stops if a compressed section spills to file.
"""
import gc,hashlib,inspect,json,math,time,weakref,os
import psutil
from checked_caps import PredictionRejected,require_sample,require_decoded
from pathlib import Path
from functools import lru_cache
import numpy as np
import numba as nb
import engine_v2 as E
import allocation_guard as guard
import recursive_profiles as rp
import source_separation as S
import incoming_e1 as I
from bounded_storage import View,Block
import file_sample_batch as FB
import analytic_mixed_v2 as BASE_CV
import analytic_mixed_kernels as AM
from experiments import rational_basis,basis_pivots
BASE=Path(__file__).resolve().parents[2];CACHE=BASE/'work/global_quotient/basis_cache'
CACHE_SIGNATURE=hashlib.sha256((inspect.getsource(rational_basis)+inspect.getsource(basis_pivots.py_func)+'e8083a7282e73c119d6cdd1bfdc890c2e2baf61186d35c570196de3771ea41d5').encode()).hexdigest()

STATS={}
OWNERS={}
PRESSURED_STORES=set()
FORCE_EXPANSION=False
RUN_ORDER=None
GROUP_CACHE_BYTES=0
GROUP_CACHE_LIMIT=32<<20
def clear():
 global GROUP_CACHE_BYTES
 assert GROUP_CACHE_BYTES==0,'live grouped-read caches at study boundary'
 prefix.cache_clear();feature_deltas.cache_clear();OWNERS.clear();PRESSURED_STORES.clear();STATS.clear();STATS.update(blocks=0,logical_bytes=0,sample_bytes=0,prepare_seconds=0.,sample_seconds=0.,read_seconds=0.,decoded_cells=0,basis_cache_builds=0,basis_cache_loads=0,grouped_reads=0,column_group_seconds=0.,group_cache_hits=0)

def release(store=None):
 stores=list(OWNERS) if store is None else [id(store)]
 released=0
 for sid in stores:
  PRESSURED_STORES.add(sid)
  for owner in list(OWNERS.get(sid,())):
   if owner.closed or not owner.diagonal or owner.symmetric:continue
   part=owner.C
   if part.array is None or part.pins!=1 or owner.R.array is None:continue
   amount=part.charge
   owner.parts.remove(part);part.pins-=1;part.close()
   owner.C=TransposedSection(owner.R);owner.symmetric=True
   owner.charge=sum(x.charge for x in owner.parts)
   released+=amount
 STATS['released_duplicate_C_bytes']=STATS.get('released_duplicate_C_bytes',0)+released
 return released

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
 # The analytic kernel needs only exact profile IDs and real anchor indices.
 # No interpolation coordinates, rational basis, or sampled H representatives.
 start=time.perf_counter();ids,P=E.ms.profiles(a,h)
 N,G=len(ids),len(P)
 guard.check((N+G)*24,'analytic mixed exact profile anchors')
 anchors=np.full(G,N,np.int64)
 np.minimum.at(anchors,ids,np.arange(N,dtype=np.int64))
 assert np.all(anchors<N) and np.array_equal(ids[anchors],np.arange(G))
 empty=np.empty(0,np.int64)
 result=(ids,np.empty((0,0),np.int64),np.zeros(G,np.int32),np.ones(G,np.int64),np.zeros(G,np.int64),empty,anchors)
 STATS['prepare_seconds']+=time.perf_counter()-start
 return result

@lru_cache(None)
def feature_deltas(a,h):
 ids,wp,rp_,dp,op,reps,anchors=prefix(a,h)
 SS,off,lev,val=rp.features(a,h)
 guard.check(len(ids)*(h+2)*24+(len(ids)+1)*8,'exact sparse feature differences')
 X=make_features(SS,off,lev,val,h)
 assert int(np.abs(X).max(initial=0))<(1<<31)
 return AM.deltas(X,ids,anchors)

def expanded_selected(a,b,h,n=None):
 N,M=E.mt.B(a,h),E.mt.B(b,h)
 if N*M<100000:return False
 gp=profile_counts(a,h)[0];gq=profile_counts(b,h)[0]
 cells=N*gp if a==b else N*gq+gp*M
 dense=N*M
 nn=RUN_ORDER if n is None else n
 if nn is not None:
  from packed_boundary import should_pack
  t=a+b-h+4
  if should_pack(nn,t,(a,b,h)):dense-=E.mt.B(a-1,h-1)**2
 return 20*cells<13*dense

def selected(a,b,h):
 old=BASE_CV.selected(a,b,h)
 if old:return True
 if not (FORCE_EXPANSION or PRESSURED_STORES):return False
 result=expanded_selected(a,b,h)
 if result:
  t=a+b-h+4
  STATS['first_extra_compact_stage']=min(STATS.get('first_extra_compact_stage',t),t)
 return result

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

@nb.njit(cache=True,parallel=True)
def symmetric_source_values(L,R,K1,ip,iq,rows,cols):
 out=np.empty((len(rows),len(cols)),np.int64)
 for ii in nb.prange(len(rows)):
  i=rows[ii];gp=ip[i]
  for jj in range(len(cols)):
   j=cols[jj];gq=iq[j]
   out[ii,jj]=np.int64(L[i,gq])+np.int64(R[j,gp])-np.int64(K1[gp,gq])
 return out

class TransposedSection:
 def __init__(self, owner):self.owner=owner
 @property
 def array(self):return self.owner.array.T

class Compact:
 def __init__(self,n,key,U,cur,maxima,store):
  start=time.perf_counter();self.store=store;self.closed=False;self.pins=0;self.group_cache=None
  a,b,h=key;self.key=key;self.diagonal=a==b;self.shape=(E.mt.B(a,h),E.mt.B(b,h));self.size=math.prod(self.shape);self.dtype=np.dtype(E.dtype_for(U,'mixed'));self.nbytes=self.size*self.dtype.itemsize
  self.p=prefix(a,h);self.q=prefix(b,h);ip,wp,rp_,dp,op,repsp,anchorsp=self.p;iq,wq,rq,dq,oq,repsq,anchorsq=self.q
  assert int(dp.max(initial=1))*int(dq.max(initial=1))<2**62,'V2 denominator product overflow'
  assert 4*U<2**62,'V2 source signed reconstruction overflow certificate'
  self.bias=2*int(U);self.cap=int(U);self.parts=[]
  def make_pool_space(needed):
   if store.ram+needed>store.ram_limit:release(store)
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
     if int(data.max(initial=0))>np.iinfo(dt).max:raise PredictionRejected('V2 source does not fit predicted storage width')
     guard.array(shape,dt,'V2 source adoption cast');data=data.astype(dt)
    part=object.__new__(Block)
    part.store=store;part.shape=shape;part.dtype=np.dtype(dt);part.size=math.prod(shape);part.nbytes=charge;part.charge=charge
    part.closed=False;part.array=data;part.file=None;part.path=None;part.pins=0
    store.live[id(part)]=part;store.ram+=charge;store.stats['peak_ram_blocks_bytes']=max(store.stats['peak_ram_blocks_bytes'],store.ram)
   self.parts.append(part)
   if part.array is None:raise MemoryError('Protected compressed sections must fit in guarded RAM')
   part.pins+=1 # Immutable sections are protected until the owning V2 block closes.
   return part
  # Proved independently of every predicted V2 cap. This check is O(1) per block.
  if 2*int(I.source_certificate(key,2))>=1<<61:raise ArithmeticError('V2 source signed-range certificate unavailable')
  K,sip,siq,P,Q,fp,fq,MP,MQ=S.data(n,key,2)
  guard.check((len(sip)*len(Q)+(0 if self.diagonal else len(P)*len(siq)))*K.dtype.itemsize,'V2 sampled analytic source')
  SR=S.row_source(K,sip,P,Q,*fp,MP,2)
  SC=None if self.diagonal else S.col_source(K,siq,P,Q,*fq,MQ,2)
  assert np.array_equal(sip[anchorsp],np.arange(len(anchorsp))) and np.array_equal(siq[anchorsq],np.arange(len(anchorsq)))
  guard.check((len(anchorsp)*len(Q))*K.dtype.itemsize,'V2 retained small source cross sections')
  # Integer advanced indexing already returns independent data. A second
  # .copy() doubles the last cross-section allocation beyond the checked sum.
  SRap=SR[anchorsp,:];SRbp=SCbq=None
  assert not np.shares_memory(SRap,SR)
  projected=(self.shape[0]*len(anchorsq)+len(anchorsp)*self.shape[1])*self.dtype.itemsize
  self.symmetric=(a==b and (20*projected>=13*self.nbytes or id(store) in PRESSURED_STORES or store.ram+projected>store.ram_limit))
  self.R=alloc((self.shape[0],len(anchorsq)),self.dtype,SR);SR=self.R.array
  self.C=TransposedSection(self.R) if self.diagonal else alloc((len(anchorsp),self.shape[1]),self.dtype,SC)
  if not self.diagonal:SC=self.C.array
  self.H=None
  self.profile_P,self.profile_Q=P,Q;self.choose=E.old.choose(n)
  self.delta_p,self.delta_q=feature_deltas(a,h),feature_deltas(b,h)
  p0,q0=(a-h)//2,(b-h)//2;Kmax=math.comb(p0+q0,p0);Tp=p0*(p0+3);Tq=q0*(q0+3)
  Amax=Kmax*(Tp+2*p0*q0);Nmax=(Amax+Kmax*p0*q0)*p0;Bmax=Nmax+Kmax*p0
  Mmax=Amax*Tq+2*q0*Bmax
  certificate=max(Amax,Nmax,Bmax,3*int(U)+2*Mmax)
  assert certificate<2**62,'analytic mixed contraction signed overflow certificate'
  STATS['maximum_mixed_intermediate_certificate']=max(STATS.get('maximum_mixed_intermediate_certificate',0),certificate)
  Up,Uq=E.mt.B(a-1,h-1),E.mt.B(b-1,h-1)
  pending={}
  def parent(aa,bb,hh,rows,cols):
   ck=(min(aa,bb),max(aa,bb),hh)
   if ck not in maxima:return None
   assert ck in cur,('V2 predecessor released',key,ck)
   A=cur[ck]
   if isinstance(A,Block) and A.array is None and A.file is not None and FB.eligible(self,A,(aa,bb,hh)):
    pending.setdefault(ck,set()).add((aa,bb,hh));return None
   return indexed(A,cols,rows).T if aa>bb else indexed(A,rows,cols)
  def fill(part,rows,cols,kind):
   if not len(rows) or not len(cols):return
   step=max(1,store.chunk//max(1,len(cols)*16))
   for z in range(0,len(rows),step):
    ir=rows[z:z+step];guard.array((len(ir),len(cols)),np.int64,'V2 sample accumulation')
    if kind=='R':values=symmetric_source_values(SR,SRap,K[1],sip,np.arange(len(anchorsq),dtype=np.int32),ir,np.arange(len(anchorsq),dtype=np.int64)) if self.diagonal else source_values(SR,SC,sip,siq,ir,cols)
    elif kind=='C':values=source_values(SRap,SC,np.arange(len(anchorsp),dtype=np.int32),siq,sip[ir].astype(np.int64),cols)
    else:values=symmetric_source_values(SRbp,SRbp,K[1],sip[repsp],sip[repsq],np.arange(z,z+len(ir),dtype=np.int64),np.arange(len(cols),dtype=np.int64)) if self.diagonal else source_values(SRbp,SCbq,sip[repsp],siq[repsq],np.arange(z,z+len(ir),dtype=np.int64),np.arange(len(cols),dtype=np.int64))
    ui=np.nonzero(ir<Up)[0];uj=np.nonzero(cols<Uq)[0]
    if len(ui) and len(uj):
     v=parent(a-1,b-1,h-1,ir[ui],cols[uj])
     if v is not None:require_sample(add_sample(values,v,ui,uj,self.cap))
    di=np.nonzero(ir>=Up)[0];dj=np.nonzero(cols>=Uq)[0]
    if len(di):
     v=parent(a-1,b+1,h+1,ir[di]-Up,cols)
     if v is not None:require_sample(add_sample(values,v,di,np.arange(len(cols),dtype=np.int64),self.cap))
    if len(dj):
     v=parent(a+1,b-1,h+1,ir,cols[dj]-Uq)
     if v is not None:require_sample(add_sample(values,v,np.arange(len(ir),dtype=np.int64),dj,self.cap))
    if int(values.min(initial=0))<0 or int(values.max(initial=0))>U:raise PredictionRejected('V2 source sample does not fit predicted cap')
    part.array[z:z+len(ir),:]=values
  with store.borrowed(*self.parts):
   fill(self.R,np.arange(self.shape[0],dtype=np.int64),anchorsq,'R')
   if not self.diagonal:fill(self.C,anchorsp,np.arange(self.shape[1],dtype=np.int64),'C')
   # The proved mixed kernel replaces the entire sampled H section.
   for ck,requests in sorted(pending.items()):
    with store.borrowed(cur[ck]):FB.apply(self,cur[ck],requests,STATS)
   if self.diagonal and not self.symmetric:
    needed=len(anchorsp)*self.shape[1]*self.dtype.itemsize
    budget=int(os.environ.get('MEANDER_BUDGET_BYTES','0'))
    m=psutil.Process().memory_info();available=psutil.virtual_memory().available
    reserve=int(os.environ.get('MEANDER_RESERVE_BYTES','0'))
    pressure=id(store) in PRESSURED_STORES or needed>store.small_limit or store.ram+needed>store.ram_limit
    if budget:pressure=pressure or max(m.rss,m.vms)+needed+(128<<20)>budget or available<reserve+needed+(128<<20)
    if pressure:release(store);self.symmetric=True
   if self.diagonal and not self.symmetric:
    self.C=alloc((len(anchorsp),self.shape[1]),self.dtype)
    self.C.array[:]=self.R.array.T
   # A00[g,q] is already R[anchorsp[g],q]; no duplicate grid is needed.
   STATS['omitted_H_sample_bytes']=STATS.get('omitted_H_sample_bytes',0)+profile_counts(a,h)[1]*profile_counts(b,h)[1]*np.dtype(E.dtype_for(4*U,'mixed')).itemsize
   # U already uses the conjectural extremal maximum before allocation.
   # The former min(U, coarse_interval_bound) did not certify that conjecture.
   # A prediction is enforced on every required access; it need not be a global maximum.
   self.bound=int(U)
  self.charge=sum(p.charge for p in self.parts);self.array=None;self.file=None;self.file_bytes=0
  STATS['symmetric_sections_saved_bytes']=STATS.get('symmetric_sections_saved_bytes',0)+(self.R.nbytes if self.symmetric else 0)
  STATS['extra_compact_blocks']=STATS.get('extra_compact_blocks',0)+(0 if BASE_CV.selected(*key) else 1)
  OWNERS.setdefault(id(store),weakref.WeakSet()).add(self)
  STATS['blocks']+=1;STATS['logical_bytes']+=self.nbytes;STATS['sample_bytes']+=self.charge;STATS['sample_seconds']+=time.perf_counter()-start
 def values(self,rows,cols,cache_key=None):
  global GROUP_CACHE_BYTES
  start=time.perf_counter();guard.array((len(rows),len(cols)),self.dtype,'analytic mixed coefficient window')
  if any(p.array is None for p in self.parts):raise MemoryError('Compressed sections spilled')
  ip,wp,rp_,dp,op,repsp,anchorsp=self.p;iq,wq,rq,dq,oq,repsq,anchorsq=self.q
  grouping=time.perf_counter();guard.check(len(cols)*32+len(rows)*((self.profile_P.shape[1]+1)*16+4),'analytic mixed decoder workspace')
  if cache_key is not None and self.group_cache is not None and self.group_cache[0]==cache_key:
   order,ends=self.group_cache[1:];STATS['group_cache_hits']+=1
  else:
   order=np.argsort(iq[cols],kind='stable');groups=iq[cols[order]]
   ends=np.concatenate((np.array([0],np.int64),np.flatnonzero(groups[1:]!=groups[:-1])+1,np.array([len(cols)],np.int64))) if len(cols) else np.array([0],np.int64)
   if cache_key is not None:
    if self.group_cache is not None:GROUP_CACHE_BYTES-=self.group_cache[1].nbytes+self.group_cache[2].nbytes;self.group_cache=None
    amount=order.nbytes+ends.nbytes
    if GROUP_CACHE_BYTES+amount<=GROUP_CACHE_LIMIT:self.group_cache=(cache_key,order,ends);GROUP_CACHE_BYTES+=amount
  STATS['column_group_seconds']+=time.perf_counter()-grouping
  with self.store.borrowed(*self.parts):
   result,errors=AM.decode(self.R.array,self.C.array,anchorsp,anchorsq,ip,iq,self.delta_p[0],self.delta_q[1],self.delta_q[2],self.delta_q[3],self.profile_P,self.profile_Q,self.choose,rows,cols,order,ends,self.cap,np.empty(0,self.dtype))
  require_decoded(errors)
  STATS['read_seconds']+=time.perf_counter()-start;STATS['decoded_cells']+=result.size;STATS['grouped_reads']+=1
  return result
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
