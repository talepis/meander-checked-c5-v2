"""Mathematical probes; no observations are inserted as enumeration constants."""
import gc,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import numba as nb
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/MATH_RESEARCH'
P=1000003
@nb.njit(cache=True)
def rank_mod(A,p=P):
 A=A.copy();r=0
 for c in range(A.shape[1]):
  pivot=-1
  for i in range(r,A.shape[0]):
   if A[i,c]%p:pivot=i;break
  if pivot<0:continue
  for j in range(c,A.shape[1]):tmp=A[r,j];A[r,j]=A[pivot,j];A[pivot,j]=tmp
  # Fermat inverse, with products below p squared < 2^40.
  inv=1;x=A[r,c]%p;e=p-2
  while e:
   if e&1:inv=inv*x%p
   x=x*x%p;e//=2
  for j in range(c,A.shape[1]):A[r,j]=A[r,j]*inv%p
  for i in range(r+1,A.shape[0]):
   f=A[i,c]%p
   if f:
    for j in range(c,A.shape[1]):A[i,j]=(A[i,j]-f*A[r,j])%p
  r+=1
  if r==A.shape[0]:break
 return r
def configure(n):
 import variants_next as V,configure_bugfirst as B
 assert V.configure('projected-family-sym')==2;B.load_maps(n);V.clear();gc.collect()
 return B.engine,V
def features(a,h):
 import engine_v2 as E
 rows=[];words=[]
 for r in range(E.mt.B(a,h)):
  w=''.join('(' if b else ')' for b in E.mt.unrank_bits(a,h,r));words.append(w)
  forest=[[]];sigma=0
  for c in w:
   if c=='(':forest.append([])
   else:
    kids=forest.pop();d=len(kids);sigma+=(d+1)**2;forest[-1].append(d)
  u=tuple(map(len,forest));ss=sigma-sum(u);rd=tuple(map(sum,forest))
  # Global s and roots by level are the shape-dependent part of V1.
  rows.append((u,(ss,*rd),tuple(tuple(f) for f in forest)))
 return rows,words
def structure():
 E,V=configure(24);mat={}
 chosen={(14,14,0),(14,14,2),(14,14,4),(16,16,4),(12,14,2)}
 def observer(t,target,maxima,U):
  for k,A in target.items():
   if k in chosen and t-k[0]-k[1]+k[2]==4:mat[k]=A.copy().astype(np.int64)
 E.LAYER_OBSERVER=observer;r=E.meander(24,strip=2);E.LAYER_OBSERVER=None
 assert str(r['answer'])=='794337831754570367812'
 details=[];counterexamples=[];checks=0
 for (a,b,h),A in sorted(mat.items()):
  fp,wp=features(a,h);fq,wq=features(b,h)
  gp=defaultdict(list);gq=defaultdict(list)
  for i,f in enumerate(fp):gp[f[0]].append(i)
  for j,f in enumerate(fq):gq[f[0]].append(j)
  Xall=np.array([f[1] for f in fp],np.int64);Yall=np.array([f[1] for f in fq],np.int64)
  tested=failed=0;maxrank=0;max_feature_rank=0;unique_rows=0;worst=None
  for up,ii in gp.items():
   if len(ii)<3:continue
   for uq,jj in gq.items():
    if len(jj)<3:continue
    # Bound probe cost while visiting all fixed-profile groups of this size.
    if len(ii)>160 or len(jj)>160:continue
    D=A[np.ix_(ii,jj)];D=(D-D[:,0,None]-D[0,None,:]+D[0,0])%P
    X=(Xall[ii]-Xall[ii[0]])%P;Y=(Yall[jj]-Yall[jj[0]])%P
    rx=rank_mod(X);ry=rank_mod(Y);rd=rank_mod(D)
    left=rank_mod(np.concatenate((X,D),axis=1));right=rank_mod(np.concatenate((Y,D.T),axis=1))
    tested+=1;checks+=D.size;maxrank=max(maxrank,rd);max_feature_rank=max(max_feature_rank,rx,ry)
    if left!=rx or right!=ry:
     failed+=1
     if worst is None:worst=dict(profiles=[up,uq],dimensions=D.shape,interaction_rank=rd,feature_ranks=[rx,ry],augmented_ranks=[left,right],P_words=[wp[i] for i in ii[:6]],Q_words=[wq[j] for j in jj[:6]])
  # Stronger source-signature quotient is checked independently on all rows.
  sig={};collision=None
  for i,f in enumerate(fp):
   s=(f[0],f[1][0],f[2])
   if s in sig and not np.array_equal(A[i],A[sig[s]]):
    j=int(np.nonzero(A[i]!=A[sig[s]])[0][0]);collision=dict(P=[wp[sig[s]],wp[i]],Q=wq[j],values=[int(A[sig[s],j]),int(A[i,j])]);break
   sig[s]=i
  item=dict(key=[a,b,h],shape=A.shape,profile_counts=[len(gp),len(gq)],profile_pairs_tested=tested,bilinear_failures=failed,max_interaction_rank=maxrank,max_feature_rank=max_feature_rank,first_failure=worst,full_V2_signature_counterexample=collision)
  details.append(item);print(json.dumps(item),flush=True)
 return dict(status='PROBED',cycle=1,hypothesis='At fixed root-count profiles, mixed V2 dependence lies in the span of the V1 shape features on both sides.',modulus=P,probe_coefficients=checks,details=details,whole_M24_matches=True)
@nb.njit(cache=True)
def basis_pivots(X,p=P):
 A=X.copy()%p;order=np.arange(len(X));reps=np.empty(min(X.shape),np.int64);cols=np.empty(min(X.shape),np.int64);r=0
 for c in range(A.shape[1]):
  pivot=-1
  for i in range(r,len(A)):
   if A[i,c]:pivot=i;break
  if pivot<0:continue
  for j in range(A.shape[1]):tmp=A[r,j];A[r,j]=A[pivot,j];A[pivot,j]=tmp
  tmp=order[r];order[r]=order[pivot];order[pivot]=tmp
  reps[r]=order[r];cols[r]=c
  inv=1;x=A[r,c];e=p-2
  while e:
   if e&1:inv=inv*x%p
   x=x*x%p;e//=2
  for j in range(c,A.shape[1]):A[r,j]=A[r,j]*inv%p
  for i in range(r+1,len(A)):
   f=A[i,c]
   if f:
    for j in range(c,A.shape[1]):A[i,j]=(A[i,j]-f*A[r,j])%p
  r+=1
  if r==len(A):break
 return reps[:r],cols[:r]
def rational_basis(X):
 from fractions import Fraction as F
 # A nonzero modular minor selects rationally independent rows. The following
 # exact rational inverse and full integer reconstruction certify the basis.
 reps,cols=basis_pivots(X)
 if not len(reps):
  assert not np.any(X)
  return np.zeros((len(X),0),np.int64),1,[]
 r=len(reps);M=[[F(int(X[i,j])) for j in cols]+[F(k==j) for j in range(r)] for k,i in enumerate(reps)]
 for c in range(r):
  z=next(i for i in range(c,r) if M[i][c]);M[c],M[z]=M[z],M[c]
  d=M[c][c];M[c]=[x/d for x in M[c]]
  for i in range(r):
   if i!=c:
    f=M[i][c];M[i]=[x-f*y for x,y in zip(M[i],M[c])]
 inv=[row[r:] for row in M]
 den=math.lcm(*(v.denominator for row in inv for v in row))
 inverse_integer=np.array([[int(v*den) for v in row] for row in inv],np.int64)
 assert int(np.abs(X[:,cols]).max(initial=0))*int(np.abs(inverse_integer).max(initial=0))*r<2**62
 Wi=X[:,cols]@inverse_integer
 common=math.gcd(den,int(np.gcd.reduce(np.abs(Wi).ravel(),initial=0)))
 if common>1:Wi//=common;den//=common
 assert np.array_equal(Wi@X[reps],X*den)
 return Wi,den,reps
def exact_structure():
 E,V=configure(26);mat={};chosen={(14,14,0),(14,14,2),(16,16,4),(18,18,6)}
 def observer(t,target,maxima,U):
  for k,A in target.items():
   if k in chosen and t-k[0]-k[1]+k[2]==4:mat[k]=A.copy().astype(np.int64)
 E.LAYER_OBSERVER=observer;r=E.meander(26,strip=2);E.LAYER_OBSERVER=None
 assert str(r['answer'])=='91412898898828176826244'
 details=[];checks=0
 for (a,b,h),A in sorted(mat.items()):
  fp,wp=features(a,h);groups=defaultdict(list)
  for i,f in enumerate(fp):groups[f[0]].append(i)
  Xall=np.array([f[1] for f in fp],np.int64);bases=[];anchors=[];allsamples=[]
  for u,ii in groups.items():
   W,den,rr=rational_basis(Xall[ii]-Xall[ii[0]])
   reps=[ii[i] for i in rr];bases.append((ii,W,den,reps));anchors.append(ii[0]);allsamples.extend(reps)
  ng=len(bases);rs=len(allsamples);fail=None;cells=0;maxden=1
  for u,(ii,Wi,di,ri) in enumerate(bases):
   for v,(jj,Wj,dj,rj) in enumerate(bases):
    D=A[np.ix_(ii,jj)]-A[ii,jj[0],None]-A[ii[0],jj][None,:]+A[ii[0],jj[0]]
    if len(ri) and len(rj):
     S=A[np.ix_(ri,rj)]-A[ri,jj[0],None]-A[ii[0],rj][None,:]+A[ii[0],jj[0]]
     bound=int(np.abs(Wi).max(initial=0))*int(np.abs(Wj).max(initial=0))*len(ri)*len(rj)*int(np.abs(S).max(initial=0))
     assert bound<2**63,'integer reconstruction product exceeds certificate'
     N=Wi@S@Wj.T;maxden=max(maxden,di*dj)
     ok=np.array_equal(N,D*(di*dj))
    else:ok=not np.any(D)
    cells+=D.size
    if not ok and fail is None:fail=dict(profiles=[u,v],dimensions=D.shape,words_P=[wp[i] for i in ii[:4]],words_Q=[wp[j] for j in jj[:4]])
   if u%100==0:print(json.dumps(dict(phase='integer_reconstruction',key=[a,b,h],profile=u,total_profiles=ng,checked_cells=cells)),flush=True)
  # Exact container estimate uses two anchor cross sections and basis cross section.
  sample_cells=2*len(A)*ng-ng*ng+rs*rs
  item=dict(key=[a,b,h],shape=A.shape,coefficients_checked=cells,profile_groups=ng,total_interaction_basis_rank=rs,largest_common_denominator=maxden,failure=fail,dense_cells=A.size,representation_sample_cells=sample_cells,representation_cell_ratio=sample_cells/A.size,weight_bytes=sum(W.nbytes for ii,W,d,rr in bases))
  details.append(item);checks+=cells;print(json.dumps(item),flush=True)
 return dict(status='PASS' if all(x['failure'] is None for x in details) else 'REFUTED',cycle=2,hypothesis='Exact rational bilinear V2 reconstruction at fixed root-count profiles',integer_coefficients_checked=checks,whole_M26_matches=True,details=details)
def run(experiment,tag,pool):
 if experiment=='structure':return structure()
 if experiment=='exact-structure':return exact_structure()
 if experiment=='extremal':return extremal()
 if experiment=='compact-tests':return compact_tests(tag,pool)
 if experiment=='benchmark-small':return benchmark(tag,pool,(25,27),3)
 if experiment=='benchmark-28':return benchmark(tag,pool,(28,),2)
 if experiment=='benchmark-30':return benchmark(tag,pool,(30,),1)
 if experiment=='storage-and-leaf':return storage_and_leaf(tag,pool)
 if experiment=='rank-integers':return rank_integers()
 if experiment=='interior-extremal':return interior_extremal()
 if experiment=='division-audit':return division_audit()
 if experiment=='source-reuse-tests':return source_reuse_tests()
 if experiment=='native-tests':return compact_tests(tag,pool,use_native=True)
 if experiment=='benchmark-native-small':return benchmark(tag,pool,(25,27),3,use_native=True)
 if experiment=='basis-preparation-30':return basis_preparation(tag)
 raise ValueError(experiment)

def extremal():
 from functools import lru_cache
 from probe_mixed import newsource,feat
 def choose(n,k):return math.comb(n,k) if 0<=k<=n else 0
 @lru_cache(None)
 def flat(p,q,h):
  if p==q==h==0:return 0
  W='()'*p+'('*h;Z='()'*q+'('*h
  source=newsource(W,Z,2)
  if h:
   u=feat(W+')')[0];v=feat(Z+')')[0]
   source+=math.prod(choose(x+y,x) for x,y in zip(u,v))
   return source+flat(p,q,h-1)
  return source+(flat(p-1,q,1) if p else 0)+(flat(p,q-1,1) if q else 0)
 x=json.loads((BASE/'outputs/M31/M31_EXACT.json').read_text());tested=0;fail=[];ratio=0.
 for L in x['layers']:
  for b in L['blocks']:
   a,c,h=b['key'];e=(L['t']-a-c+h)//2
   if e!=2:continue
   actual=int(b['max_decimal']);pred=flat((a-h)//2,(c-h)//2,h);tested+=1
   if actual!=pred:fail.append(dict(key=b['key'],exact_max=actual,flat_coefficient=pred,ratio=actual/max(1,pred)))
 return dict(status='PASS' if not fail else 'REFUTED',cycle=3,hypothesis='The V2 block maximum is attained by both forests consisting of leaves before all unmatched openings.',keys_checked=tested,failures=fail,scalar_states=flat.cache_info().currsize,uses_answers_as_input=False)

def compact_tests(tag,pool,use_native=False):
 import compact_exact as X,compact_v2 as CV
 if use_native:import reuse_exact as X
 E,V=configure(25);reference={};cells=0;comp=0;details=[]
 for n in (8,12,18,22,24,25):
  V.clear();CV.clear();gc.collect();reference={}
  def record(t,target,maxima,U):reference[t]={k:A.copy() for k,A in target.items()}
  E.LAYER_OBSERVER=record;old=E.meander(n,strip=2);E.LAYER_OBSERVER=None
  local=[0,0]
  def compare(t,target,maxima,U):
   rr=reference.pop(t);assert set(rr)==set(target),(n,t,'keys')
   for k,A in target.items():
    if isinstance(A,CV.Compact):assert maxima[k]==int(rr[k].max(initial=0)),(n,t,k,'extremal maximum conjecture')
    step=max(1,(16<<20)//max(1,A.shape[1]*A.dtype.itemsize))
    for i in range(0,A.shape[0],step):
     v=A.read(i,min(A.shape[0],i+step),0,A.shape[1])
     assert np.array_equal(v,rr[k][i:i+len(v)]),(n,t,k,'coefficient reconstruction')
     assert int(v.max(initial=0))<=maxima[k],(n,t,k,'maximum certificate')
     local[0]+=v.size
    if isinstance(A,CV.Compact):local[1]+=A.size
  V.clear();CV.clear();gc.collect()
  got=X.meander(n,BASE/'work/math_research/scratch'/f'{tag}_n{n}',ram_limit=pool,small_limit=pool,verbose=False,layer_check=compare)
  assert not reference and got['answer_decimal']==str(old['answer'])
  cells+=local[0];comp+=local[1]
  item=dict(n=n,answer_decimal=got['answer_decimal'],all_coefficients=local[0],compressed_coefficients=local[1],statistics=got.get('compact_statistics',{}),source_reuse_statistics=got.get('source_reuse_statistics'),seconds_including_comparisons=got['seconds'])
  details.append(item);print(json.dumps(item),flush=True)
 return dict(status='PASS',cycle=4,all_coefficients_checked=cells,compressed_coefficients_checked=comp,details=details,conjectural_V2_used=not use_native,conjectural_extremal_bounds_used=not use_native)

def benchmark(tag,pool,orders,repeats,use_native=False):
 import threading,psutil
 import bounded_exact as baseline,compact_exact as candidate,compact_v2 as CV
 if use_native:import reuse_exact as candidate
 E,V=configure(max(orders));warm={};details=[]
 for name,fn in (('baseline',baseline.meander),('candidate',candidate.meander)):
  V.clear();CV.clear();gc.collect();start=time.perf_counter()
  warm[name]=fn(25,BASE/'work/math_research/scratch'/f'{tag}_warm_{name}',ram_limit=pool,small_limit=pool,verbose=False)['seconds']
  print(json.dumps(dict(phase='warmed',version=name,seconds=warm[name])),flush=True)
 def known(n):
  # Known answers are assertions after the run, never input to a recurrence.
  source=BASE/f'outputs/MN_TABLE/RUNS/TABLE_CURRENT_M{n}.json'
  return json.loads(source.read_text())['answer_decimal']
 for n in orders:
  for rep in range(repeats):
   seq=(('baseline',baseline.meander),('candidate',candidate.meander)) if rep%2==0 else (('candidate',candidate.meander),('baseline',baseline.meander))
   for name,fn in seq:
    V.clear();CV.clear();gc.collect();stop=threading.Event();measure=[0,0]
    def sample():
     process=psutil.Process()
     while not stop.is_set():
      mem=process.memory_info();measure[0]=max(measure[0],mem.rss);measure[1]=max(measure[1],mem.vms);stop.wait(.02)
    th=threading.Thread(target=sample,daemon=True);th.start();start=time.perf_counter()
    def progress(t,target,maxima,U):
     if n>=28 and (t%2==0 or t>=2*n-4):
      (OUT/f'{tag}_LAYER_PROGRESS.json').write_text(json.dumps(dict(n=n,version=name,repetition=rep,completed_layer=t,total_layers=2*n-1,elapsed_seconds=time.perf_counter()-start,ram_coefficients_bytes=sum((A.charge if hasattr(A,'parts') else A.nbytes) for A in target.values() if A.array is not None or hasattr(A,'parts')))),encoding='utf-8')
      print(json.dumps(dict(n=n,version=name,layer=t,elapsed_seconds=time.perf_counter()-start)),flush=True)
    try:r=fn(n,BASE/'work/math_research/scratch'/f'{tag}_n{n}_{name}_{rep}',ram_limit=pool,small_limit=pool,verbose=False,layer_check=progress)
    finally:stop.set();th.join()
    total=time.perf_counter()-start;assert r['answer_decimal']==known(n)
    item=dict(n=n,version=name,repetition=rep,variant=r.get('variant'),answer_decimal=r['answer_decimal'],core_seconds=r['seconds'],total_measured_seconds=total,phase_seconds=r['phase_seconds'],peak_rss_bytes=measure[0],peak_commit_bytes=measure[1],peak_layer_bytes=max(L['array_bytes'] for L in r['layers']),peak_ram_coefficients_bytes=r['storage_stats']['peak_ram_blocks_bytes'],storage_stats=r['storage_stats'],compact_statistics=r.get('compact_statistics'),source_reuse_statistics=r.get('source_reuse_statistics'),conjectural_extremal_bounds_used=r.get('conjectural_extremal_bounds_used',False),all_final_answers_match=True)
    details.append(item)
    (OUT/f'{tag}_PROGRESS.json').write_text(json.dumps(dict(phase='partial',completed=details,warm=warm),indent=2),encoding='utf-8')
    print(json.dumps(item),flush=True)
 return dict(status='PASS',cycle=5,orders=orders,repeats=repeats,details=details,warm_up_seconds=warm,arithmetic='exact mixed-width integers',threads=6,conjectural_V2_used=not use_native,basis_preparation_included_in_core=True)

def storage_and_leaf(tag,pool):
 from bounded_storage import Store
 import compact_v2 as CV,compact_exact as X
 from functools import lru_cache
 from probe_mixed import newsource,feat
 tests=0;root=BASE/'work/math_research/scratch'
 for dt in (np.uint16,np.uint32,np.uint64):
  store=Store(root/f'{tag}_storage_{np.dtype(dt)}',ram_limit=64<<20,small_limit=64<<20)
  B=store.create((1027,1031),dt);values=np.full(B.shape,np.iinfo(dt).max,dt)-np.arange(B.size,dtype=dt).reshape(B.shape)
  B.write(0,1027,0,1031,values);B.spill()
  rows=np.array([1026,0,512,511,512,7],np.int64);cols=np.array([1030,513,0,511,512,513],np.int64)
  actual=CV.indexed(B,rows,cols);assert np.array_equal(actual,values[np.ix_(rows,cols)]);tests+=actual.size;B.close();store.finish()
 E,V=configure(22);rr={}
 def record(t,target,maxima,U):rr[t]={k:A.copy() for k,A in target.items()}
 E.LAYER_OBSERVER=record;old=E.meander(22,strip=2);E.LAYER_OBSERVER=None
 V.clear();CV.clear();gc.collect();counters=[0,0]
 def compare(t,target,maxima,U):
  ref=rr.pop(t)
  for k,A in target.items():assert np.array_equal(A.read(0,A.shape[0],0,A.shape[1]),ref[k]);counters[0]+=A.size
  for k,A in sorted(target.items(),key=lambda kv:kv[1].size,reverse=True):
   if not isinstance(A,CV.Compact) and A.array is not None and A.dtype.kind=='u' and A.size>5000 and t-sum(k[:2])+k[2]==4 and counters[1]<8:
    A.spill();counters[1]+=1;break
 got=X.meander(22,root/f'{tag}_migration',ram_limit=pool,small_limit=pool,verbose=False,layer_check=compare)
 assert not rr and got['answer_decimal']==str(old['answer']) and counters[1]>0
 def source(W,Z):
  s=newsource(W,Z,2)
  h=W.count('(')-W.count(')')
  if h:
   u=feat(W+')')[0];v=feat(Z+')')[0];s+=math.prod(math.comb(x+y,x) for x,y in zip(u,v))
  return s
 @lru_cache(None)
 def leaf(p,q,h):
  W='('*h+'()'*p;Z='('*h+'()'*q
  if not p and not q:return source(W,Z)+(leaf(0,0,h-1) if h else 0)
  value=source(W,Z)
  if p:value+=leaf(p-1,q,h)+source('('*h+'()'*(p-1)+'(',Z+'(')
  if q:value+=leaf(p,q-1,h)+source(W+'(','('*h+'()'*(q-1)+'(')
  return value
 failures=[];keys=0
 original=json.loads((BASE/'outputs/M31/M31_EXACT.json').read_text())
 for L in original['layers']:
  for b in L['blocks']:
   a,c,h=b['key']
   if L['t']-a-c+h!=4:continue
   actual=int(b['max_decimal']);pred=leaf((a-h)//2,(c-h)//2,h);keys+=1
   if actual!=pred:failures.append(dict(key=b['key'],exact_max=actual,leaf_value=pred))
 return dict(status='PASS',cycle=6,file_indexed_cells_checked=tests,all_coefficients_after_forced_spills=counters[0],forced_native_spills=counters[1],terminal_leaf_extremal=dict(status='PASS' if not failures else 'REFUTED',keys=keys,failures=failures,scalar_states=leaf.cache_info().currsize),uses_extremal_conjecture_for_arithmetic=got['conjectural_extremal_bounds_used'])

def rank_integers():
 import compact_v2 as CV
 profiles=0;files=0;prefixes=[]
 for path in sorted(CV.CACHE.glob('*.npz')):
  meta=json.loads(path.with_suffix('.json').read_text());a,h=meta['a'],meta['h']
  if meta['signature']!=CV.CACHE_SIGNATURE:continue
  import hashlib
  assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
  with np.load(path,allow_pickle=False) as z:ranks=z['x2']
  groups,total=CV.profile_counts(a,h)
  assert len(ranks)==groups and int(ranks.sum())==total,(a,h,'total ranks')
  p=(a-h)//2;d=h+1;hist=np.zeros(h+3,np.int64)
  if p==0:hist[0]=1
  for r in range(1,p+1):
   for c in range(1,min(r,d)+1):
    k=p-r;rank=0 if k==0 else c-1 if k==1 else (1 if k==2 else 2) if r==1 else c+1
    hist[rank]+=math.comb(d,c)*math.comb(r-1,c-1)
  assert np.array_equal(np.bincount(ranks,minlength=h+3),hist),(a,h,'rank distribution')
  profiles+=groups;files+=1;prefixes.append([a,h])
 checks=0
 for dt in (np.uint16,np.uint32,np.uint64):
  cap=(1<<62)-1;vals=np.array([[0,17],[19,23]],np.int64)
  large=min(np.iinfo(dt).max, (1<<61)+1)
  v=np.array([[1,large],[large-1,3]],dt);expected=[[int(vals[i,j])+int(v[i,j]) for j in range(2)] for i in range(2)]
  errors=CV.add_sample(vals,v,np.arange(2,dtype=np.int64),np.arange(2,dtype=np.int64),cap)
  assert not np.any(errors) and np.array_equal(vals,np.array(expected,np.int64));checks+=4
 vals=np.array([[(1<<62)-1]],np.int64);original=vals.copy()
 assert np.any(CV.add_sample(vals,np.array([[1]],np.uint64),np.array([0]),np.array([0]),(1<<62)-1)) and np.array_equal(vals,original)
 vals=np.array([[0]],np.int64)
 assert np.any(CV.add_sample(vals,np.array([[(1<<64)-1]],np.uint64),np.array([0]),np.array([0]),(1<<62)-1)) and vals[0,0]==0
 return dict(status='PASS',cycle=8,certified_basis_files=files,profiles_checked=profiles,prefixes=prefixes,integer_sum_cells=checks,overflow_rejections=2,exact_rank_formula=True)

def interior_extremal():
 from functools import lru_cache
 from probe_mixed import newsource,feat
 @lru_cache(None)
 def value(p,q,k,h):
  W='('*k+'()'*p+'('*(h-k);Z='('*k+'()'*q+'('*(h-k)
  result=newsource(W,Z,2)
  if h:
   u=feat(W+')')[0];v=feat(Z+')')[0]
   result+=math.prod(math.comb(x+y,x) for x,y in zip(u,v))
  if not p and not q:return result+(value(0,0,min(k,h-1),h-1) if h else 0)
  if h>k:return result+value(p,q,k,h-1)
  if p:result+=value(p-1,q,k,h+1)
  if q:result+=value(p,q-1,k,h+1)
  return result
 data=json.loads((BASE/'outputs/M31/M31_EXACT.json').read_text());details=[];fail=[]
 for layer in data['layers']:
  for block in layer['blocks']:
   a,b,h=block['key']
   if layer['t']-a-b+h!=4:continue
   p,q=(a-h)//2,(b-h)//2
   choices=[value(p,q,k,h) for k in range(h+1)];pred=max(choices);actual=int(block['max_decimal'])
   item=dict(key=block['key'],exact_max=actual,leaf_max=pred,attaining_levels=[k for k,v in enumerate(choices) if v==pred],center_value=choices[h//2])
   details.append(item)
   if actual!=pred:fail.append(item)
 return dict(status='PASS' if not fail else 'REFUTED',cycle=9,hypothesis='All leaves concentrated at one common, possibly interior level attain the V2 block maximum.',keys_checked=len(details),failures=fail,details=details,scalar_states=value.cache_info().currsize,uses_known_maxima_only_for_posthoc_comparison=True)

def division_audit():
 import compact_v2 as CV,hashlib
 profiles=0;largest=1;files=0
 for path in CV.CACHE.glob('*.npz'):
  meta=json.loads(path.with_suffix('.json').read_text())
  if meta['signature']!=CV.CACHE_SIGNATURE:continue
  assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
  with np.load(path,allow_pickle=False) as z:d=z['x3']
  assert np.all(d>0);largest=max(largest,int(d.max(initial=1)));profiles+=len(d);files+=1
 values=list(range(-101,102))+[-(1<<62)+1,-(1<<53)-1,(1<<53)+1,(1<<62)-1]
 checks=0
 for value in values:
  for denominator in (1,2,3,4,7,8,16,25):
   q,r=CV.integer_division(value,denominator)
   assert int(q)==value//denominator and int(r)==value%denominator,(value,denominator,q,r);checks+=1
 return dict(status='PASS',cycle=12,basis_files=files,basis_profiles=profiles,largest_individual_denominator=largest,division_checks=checks,includes_negative_and_large_integers=True,generic_division_fallback_retained=True)

def source_reuse_tests():
 E,V=configure(26)
 import source_reuse as GS,source_separation as S,projected_source as P3
 cells=0;details=[]
 for key in ((14,14,2),(12,16,0),(16,18,4)):
  K,ip,iq,P,Q,fp,fq,MP,MQ=S.data(26,key,2)
  for dt in (np.uint32,np.uint64):
   kk=K.astype(dt)
   a=GS.row(kk,ip,P,Q,*fp,MP,2);b=S.row_source(kk,ip,P,Q,*fp,MP,2)
   assert np.array_equal(a,b);cells+=a.size
   a=GS.col(kk,iq,P,Q,*fq,MQ,2);b=S.col_source(kk,iq,P,Q,*fq,MQ,2)
   assert np.array_equal(a,b);cells+=a.size
   details.append(dict(key=key,source='V2',dtype=str(np.dtype(dt))))
  K,ip,iq,P,Q,fp,fq=P3.data(26,key) if key[2] else (None,)*7
  if K is not None:
   M=np.empty((0,0),np.uint64)
   a=GS.row(K,ip,P,Q,*fp,M,3);b=P3.row(K,ip,P,Q,*fp);assert np.array_equal(a,b);cells+=a.size
   a=GS.col(K,iq,P,Q,*fq,M,3);b=P3.col(K,iq,P,Q,*fq);assert np.array_equal(a,b);cells+=a.size
   details.append(dict(key=key,source='projected V3',dtype=str(K.dtype)))
 return dict(status='PASS',cycle=13,source_cells_checked=cells,details=details,includes_both_unsigned_widths=True,has_new_conjecture=False)

def basis_preparation(tag):
 import compact_v2 as CV,hashlib,psutil
 E,V=configure(30);original=CV.CACHE
 source=json.loads((BASE/'outputs/MN_TABLE/RUNS/TABLE_CURRENT_M30.json').read_text())
 keys={tuple(b['key']) for L in source['layers'] for b in L['blocks'] if L['t']-sum(b['key'][:2])+b['key'][2]==4 and CV.selected(*b['key'])}
 prefixes=sorted({(x,h) for a,b,h in keys for x in (a,b)})
 records=[];first_hashes=None;fresh=OUT/'FRESH_BASIS'/tag
 assert not fresh.exists()
 try:
  for mode,cache in (('existing',original),('fresh',fresh)):
   V.clear();CV.clear();gc.collect();CV.CACHE=cache;hashes=[];peak=0;start=time.perf_counter()
   for a,h in prefixes:
    result=CV.prefix(a,h);digest=hashlib.sha256()
    for array in result:digest.update(array.dtype.str.encode()+str(array.shape).encode()+array.tobytes())
    hashes.append(digest.hexdigest());peak=max(peak,psutil.Process().memory_info().rss)
   elapsed=time.perf_counter()-start
   if first_hashes is None:first_hashes=hashes
   else:assert hashes==first_hashes,'fresh bases differ from the retained certified coordinates'
   records.append(dict(mode=mode,wall_seconds_including_hashes=elapsed,basis_prepare_seconds=CV.STATS['prepare_seconds'],basis_cache_builds=CV.STATS['basis_cache_builds'],basis_cache_loads=CV.STATS['basis_cache_loads'],peak_RSS_sampled_after_prefix=peak))
 finally:CV.clear();V.clear();CV.CACHE=original;gc.collect()
 return dict(status='PASS',cycle=14,prefixes=prefixes,selected_V2_blocks=len(keys),all_fresh_and_existing_bases_identical=True,details=records,measures_preparation_not_whole_M30=True)
