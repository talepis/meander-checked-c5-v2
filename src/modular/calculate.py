"""Exact checked V2 driver with a mathematically bounded retry.
The global common-level maximum is NOT assumed to be a theorem.
"""
from pathlib import Path
import os,sys,json,gc,time,argparse
from bootstrap import load,W
from coarse_bound import upper

def calculate(n,threads=6,root=None,force_proved=False,_test_predictor=None,layer_check=None):
 if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=34:raise ValueError('Audited supported orders: 1..34')
 if not __debug__:raise RuntimeError('Optimized Python -O/-OO is not supported')
 F,V,BS,G,pool=load('checked',n,threads)
 import extended_v2 as CV,extremal_v2 as X
 from checked_caps import PredictionRejected
 original_prediction=X.maximum;original_init=BS.Store.__init__;stores=[];attempts=[]
 def track(store,*a,**kw):
  original_init(store,*a,**kw);stores.append(store)
 def cleanup():
  G.PRESSURE_HANDLER=None
  for store in stores:
   for block in list(store.live.values()):
    block.pins=0;block.close()
   store.finish()
  stores.clear();CV.GROUP_CACHE_BYTES=0;CV.OWNERS.clear();CV.PRESSURED_STORES.clear();CV.clear();V.clear();gc.collect()
 BS.Store.__init__=track
 root=Path(root) if root else W/('run_'+str(os.getpid())+'_'+str(time.time_ns()))
 try:
  for mode in (['proved'] if force_proved else ['predicted','proved']):
   X.maximum=upper if mode=='proved' else (_test_predictor or original_prediction)
   V.clear();gc.collect();start=time.perf_counter()
   try:
    r=F.meander(n,root.with_name(root.name+'_'+mode),ram_limit=pool,small_limit=pool,chunk=64<<20,verbose=False,force_expansion=n>=30 or os.environ.get('V2_CRT_TEST_EXPANSION')=='1',layer_check=layer_check)
   except PredictionRejected as error:
    attempts.append(dict(mode=mode,status='PREDICTION_REJECTED',seconds=time.perf_counter()-start,reason=str(error)))
    cleanup()
    if mode=='proved':raise ArithmeticError('The proved-cap attempt failed; no answer is accepted') from error
    continue
   except BaseException:
    cleanup();raise
   attempts.append(dict(mode=mode,status='PASS',seconds=time.perf_counter()-start))
   # Replace legacy labels with explicit, non-conflicting meanings.
   r.pop('conjectural_V2_used',None);r.pop('conjectural_extremal_bounds_used',None)
   r.update(status='PASS',variant='V2_EXACT_CHECKED_PREDICTIONS',mathematical_contract='Exact on successful completion without the global extremal hypothesis; bad predictions reject before unsafe arithmetic.',global_extremal_conjecture_proved=False,extremal_prediction_used=mode=='predicted',extremal_hypothesis_required_for_correctness=False,proved_cap_fallback_used=mode=='proved' and not force_proved,attempts=attempts,threads=threads,independent_CRT_performed=False)
   return r
  raise ArithmeticError('No attempt produced an accepted result')
 finally:
  X.maximum=original_prediction;BS.Store.__init__=original_init

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('n',type=int);ap.add_argument('--threads',type=int,default=6);ap.add_argument('--proved-cap',action='store_true');ap.add_argument('--output',type=Path)
 ar=ap.parse_args();r=calculate(ar.n,ar.threads,force_proved=ar.proved_cap)
 if ar.output:ar.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:r[k] for k in ['status','n','answer_decimal','seconds','extremal_prediction_used','extremal_hypothesis_required_for_correctness','proved_cap_fallback_used','attempts']}))
