from pathlib import Path
import os,sys,argparse,json,time,threading,math
ROOT=Path(__file__).resolve().parents[2]

def resources(ram_gib,reserve_gib,threads):
 import psutil
 process=psutil.Process();vm=psutil.virtual_memory();total=int(vm.total);avail=int(vm.available)
 if sys.platform.startswith('linux'):
  mount=Path('/sys/fs/cgroup');roots={mount}
  for line in Path('/proc/self/cgroup').read_text().splitlines():
   if line.startswith('0::'):
    p=mount/line[3:].lstrip('/')
    if p.is_dir() and (p==mount or mount in p.parents):roots.update(x for x in (p,*p.parents) if x==mount or mount in x.parents)
  for p in roots:
   cap=p/'memory.max';used=p/'memory.current'
   if cap.is_file() and used.is_file() and cap.read_text().strip()!='max':
    limit=int(cap.read_text());total=min(total,limit);avail=min(avail,max(0,limit-int(used.read_text())))
 reserve=int(reserve_gib*(1<<30)) if reserve_gib is not None else max(512<<20,min(16<<30,total//20))
 if reserve<512<<20:raise ValueError('Reserve must be at least 0.5 GiB')
 if ram_gib is not None and (not math.isfinite(ram_gib) or ram_gib<=0):raise ValueError('RAM cap must be positive and finite')
 budget=min(avail-reserve,int(ram_gib*(1<<30)) if ram_gib is not None else total)
 if budget<768<<20:raise MemoryError('Less than 0.75 GiB is available after the protected reserve')
 affinity=process.cpu_affinity() if hasattr(process,'cpu_affinity') else list(range(os.cpu_count() or 1))
 auto=min(12,psutil.cpu_count(logical=False) or len(affinity),len(affinity))
 threads=auto if threads is None else threads
 if not 1<=threads<=len(affinity):raise ValueError('Threads must fit the allowed logical CPU affinity')
 os.environ.update(MEANDER_BUDGET_BYTES=str(budget),MEANDER_RESERVE_BYTES=str(reserve),NUMBA_NUM_THREADS=str(threads),NUMBA_THREADING_LAYER='omp',MEANDER_NUMBA_THREADS=str(threads),OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 return dict(threads=threads,budget_bytes=budget,reserve_bytes=reserve,pool_bytes=budget-(512<<20),effective_total_bytes=total,initial_available_bytes=avail)

class Sampler:
 def __init__(self):
  import psutil
  self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.event=threading.Event();self.thread=threading.Thread(target=self.sample,daemon=True)
 def sample(self):
  while not self.event.wait(.005):self.peak=max(self.peak,self.process.memory_info().rss)
 def close(self):
  self.event.set();self.thread.join();self.peak=max(self.peak,self.process.memory_info().rss)

def main():
 if not __debug__:raise RuntimeError('Python -O/-OO is unsupported')
 a=argparse.ArgumentParser();a.add_argument('n',type=int);a.add_argument('--mode',default='exact');a.add_argument('--modulus',type=int);a.add_argument('--threads',type=int);a.add_argument('--ram-gib',type=float);a.add_argument('--reserve-gib',type=float);a.add_argument('--proved-cap',action='store_true');a.add_argument('--output',type=Path);v=a.parse_args()
 if not 0<=v.n<=34 or v.mode not in ('exact','modular','independent'):raise ValueError('Invalid public domain or mode')
 started=time.perf_counter();settings=resources(v.ram_gib,v.reserve_gib,v.threads);sampler=Sampler();sampler.thread.start()
 import public_runtime as RT
 RT.SETTINGS.update(settings)
 output=v.output.resolve() if v.output else ROOT/'runs'/f'M{v.n}_{v.mode}_{time.time_ns()}.json';output.parent.mkdir(parents=True,exist_ok=True)
 runroot=output.parent/(output.stem+'_arrays')
 try:
  if v.n==0:
   if v.mode!='exact':raise ValueError('M0 is only an exact convention')
   result=dict(n=0,status='PASS',answer_decimal='1',seconds=0,scope='OEIS empty-case convention')
  elif v.mode in ('exact','modular'):
   folder=ROOT/'src'/('exact' if v.mode=='exact' else 'modular');sys.path.insert(0,str(folder))
   if v.mode=='exact':
    from calculate import calculate
    result=calculate(v.n,settings['threads'],root=runroot,force_proved=v.proved_cap)
   else:
    from v2_crt import calculate
    result=calculate(v.n,v.modulus,settings['threads'],root=runroot,force_proved=v.proved_cap)
  else:
   if not 1<=v.n<=31 or v.modulus is None:raise ValueError('Independent domain is 1..31 with explicit modulus')
   folder=ROOT/'src/independent';RT.materialize_map(folder/'src/baseline/base/meander_closemaps_n25.pkl');sys.path.insert(0,str(folder))
   import tempfile,hashlib; os.environ['NUMBA_CACHE_DIR']=str(Path(os.environ.get('MEANDER_CACHE_DIR',str(Path(tempfile.gettempdir())/'mc'/hashlib.sha256(str(ROOT).encode()).hexdigest()[:6]))))
   import worker
   result=worker.calculate(v.n,v.modulus,settings['threads'],output,'public-reproduction')
   result['status']='PASS'
 finally:sampler.close()
 result.update(public_release='1.0.0-rc1',resource_settings=settings,public_preparation=RT.TIMINGS,entry_wall_seconds=time.perf_counter()-started,entry_sampled_peak_RSS_bytes=sampler.peak,entry_RSS_sample_interval_ms=5)
 if sys.platform.startswith('linux'):
  import resource
  result['entry_process_lifetime_maxRSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 keys=('status','n','answer_decimal','residue_decimal','modulus_decimal','entry_wall_seconds','entry_sampled_peak_RSS_bytes','resource_settings','proved_cap_fallback_used')
 print(json.dumps({k:result[k] for k in keys if k in result}))

if __name__=='__main__':main()
