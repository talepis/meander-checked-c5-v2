"""Independent RAM-only original four-move enumeration; no candidate is an input."""
from pathlib import Path
import argparse, gc, hashlib, json, os, pickle, sys, threading, time, traceback
HERE=Path(__file__).resolve().parent
SRC=HERE/'src'
META=HERE if (HERE/'FROZEN_IDENTITY.json').exists() else HERE.parents[1]/'outputs/M31_CRT_SERVER_20261023'
FORBIDDEN={'engine_v2','variants_next','projected_source','source_separation','configure_v2',
           'functional_core','functional_engine','extended_v2','independent_verify'}
MAP_SHA='36b69e1c416c2e6888718b39aed0e391559dd16c720e1b39907174b3f848850a'

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_suffix(path.suffix+'.next')
    q.write_text(json.dumps(value,indent=2)+'\n');q.replace(path)

def check_identity():
    manifest=json.loads((META/'SOURCE_SHA256.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen source changed: '+name)
    actual=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert actual==json.loads((META/'FROZEN_IDENTITY.json').read_text())['source_bundle_digest']
    return actual

def load_core(threads=1):
    sys.path.insert(0,str(SRC))
    import allocation_guard as G
    import adaptive_exact as A
    import numpy as np
    import numba
    numba.set_num_threads(threads)
    assert not FORBIDDEN.intersection(sys.modules)
    return A,G,np

class Sampler:
    def __init__(self):
        import psutil
        self.process=psutil.Process();self.stop=threading.Event();self.lock=threading.Lock()
        self.stage_peak=self.total_peak=self.process.memory_info().rss
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.wait(.005):
            rss=self.process.memory_info().rss
            with self.lock:
                self.stage_peak=max(self.stage_peak,rss);self.total_peak=max(self.total_peak,rss)
    def boundary(self):
        rss=self.process.memory_info().rss
        with self.lock:
            peak=max(self.stage_peak,rss);self.stage_peak=rss;self.total_peak=max(self.total_peak,peak)
        return rss,peak

def load_maps(A,n,np):
    payload=(SRC/'baseline/base/meander_closemaps_n25.pkl').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==MAP_SHA
    A.mt._close=pickle.loads(payload)
    from extend_maps import extend
    created=extend(n);needed=set()
    for t in range(2*n-1):
        op,_=A.operations(n,t,A.mt.canon_keys(n,t),2)
        for items in op.values():
            for dest,kind,flip,rect,off,mp in items:
                if mp is not None:needed.add(mp)
    records=[]
    for m,h in sorted(needed):
        a=A.mt._close[(m,h)];target=A.mt.B(m,h-2);limit=(m-h)//2+1
        assert len(a)==A.mt.B(m,h) and int(a.min(initial=0))>=0 and int(a.max(initial=0))<target
        Aguard=sys.modules['allocation_guard'];Aguard.array((target,),np.int64,'map multiplicity audit')
        observed=int(np.bincount(a,minlength=target).max(initial=0))
        assert observed<=limit
        records.append(dict(m=m,h=h,entries=len(a),target_entries=target,maximum_preimages=observed,bound=limit))
    return dict(created=created,checked_maps=records)

def calculate(n,p,threads=1,output=None,token='small-test'):
    if isinstance(n,bool) or isinstance(p,bool) or not 1<=n<=31 or p<2:raise ValueError('Audited domain 1..31')
    D=4+2*((n-1)//2);limit=D*(p-1)
    if limit>=1<<63:raise ValueError('Unsafe accumulation before first addition')
    identity=check_identity();start=time.perf_counter();A,G,np=load_core(threads)
    dtype=np.uint16 if limit<1<<16 else np.uint32 if limit<1<<32 else np.uint64
    bits=np.dtype(dtype).itemsize*8
    maps=load_maps(A,n,np)
    import numba, llvmlite, psutil
    assert not FORBIDDEN.intersection(sys.modules)
    sampler=Sampler();sampler.thread.start()
    setup=time.perf_counter()-start
    cur={(0,0,0):None};history=[];maxpre=0;previous=0
    t0=time.perf_counter();cpu0=time.process_time();io0=psutil.Process().io_counters()
    try:
        for t in range(2*n-1):
            cur,mx,k,expanded=A.old.transfer(n,cur,t,p,dtype)
            if mx>limit:raise ArithmeticError(('Exceeded BEFORE-addition certificate',t+1,mx,limit))
            maxpre=max(maxpre,mx)
            A.ms.profiles.cache_clear();A.old.features.cache_clear()
            rss,peak=sampler.boundary();elapsed=time.perf_counter()-t0
            row=dict(t=t+1,total_stages=2*n-1,elapsed_seconds=elapsed,stage_seconds=elapsed-previous,
                 RSS_end_bytes=rss,RSS_stage_sampled_peak_bytes=peak,RSS_sample_interval_ms=5,
                 coefficient_layer_bytes=sum(v.nbytes for v in cur.values() if v is not None),
                 max_pre_reduce_decimal=str(mx),matrix_disk_read_bytes=0,matrix_disk_write_bytes=0,
                 matrix_disk_bytes=0)
            history.append(row);previous=elapsed
            if output:
                save(Path(output).with_suffix('.progress.json'),dict(n=n,modulus_decimal=str(p),**row))
                save(Path(output).with_suffix('.stages.json'),history)
            print(json.dumps(dict(n=n,modulus_decimal=str(p),**row)),flush=True)
        value=1 if n==1 else 2 if n==2 else int(cur[(1,1,1)][0,0])
        residue=value%p
        math_seconds=time.perf_counter()-t0;cpu_seconds=time.process_time()-cpu0
        del cur;gc.collect()
        assert check_identity()==identity and not FORBIDDEN.intersection(sys.modules)
        io1=psutil.Process().io_counters()
        if os.name=='posix':
            import resource
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        else:peak_rss=max(sampler.total_peak,getattr(psutil.Process().memory_info(),'peak_wset',0))
        result=dict(n=n,modulus_decimal=str(p),residue_decimal=str(residue),storage_bits=bits,
            max_pre_reduce_decimal=str(maxpre),accumulation_bound_decimal=str(limit),D=D,
            setup_and_map_seconds=setup,math_seconds=math_seconds,math_process_cpu_seconds=cpu_seconds,
            worker_elapsed_seconds=time.perf_counter()-start,peak_RSS_bytes=peak_rss,
            sampled_peak_RSS_bytes=sampler.total_peak,peak_coefficient_layer_bytes=max(r['coefficient_layer_bytes'] for r in history),
            matrix_disk_read_bytes=0,matrix_disk_write_bytes=0,matrix_disk_bytes=0,
            process_io_read_bytes=io1.read_bytes-io0.read_bytes,process_io_write_bytes=io1.write_bytes-io0.write_bytes,
            process_io_scope='Includes logs/progress/JIT and ordinary files; distinct from matrix disk I/O, which is absent by construction.',
            threads=threads,threading_backend=numba.threading_layer() if n>=3 else 'not_initialized',
            worker_pid=os.getpid(),run_token=token,source_bundle_digest=identity,
            worker_completed=True,independent_of_C5_V2=True,analytical_shells_used=[0,1],
            arithmetic_scope='Original four-move transfer with retained proved e0/e1 source identities. No C5, V2 or projected sources.',
            maps=maps,layers=history,allocation_checks=G.COUNT,
            largest_checked_allocation_bytes=G.MAX_REQUEST,
            versions=dict(python=sys.version,numpy=np.__version__,numba=numba.__version__,
                          llvmlite=llvmlite.__version__,psutil=psutil.__version__))
        if output:save(output,result)
        return result
    finally:
        sampler.stop.set();sampler.thread.join(timeout=2)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,required=True)
    ap.add_argument('--modulus',type=int,required=True);ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--token',required=True);a=ap.parse_args()
    if a.n==31:
        if int(os.environ.get('M31_SUPERVISOR_PID','-1'))!=os.getppid():raise RuntimeError('M31 requires verified sequential controller')
        gate=json.loads((HERE/'PREDECESSOR_M34_ARCHIVED.json').read_text())
        if gate.get('status')!='M34_PASS_ARCHIVED':raise RuntimeError('M34 predecessor gate has not passed')
    try:
        calculate(a.n,a.modulus,a.threads,a.output,a.token)
    except BaseException as e:
        save(a.output.with_suffix('.error.json'),dict(worker_completed=False,n=a.n,
             modulus_decimal=str(a.modulus),error=repr(e),traceback=traceback.format_exc(),run_token=a.token))
        raise
