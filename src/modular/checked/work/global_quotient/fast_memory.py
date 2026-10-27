"""Fresh native available-RAM query; preserve every allocation guard check.

No cached memory readings and no change to Job, reserve or 128-MiB headroom.
MEMORYSTATUSEX.ullAvailPhys and PERFORMANCE_INFORMATION.PhysicalAvailable
measure immediately available physical memory including standby/free/zero.
"""
import ctypes as C
import os,time
import psutil
import allocation_guard as G

class Status(C.Structure):
    _fields_=[('length',C.c_uint32),('load',C.c_uint32),
              ('total',C.c_uint64),('available',C.c_uint64),
              ('total_pagefile',C.c_uint64),('available_pagefile',C.c_uint64),
              ('total_virtual',C.c_uint64),('available_virtual',C.c_uint64),
              ('extended',C.c_uint64)]

K=C.WinDLL('kernel32',use_last_error=True)
QUERY=K.GlobalMemoryStatusEx
QUERY.argtypes=[C.POINTER(Status)];QUERY.restype=C.c_int

def status():
    v=Status();v.length=C.sizeof(v)
    if not QUERY(C.byref(v)):raise C.WinError(C.get_last_error())
    return v

def check(needed,label='allocation'):
    needed=int(needed)
    if needed<0:raise ValueError('Negative allocation estimate')
    budget=int(os.environ.get('MEANDER_BUDGET_BYTES','0'))
    if not budget:return
    reserve=int(os.environ['MEANDER_RESERVE_BYTES']);overhead=128<<20
    mem=psutil.Process().memory_info();committed=max(mem.rss,mem.vms)
    available=status().available
    G.COUNT+=1;G.MAX_REQUEST=max(G.MAX_REQUEST,needed)
    if (committed+needed+overhead>budget or available<reserve+needed+overhead) and G.PRESSURE_HANDLER is not None and not G._HANDLING:
        G._HANDLING=True
        try:G.PRESSURE_HANDLER(needed,overhead)
        finally:G._HANDLING=False
        mem=psutil.Process().memory_info();committed=max(mem.rss,mem.vms);available=status().available
    if committed+needed+overhead>budget or available<reserve+needed+overhead:
        raise MemoryError(dict(label=label,needed_bytes=needed,committed_bytes=committed,
            available_bytes=available,budget_bytes=budget,reserve_bytes=reserve))

def units():
    assert C.sizeof(Status)==64
    pairs=[]
    for _ in range(100):
        a=psutil.virtual_memory();b=status();c=psutil.virtual_memory()
        assert a.total==b.total==c.total
        pairs.append(dict(old_before=a.available,native=b.available,old_after=c.available))
    assert max(min(abs(x['native']-x['old_before']),abs(x['native']-x['old_after'])) for x in pairs)<64<<20
    timings={}
    for name,fn in (('psutil',psutil.virtual_memory),('native',status)):
        start=time.perf_counter()
        for _ in range(3000):fn()
        timings[name]=time.perf_counter()-start
    # Exercise the same rejection/pressure decisions with deterministic mocks.
    original_status=status;old_process=psutil.Process;old_budget=os.environ.get('MEANDER_BUDGET_BYTES')
    old_reserve=os.environ.get('MEANDER_RESERVE_BYTES');old_handler=G.PRESSURE_HANDLER
    global QUERY
    calls=[0];saved_query=QUERY;old_virtual=psutil.virtual_memory;available_mock=[8<<30];decisions=0
    def fake_query(ptr):
        v=C.cast(ptr,C.POINTER(Status)).contents;v.available=available_mock[0];return 1
    class Memory:
        rss=1<<30;vms=1<<30
    class Process:
        def memory_info(self):return Memory()
    try:
        psutil.Process=Process;QUERY=fake_query
        os.environ['MEANDER_BUDGET_BYTES']=str(4<<30);os.environ['MEANDER_RESERVE_BYTES']=str(2<<30)
        from types import SimpleNamespace
        psutil.virtual_memory=lambda:SimpleNamespace(available=available_mock[0])
        G.PRESSURE_HANDLER=None
        for rss in (0,1<<30,4<<30):
            for commit in (0,1<<30,5<<30):
                for avail in (1<<30,2<<30,8<<30):
                    for needed in (-1,0,1<<20,4<<30):
                        Memory.rss=rss;Memory.vms=commit;available_mock[0]=avail
                        outcome=[]
                        for fn in (G.check,check):
                            try:fn(needed);outcome.append('accepted')
                            except (ValueError,MemoryError) as error:outcome.append(type(error).__name__)
                        assert outcome[0]==outcome[1],(rss,commit,avail,needed,outcome)
                        decisions+=1
        Memory.rss=Memory.vms=1<<30;available_mock[0]=8<<30
        check(1<<20)
        for amount in (-1,4<<30):
            try:check(amount)
            except (ValueError,MemoryError):pass
            else:raise AssertionError('missing allocation rejection')
        Memory.rss=5<<30
        def pressure(needed,overhead):calls[0]+=1;Memory.rss=1<<30
        G.PRESSURE_HANDLER=pressure;check(1<<20);assert calls[0]==1
        def failed_query(ptr):return 0
        QUERY=failed_query
        try:status()
        except OSError:pass
        else:raise AssertionError('failed native call accepted')
    finally:
        QUERY=saved_query;psutil.Process=old_process;psutil.virtual_memory=old_virtual;G.PRESSURE_HANDLER=old_handler
        if old_budget is None:os.environ.pop('MEANDER_BUDGET_BYTES',None)
        else:os.environ['MEANDER_BUDGET_BYTES']=old_budget
        if old_reserve is None:os.environ.pop('MEANDER_RESERVE_BYTES',None)
        else:os.environ['MEANDER_RESERVE_BYTES']=old_reserve
    return dict(status='PASS',paired_live_queries=pairs,query_3000_seconds=timings,
        speedup_query=timings['psutil']/timings['native'],guard_decision_tests=decisions+5,
        fresh_query_for_every_guard=True,system_reserve_unchanged=True,headroom_bytes=128<<20)
