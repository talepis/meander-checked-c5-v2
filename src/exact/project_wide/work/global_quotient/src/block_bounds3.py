"""Certified, target-oriented upper bounds for excess-one blocks."""
import math,heapq
import numpy as np
import engine_v2 as E

def C(n,k):return math.comb(n,k) if 0<=k<=n else 0

def refine(n,t,U,previous_maxima):
    for key in U:
        a,b,h=key
        if t-a-b+h!=2:continue
        p=(a-h)//2;q=(b-h)//2;k=p+q
        def mx(aa,bb,hh):return previous_maxima.get((min(aa,bb),max(aa,bb),hh),0)
        z=mx(a-1,b-1,h-1);x=mx(a-1,b+1,h+1);y=mx(a+1,b-1,h+1)
        # Full last forests force a final ')'; an empty last forest forces '('
        # when h>0. Each binomial-product bound uses Vandermonde injection.
        if h:
            cap=max(z+C(k+1,p)+C(k+1,q),
                    x+C(k+1,q+1), y+C(k+1,p+1),
                    x+y+C(k,p+1)+C(k,q+1))
        else:cap=x+y+C(k,p+1)+C(k,q+1)
        U[key]=min(U[key],cap)


def memory_order(cur,plan,U,allocated):
    """Greedy order minimizing newly allocated bytes minus released source bytes."""
    targets={}
    for key,bound in U.items():
        a,b,h=key
        targets[key]=E.mt.B(a,h)*E.mt.B(b,h)*np.dtype(E.dtype_for(bound,'mixed')).itemsize
    parents={};children={};cost={};version={};heap=[]
    for key,A in cur.items():
        ch=set(op[0] for op in plan.get(key,[]) if op[0] not in allocated)
        children[key]=ch;version[key]=0;cost[key]=sum(targets[x] for x in ch)-A.nbytes
        heapq.heappush(heap,(cost[key],0,key))
        for x in ch:parents.setdefault(x,[]).append(key)
    created=set(allocated);done=set();order=[]
    while heap:
        _,ver,key=heapq.heappop(heap)
        if key in done or ver!=version[key]:continue
        done.add(key);order.append(key)
        for x in children[key]:
            if x in created:continue
            created.add(x)
            for y in parents[x]:
                if y in done:continue
                cost[y]-=targets[x];version[y]+=1
                heapq.heappush(heap,(cost[y],version[y],y))
    assert len(order)==len(cur)
    return order

def add_forcing_bounds(n,t,allowed,U):
    """Direct source certificates, eliminating point-map work for analytic inputs."""
    for key in allowed:
        a,b,h=key;e=(t-a-b+h)//2;p=(a-h)//2;q=(b-h)//2;k=p+q
        bound=0
        if e==1:
            bound=C(k,p+1)+C(k,q+1)
            if h:bound=max(bound,C(k+1,p)+C(k+1,q),C(k+1,p+1),C(k+1,q+1))
        elif e==2 and h:bound=C(k+2,p+1)
        if bound:U[key]=U.get(key,0)+bound
