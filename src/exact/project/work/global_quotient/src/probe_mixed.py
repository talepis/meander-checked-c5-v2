"""Exact-integer experiment: sum excess-one predecessors by terminal root moments."""
import sys, math, json, time, random
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
(ROOT/'reproduction_results').mkdir(exist_ok=True)
from test_exact import splice

def C(n,k):return math.comb(n,k) if n>=0 and 0<=k<=n else 0

def feat(p):
    forest=[[]];sigma=0
    for c in p:
        if c=='(':forest.append([])
        else:
            kids=forest.pop();d=len(kids);sigma+=(d+1)**2
            forest[-1].append(d)
    u=tuple(map(len,forest));r=tuple(map(sum,forest))
    return u,sigma-sum(u),r,forest

def leaf1(u,v):
    Z=0;L=1;A=B=0
    for x,y in zip(u,v):
        m=x+y;c=C(m,x)
        loc=c*m+C(m,x+1)*x+C(m,y+1)*y+2*((y+1)*C(m+1,y+2)+(x+1)*C(m+1,x+2))
        Z=c*Z+A*C(m+1,x)+B*C(m+1,y)+L*loc
        A=L*C(m+1,y);B=L*C(m+1,x);L*=c
    return L,Z

def val1(u,s,r,v,t,z):
    K,W=leaf1(u,v);W+=K*(s+t)
    for i in range(len(u)):
        W+=K*v[i]//(u[i]+1)*r[i]
        W+=K*u[i]//(v[i]+1)*z[i]
    return W

def predJ_leaf(u,v):
    ans=0
    for k in range(v[-1]):
        U=tuple(u)+(0,);V=tuple(v[:-1])+(k,1)
        K,W=leaf1(U,V);w=v[-1]-1-k
        ans+=W+K*w*(w+3)
    return ans

def mixed(P,Q):
    u,s,r,fp=feat(P);v,t,z,fq=feat(Q)
    h=len(u)-1;L=math.prod(C(x+y,x) for x,y in zip(u[:-1],v[:-1]))
    x,y=u[-1],v[-1]
    B=L*(C(x+y,x+1)+C(x+y,y+1))
    ans=predJ_leaf(u,v)+predJ_leaf(v,u)+B*(s+t)
    for i in range(h):
        ans+=B*v[i]//(u[i]+1)*r[i]+B*u[i]//(v[i]+1)*z[i]
    ans+=L*((C(x+y,x+2)+C(x+y,x))*r[h]+(C(x+y,y+2)+C(x+y,x))*z[h])
    ans+=L*sum(C(y+k,y)*deg for k,deg in enumerate(fp[h]))
    ans+=L*sum(C(x+k,x)*deg for k,deg in enumerate(fq[h]))
    return ans

def closef(u,s,r):
    d=u[-1]
    return tuple(u[:-2])+(u[-2]+1,),s+d*(d+3),tuple(r[:-2])+(r[-2]+d,)

def newsource(P,Q,e):
    u,s,r,_=feat(P);v,t,z,_=feat(Q);h=len(u)-1
    if e==3:
        if not h:return 0
        return val1(*closef(u,s,r),*closef(v,t,z))
    assert e==2
    ans=mixed(P,Q)
    if h and not u[-1]:ans+=val1(u[:-1],s,r[:-1],*closef(v,t,z))
    if h and not v[-1]:ans+=val1(*closef(u,s,r),v[:-1],t,z[:-1])
    return ans

def check(n):
    from collections import defaultdict
    start=time.perf_counter();cur={('',0):1};checks={2:0,3:0};first=None
    for t in range(1,2*n):
        nxt=defaultdict(int);force=defaultdict(int);L=2*n-t
        for (w,a),weight in cur.items():
            h=2*w[:a].count('(')-a;e=(t-1-len(w)+h)//2
            def add(ww,aa):
                if aa>L or len(ww)-aa>L:return
                nxt[ww,aa]+=weight
                if e==1:
                    hh=2*ww[:aa].count('(')-aa;ee=(t-len(ww)+hh)//2
                    if ee>1:force[ww,aa]+=weight
            add(w[:a]+'()'+w[a:],a+1)
            if a<len(w):add(w,a+1)
            if a>0:add(w,a-1)
            if 0<a<len(w):
                ww=splice(w,a)
                if ww is not None:add(ww,a-1)
        for (w,a),weight in nxt.items():
            h=2*w[:a].count('(')-a;e=(t-len(w)+h)//2
            if e in (2,3):
                P=w[:a];Q=''.join(')' if z=='(' else '(' for z in w[a:][::-1])
                predicted=newsource(P,Q,e);actual=force.get((w,a),0)
                if predicted!=actual:return {'failed':(n,t,P,Q,e,predicted,actual)}
                checks[e]+=1
        cur=nxt
    return {'n':n,'checks':checks,'answer':str(cur['()',1]),'seconds':time.perf_counter()-start,'status':'PASS'}
if __name__=='__main__':
    out=check(int(sys.argv[1]) if len(sys.argv)>1 else 12)
    print(json.dumps(out),flush=True)
    open(str(ROOT/'reproduction_results')+'/MIXED_SOURCE_PROBE.json','w').write(json.dumps(out,indent=2))
