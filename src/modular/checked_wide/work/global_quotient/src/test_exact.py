"""Regression and independent tests; reference numbers never enter enumeration code."""
import os,sys,json,pickle,time,math,random,gc
from collections import defaultdict
from pathlib import Path
import numpy as np
import adaptive_exact as ae
import seed_fast as sf
from seed_ranges import RangeInjector
ROOT=Path(__file__).parent
REF=[1,1,2,8,42,262,1828,13820,110954,933458,8152860,73424650,678390116,6405031050,61606881612,602188541928,5969806669034,59923200729046,608188709574124,6234277838531806,64477712119584604,672265814872772972,7060941974458061392]

def root_counts(p):
    depth=0;r=[0]
    for c in p:
        if c=='(':depth+=1;r.append(0)
        else:r.pop();depth-=1;r[depth]+=1
    return r

def splice(w,a):
    stack=[];p=[-1]*len(w)
    for i,c in enumerate(w):
        if c=='(':stack.append(i)
        else:j=stack.pop();p[i]=j;p[j]=i
    x,y=a-1,a
    if p[x]==y:return None
    u,v=p[x],p[y];p[u]=v;p[v]=u
    return ''.join('(' if k<p[k] else ')' for k in range(len(w)) if k!=x and k!=y)

def independent(n):
    cur={('',0):1};checks=0;allstates=0;t0=time.perf_counter()
    for t in range(1,2*n):
        nxt=defaultdict(int);L=2*n-t
        def add(w,a,v):
            if a<=L and len(w)-a<=L:nxt[w,a]+=v
        for (w,a),v in cur.items():
            add(w[:a]+'()'+w[a:],a+1,v)
            if a<len(w):add(w,a+1,v)
            if a>0:add(w,a-1,v)
            if 0<a<len(w):
                ww=splice(w,a)
                if ww is not None:add(ww,a-1,v)
        cur=nxt;allstates+=len(cur)
        for (w,a),v in cur.items():
            p=w[:a];q=''.join(')' if c=='(' else '(' for c in w[a:][::-1])
            h=p.count('(')-p.count(')')
            if t==len(w)-h:
                expected=math.prod(math.comb(u+vv,u) for u,vv in zip(root_counts(p),root_counts(q)))
                assert expected==v,(t,w,a,v,expected)
                checks+=1
    ans=cur['()',1];assert ans==REF[n]
    return {'n':n,'answer':ans,'zero_shell_coefficient_checks':checks,'forward_states_visited':allstates,'seconds':time.perf_counter()-t0}

def reciprocals_test(n=26):
    R=sf.reciprocals(n);k=2*(n-1)//3;cap=k*(1<<k);checks=0
    for d in range(1,n+3):
        q=np.arange(cap//d+1,dtype=np.uint64);a=q*np.uint64(d)
        res=(a*R[d])>>np.uint64(32)
        assert np.array_equal(q,res);checks+=q.size
        qmax=((1<<32)-1)//d
        for qq in [0,1,qmax,qmax//2,max(0,qmax-1)]:
            if qq*d<(1<<32):assert ((qq*d*int(R[d]))>>32)==qq;checks+=1
    return {'n':n,'checks':checks,'numerator_cap_exhaustive':cap,'status':'PASS'}

def maps_test():
    rng=random.Random(160626);checks=0
    for (m,h),arr in sorted(ae.mt._close.items()):
        ranks=range(len(arr)) if len(arr)<20 else [0,len(arr)-1]+[rng.randrange(len(arr)) for _ in range(18)]
        for rr in ranks:
            bits=ae.mt.unrank_bits(m,h,int(rr));stack=[]
            for i,b in enumerate(bits):
                if b:stack.append(i)
                else:stack.pop()
            bits[stack[-1]]=0
            hh,r=ae.mt.rank_bits(bits)
            assert hh==h-2 and r==arr[rr];checks+=1
    return {'checks':checks,'status':'PASS'}

if __name__=='__main__':
    with open(ROOT/'baseline/base/meander_closemaps_n25.pkl','rb') as f:ae.mt._close=pickle.load(f)
    t0=time.perf_counter();out={'independent':independent(14),'reciprocals':reciprocals_test(),'maps':maps_test(),'regression':{}}
    for mode in ['speed','low-memory']:
        ae.COMPACT_ZERO=(mode=='speed');ae.SEED_INJECTOR=RangeInjector(True) if mode=='low-memory' else None
        vals=[]
        for n in range(1,23):
            r=ae.meander(n,strip=1 if mode=='speed' else 2)
            assert r['answer']==REF[n],(mode,n,r['answer'],REF[n]);vals.append(r['answer'])
        out['regression'][mode]={'orders':list(range(1,23)),'values':vals,'status':'PASS'}
    out['seconds']=time.perf_counter()-t0;out['status']='PASS'
    (ROOT/'results'/'REGRESSION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
