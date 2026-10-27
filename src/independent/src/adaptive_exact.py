"""Original transfer aliases and unchanged geometric map planner only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'baseline'))
import seed_two as old
mt=old.mt;ms=old.ms

def operations(n,t,keys,strip=2):
    allowed={k for k in mt.canon_keys(n,t+1) if t+1-sum(k[:2])+k[2]>=2*strip}
    plan={}
    for key in keys:
        a0,b0,h=key
        e=(t-a0-b0+h)//2
        out=[]
        for flip in range(1+(a0<b0)):
            a,b=(b0,a0) if flip else (a0,b0)
            Na,Nb=mt.B(a,h),mt.B(b,h)
            Up,Uq=mt.B(a-1,h-1),mt.B(b-1,h-1)
            Dp,Dq=Na-Up,Nb-Uq
            def push(dest,kind,rect,off=(0,0),mp=None):
                if dest in allowed and rect[0]<rect[1] and rect[2]<rect[3]:
                    out.append((dest,kind,flip,rect,off,mp))
            push((a+1,b+1,h+1),'add',(0,Na,0,Nb))
            if True:
                if b:
                    push((a+1,b-1,h-1),'add',(0,Na,0,Uq),(mt.B(a,h-2),0))
                    push((a+1,b-1,h+1),'add',(0,Na,Uq,Nb))
                if a:
                    push((a-1,b+1,h-1),'add',(0,Up,0,Nb),(0,mt.B(b,h-2)))
                    push((a-1,b+1,h+1),'add',(Up,Na,0,Nb))
                if a and b and h>=1:
                    push((a-1,b-1,h-1),'cols',(0,Up,Uq,Nb),mp=(b-1,h+1))
                    push((a-1,b-1,h-1),'rows',(Up,Na,0,Uq),mp=(a-1,h+1))
            if a and b:push((a-1,b-1,h+1),'add',(Up,Na,Uq,Nb))
        if out:plan[key]=out
    return plan,allowed
