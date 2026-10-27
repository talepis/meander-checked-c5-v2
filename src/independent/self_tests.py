"""Independent small-order coefficient tests and exact-arithmetic edge cases."""
from pathlib import Path
import copy, gc, hashlib, json, math, os, sys, time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import worker as W
from check_certificate import check,reconstruct
REF={1:1,2:2,3:8,4:42,5:262,6:1828,7:13820,8:110954,10:8152860,12:678390116}

def matchings(n):
    if n==0:return [()]
    return [((0,2*k+1),)+tuple((i+1,j+1) for i,j in a)+tuple((i+2*k+2,j+2*k+2) for i,j in b)
            for k in range(n) for a in matchings(k) for b in matchings(n-1-k)]

def brute(n):
    maps=[]
    for pairs in matchings(n):
        m=[0]*(2*n)
        for a,b in pairs:m[a]=b;m[b]=a
        maps.append(m)
    count=0
    for a in maps:
        for b in maps:
            seen={0};stack=[0]
            while stack:
                v=stack.pop()
                for q in (a[v],b[v]):
                    if q not in seen:seen.add(q);stack.append(q)
            count+=len(seen)==2*n
    return count

def main(output=None):
    started=time.perf_counter();identity=W.check_identity()
    A,G,np=W.load_core(int(os.environ.get('NUMBA_NUM_THREADS','1')))
    moduli=[2**57-1,2**57-3,1927];cells=0;results=[]
    for n in REF:
        if n<=6:assert brute(n)==REF[n]
        for p in moduli:
            D=4+2*((n-1)//2);dtype=np.uint16 if D*(p-1)<2**16 else np.uint64
            dense={(0,0,0):np.array([[1]],dtype=dtype)}
            shell={(0,0,0):None}
            for t in range(2*n-1):
                dense,_=A.mt.transfer_sym_lazy(n,dense,t,p,dtype)
                shell,mx,_,_=A.old.transfer(n,shell,t,p,dtype)
                assert mx<=D*(p-1)
                for key,value in shell.items():
                    if value is not None:
                        assert np.array_equal(value,dense[key]),(n,p,t,key)
                        cells+=value.size
            result=1 if n==1 else 2 if n==2 else int(shell[(1,1,1)][0,0])
            full=int(dense[(1,1,1)][0,0]) if n>1 else 1
            assert result%p==full%p==REF[n]%p
            results.append(dict(n=n,modulus_decimal=str(p),residue_decimal=str(result%p)))
            A.ms.profiles.cache_clear();A.old.features.cache_clear();del dense,shell;gc.collect()
    # Original e1 profile recurrence checked using Python integers, no NumPy arithmetic.
    import random
    rng=random.Random(31023);profile_cells=0
    def C(m,k):return math.comb(m,k) if 0<=k<=m else 0
    for h in (0,1,3,5):
        P=np.zeros((6,h+1),np.uint8);Q=np.zeros_like(P)
        for T in (P,Q):
            for row in T:
                for _ in range(12):row[rng.randrange(h+1)]+=1
        K0,K1=A.old.profile_kernels(P,Q,A.old.choose(31))
        for i,pp in enumerate(P):
            for j,qq in enumerate(Q):
                Z=AA=BB=0;L=1
                for uu,vv in zip(pp,qq):
                    u=int(uu);v=int(vv);m=u+v;c=C(m,u)
                    loc=c*m+C(m,u+1)*u+C(m,v+1)*v+2*((v+1)*C(m+1,v+2)+(u+1)*C(m+1,u+2))
                    Z=c*Z+AA*C(m+1,u)+BB*C(m+1,v)+L*loc
                    AA=L*C(m+1,v);BB=L*C(m+1,u);L*=c
                assert int(K0[i,j])==L and int(K1[i,j])==Z
                profile_cells+=1
    # The actual modular arithmetic is explicitly unsigned above 2^53.
    for p in moduli[:2]:
        values=[0,p-1,2*(p-1),17*(p-1),34*(p-1),2**63-1]
        arr=np.array(values,np.uint64);np.remainder(arr,np.uint64(p),out=arr)
        assert arr.tolist()==[x%p for x in values]
    from extend_maps import build_map
    checked_maps=0
    for m in range(2,14):
        for h in range(2,m+1,2):
            if not A.mt.B(m,h):continue
            tab=np.zeros((m+1,m+3),np.int64)
            for j in range(m+1):
                for k in range(m+2):tab[j,k]=A.mt.B(j,k)
            before=A.mt._close.pop((m,h),None)
            slow=A.mt.close_map(m,h);fast=build_map(m,h,A.mt.B(m,h),tab)
            assert np.array_equal(slow,fast)
            checked_maps+=1
    # A small certificate is checked positively and deliberately corrupted.
    n=8;answer=REF[n];records=[]
    for p in moduli:
        records.append(dict(n=n,modulus_decimal=str(p),residue_decimal=str(answer%p),
             worker_completed=True,independent_of_C5_V2=True,storage_bits=64 if p>2**16 else 16,
             source_bundle_digest='SELF_TEST_ONLY',max_pre_reduce_decimal='0'))
    x,P=reconstruct(records)
    cert=dict(n=n,candidate_decimal=str(answer),records=records,source_bundle_digest='SELF_TEST_ONLY',
          reconstructed_decimal=str(x),moduli_product_decimal=str(P),
          upper_bound_decimal=str((math.comb(2*n,n)//(n+1))**2))
    check(cert);rejected=0
    bad=[]
    c=copy.deepcopy(cert);c['records'][0]['residue_decimal']='0';bad.append(c)
    c=copy.deepcopy(cert);c['records'].append(copy.deepcopy(c['records'][0]));bad.append(c)
    c=copy.deepcopy(cert);c['records']=[c['records'][-1]];bad.append(c)
    c=copy.deepcopy(cert);c['records'][0]['worker_completed']=False;bad.append(c)
    c=copy.deepcopy(cert);c['candidate_decimal']=True;bad.append(c)
    c=copy.deepcopy(cert);c['records'][0]['source_bundle_digest']='CHANGED';bad.append(c)
    for c in bad:
        try:check(c)
        except ValueError:rejected+=1
        else:raise AssertionError('Corrupted certificate accepted')
    # Oversized request must be rejected before allocation, with no large allocation.
    try:G.check(int(os.environ['MEANDER_BUDGET_BYTES'])+1,'synthetic oversized request')
    except MemoryError:pass
    else:raise AssertionError('Memory guard failed')
    assert W.check_identity()==identity and not W.FORBIDDEN.intersection(sys.modules)
    result=dict(status='PASS',scope='Small-order native tests, not a CRT execution of M31.',
        source_bundle_digest=identity,elapsed_seconds=time.perf_counter()-started,
        known_counts_and_full_four_move_coefficients=results,coefficient_comparisons=cells,
        independent_brute_orders=list(range(1,7)),python_integer_profile_kernel_cells=profile_cells,
        generated_maps_compared_to_original_rank_unrank=checked_maps,
        corrupted_certificates_rejected=rejected,oversized_allocation_rejected=True,
        tested_moduli_decimal=[str(p) for p in moduli])
    if output:W.save(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='known_counts_and_full_four_move_coefficients'}),flush=True)
    return result

if __name__=='__main__':
    output=Path(sys.argv[1]) if len(sys.argv)>1 else W.META/'SELF_TESTS_LOCAL.json'
    main(output)
