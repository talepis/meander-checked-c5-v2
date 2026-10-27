"""Hybrid modular checked-V2 entry point; never reports a residue as Mn."""
from modular_reduction import validate_modulus
def calculate(n,modulus,threads=16,root=None,layer_check=None,force_proved=False,_test_predictor=None):
 validate_modulus(n,modulus)
 from bootstrap import load
 load('checked',n,threads)
 import functional_core as CORE
 from calculate import calculate as exact_driver
 if CORE.CRT_MODULUS is not None:raise RuntimeError('Nested modular calculations unsupported')
 CORE.CRT_MODULUS=modulus
 try:r=exact_driver(n,threads,root=root,layer_check=layer_check,force_proved=force_proved,_test_predictor=_test_predictor)
 finally:CORE.CRT_MODULUS=None
 r['residue_decimal']=str(int(r.pop('answer_decimal'))%modulus)
 r.update(variant='HYBRID_CRT_CHECKED_V2',modulus_decimal=str(modulus),exact_retained_shells=[2],residue_retained_shells='excess >= 3',independent_of_C5_V2=False,independent_CRT_performed=False,CRT_arithmetic_performed=True,modular_divisions_used=False)
 return r
