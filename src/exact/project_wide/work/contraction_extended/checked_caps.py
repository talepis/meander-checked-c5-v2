"""Checked predictions: underestimates reject, never return wrapped data."""
import numpy as np
class PredictionRejected(ArithmeticError):
    pass

def require_sample(errors):
    if np.any(errors):raise PredictionRejected('Predicted V2 sample cap rejected before accumulation')

def require_decoded(errors):
    if np.any(errors & np.int32(3)):
        raise ArithmeticError('V2 exact contraction divisibility or nonnegativity failed')
    if np.any(errors):raise PredictionRejected('Predicted V2 cap rejected on exact decoded value')
