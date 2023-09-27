import controldiffeq
from .interpolation_linear import linear_interpolation_coeffs

def get_interp_coeffs(X,times,interpolate='cubic_spline',append_times=True):
    if interpolate == 'cubic_spline':
            coeffs = controldiffeq.natural_cubic_spline_coeffs(times, X)
    elif interpolate == 'linear':
        coeffs = linear_interpolation_coeffs(X, t=times)
    elif interpolate == 'rectilinear':
        assert append_times
        coeffs = linear_interpolation_coeffs(X, t=None,rectilinear=len(X.shape)-1)
        
    else:
        raise TypeError('Not supported interploation type!')
    return coeffs