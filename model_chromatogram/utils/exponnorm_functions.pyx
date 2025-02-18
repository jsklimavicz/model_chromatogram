# exponnorm_functions.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp, erfc
cimport cython
from .utils cimport binary_search

# Define the exponnorm function in Cython for scalar inputs
def exponnorm_scalar(double x, double K, double loc=0, double scale=1, double cutoff=1e-8):
    cdef double y = (x - loc) / scale, temp
    cdef double vals = 1 / (2 * K**2) - y / K
    vals += log(erfc(((1 / K) - y) / sqrt(2)))
    temp = exp(vals) / (2 * K * scale)
    if temp > cutoff:
        return temp
    else:
        return 0

# Define the scaled_exponnorm function in Cython for scalar inputs
def scaled_exponnorm_scalar(double x, double h, double K, double loc=0, double scale=1):
    return h * exponnorm_scalar(x, K=K, loc=loc, scale=scale)

# Expose the functions to Python using memoryviews
def exponnorm_array(double[::1] x, double K, double loc=0, double scale=1, double cutoff=1e-8):
    """
    Compute the exponentially modified normal (exponnorm) values for array x,
    but only around the peak.  The function computes the values forward from
    the approximate peak index and backward from the peak until the computed
    value falls below cutoff.  Values outside this region remain zero.
    
    Parameters:
      x       : 1D array of x values (assumed sorted in ascending order)
      K       : Parameter K in the exponnorm function
      loc     : Location parameter
      scale   : Scale parameter
      cutoff  : Cutoff threshold below which values are set to zero
      
    Returns:
      result  : 1D NumPy array with computed exponnorm values.
    """
    cdef:
        int n = x.shape[0]
        np.ndarray[np.double_t, ndim=1] result = np.zeros(n, dtype=np.double)
        int i, peak_ind
        double y, vals
        double SQRT2 = sqrt(2.0)
        double coeff = 1.0 / (2.0 * K**2)
        double adj = 1.0 / (2.0 * K * scale)
    
    peak_ind = binary_search(x, n, loc)
    
    # Get an approximate peak index.  This binary_search function should
    # return the smallest index such that x[i] >= loc.
    # (You can replace this with your own implementation.)

    # Process forward from peak_ind until the value drops below cutoff.
    for i in range(peak_ind, n):
        y = (x[i] - loc) / scale
        vals = coeff - y / K
        vals += log(erfc(((1.0 / K) - y) / SQRT2))
        vals = exp(vals) * adj
        if vals <= cutoff:
            break  # values beyond i will remain 0
        result[i] = vals

    # Process backward from (peak_ind - 1) down to index 0.
    for i in range(peak_ind - 1, -1, -1):
        y = (x[i] - loc) / scale
        vals = coeff - y / K
        vals += log(erfc(((1.0 / K) - y) / SQRT2))
        vals = exp(vals) * adj
        if vals <= cutoff:
            break  # values before i remain 0
        result[i] = vals

    return result

def scaled_exponnorm_array(double[::1] x, double h, double K, double loc=0, double scale=1, double cutoff=1e-8):
    return h * exponnorm_array(x, K=K, loc=loc, scale=scale, cutoff=cutoff)
