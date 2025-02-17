# exponnorm_functions.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp, erfc
cimport cython

# Define the exponnorm function in Cython for scalar inputs
def exponnorm_scalar(double x, double K, double loc=0, double scale=1):
    cdef double y = (x - loc) / scale
    cdef double vals = 1 / (2 * K**2) - y / K
    vals += log(erfc(((1 / K) - y) / sqrt(2)))
    return exp(vals) / (2 * K * scale)

# Define the scaled_exponnorm function in Cython for scalar inputs
def scaled_exponnorm_scalar(double x, double h, double K, double loc=0, double scale=1):
    return h * exponnorm_scalar(x, K=K, loc=loc, scale=scale)

# Expose the functions to Python using memoryviews
def exponnorm_array(double[::1] x, double K, double loc=0, double scale=1):
    cdef:
        int n = x.shape[0]
        np.ndarray[np.double_t, ndim=1] result = np.zeros(n, dtype=np.double)
        int i
        double y, vals
        double SQRT2 = sqrt(2)
        double coeff = 1 / (2 * K**2)
        double adj = 1/(2 * K * scale)


    for i in range(n):
        y = (x[i] - loc) / scale
        vals = coeff - y / K
        vals += log(erfc(((1 / K) - y) / SQRT2))
        result[i] = exp(vals) * adj

    return result

def scaled_exponnorm_array(double[::1] x, double h, double K, double loc=0, double scale=1):
    cdef:
        int n = x.shape[0]
        np.ndarray[np.double_t, ndim=1] result = np.zeros(n, dtype=np.double)
        int i
        double y, vals
        double SQRT2 = sqrt(2)
        double coeff = 1 / (2 * K**2)
        double adj = h/(2 * K * scale)

    for i in range(n):
        y = (x[i] - loc) / scale
        vals = coeff - y / K
        vals += log(erfc(((1 / K) - y) / SQRT2))
        result[i] = adj * exp(vals)

    return result
