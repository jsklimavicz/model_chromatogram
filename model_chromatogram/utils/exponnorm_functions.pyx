import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp, erfc
cimport cython

# Define the exponnorm function in Cython for scalar inputs
@cython.boundscheck(False)
@cython.wraparound(False)
def exponnorm_scaler(double x, double K, double loc=0, double scale=1):
    cdef double y = (x - loc) / scale
    cdef double vals = 1 / (2 * K**2) - y / K
    vals += log(erfc(((1 / K) - y) / sqrt(2)))
    return exp(vals) / (2 * K * scale)

# Define the scaled_exponnorm function in Cython for scalar inputs
@cython.boundscheck(False)
@cython.wraparound(False)
def scaled_exponnorm_scaler(double x, double h, double K, double loc=0, double scale=1):
    return h * exponnorm_scaler(x, K=K, loc=loc, scale=scale)

# Expose the functions to Python using memoryviews
@cython.boundscheck(False)
@cython.wraparound(False)
def exponnorm_array(np.ndarray[np.float64_t, ndim=1] x, double K, double loc=0, double scale=1):
    cdef:
        int n = x.shape[0]
        np.ndarray[np.double_t, ndim=1] result = np.zeros(n, dtype=np.double)
        int i
        double y, vals

    for i in range(n):
        y = (x[i] - loc) / scale
        vals = 1 / (2 * K**2) - y / K
        vals += log(erfc(((1 / K) - y) / sqrt(2)))
        result[i] = exp(vals) / (2 * K * scale)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def scaled_exponnorm_array(np.ndarray[np.float64_t, ndim=1] x, double h, double K, double loc=0, double scale=1):
    cdef:
        int n = x.shape[0]
        np.ndarray[np.double_t, ndim=1] result = np.zeros(n, dtype=np.double)
        int i
        double y, vals

    for i in range(n):
        y = (x[i] - loc) / scale
        vals = 1 / (2 * K**2) - y / K
        vals += log(erfc(((1 / K) - y) / sqrt(2)))
        result[i] = h * exp(vals) / (2 * K * scale)

    return result
