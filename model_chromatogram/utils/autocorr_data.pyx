# autocorr_data.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, M_PI
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel import prange
cimport cython

# Disable Python interaction and bounds checks for speed
@cython.boundscheck(False)
@cython.wraparound(False)
def create_autocorrelated_data(int length, double sigma, double corr=0.1):
    cdef:
        double eps = sqrt((sigma * sigma) / (1 - corr * corr))
        double[::1] vals = np.zeros(length, dtype=np.double)
        int i = 0
        double u1, u2, R, arg
        double rho = 2.0 * M_PI
        double adj_var = eps * sigma

    for i in prange(0, length, 2, nogil=True):
    # generate normally distributed noise with box-mueller transform
        u1 = rand() / RAND_MAX
        u2 = rand() / RAND_MAX
        R = adj_var * sqrt(-2.0 * log(u1))
        arg = rho * u2
        vals[i] = R * cos(arg)
        if i+1<length:
            vals[i+1] = R * sin(arg)

    # Apply autocorrelation (this can't be parallelized because of the dependency on previous values)
    vals[0] *= corr
    for i in range(1, length):
        vals[i] += corr * vals[i - 1]

    return vals