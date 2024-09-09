import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, M_PI
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel import prange
cimport cython

# Function to generate normally distributed random numbers using Box-Muller transform
cdef double[:] generate_normal_random(double sigma):
    cdef double u1 = rand() / RAND_MAX
    cdef double u2 = rand() / RAND_MAX
    cdef double[2] vals
    cdef double R = sigma * sqrt(-2.0 * log(u1))
    cdef double arg = 2.0 * M_PI * u2
    vals[0] = R * cos(arg)
    vals[1] = R * sin(arg)
    return vals

# Disable Python interaction and bounds checks for speed
@cython.boundscheck(False)
@cython.wraparound(False)
def create_autocorrelated_data(int length, double sigma, double corr=0.1):
    cdef:
        double eps = sqrt((sigma * sigma) * (1 - corr * corr))
        double[::1] vals = np.zeros(length, dtype=np.double)
        int i = 0

    # Generate random values using parallel processing

    while i < length:
        norm_rand = generate_normal_random(sigma)
        vals[i] = norm_rand[0] * eps
        if i+1<length:
            vals[i+1] = norm_rand[1]
        i += 2
        
    # Apply autocorrelation (this can't be parallelized because of the dependency on previous values)
    for i in range(1, length):
        vals[i] += corr * vals[i - 1]

    return vals