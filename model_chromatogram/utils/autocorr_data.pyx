import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, cos, M_PI
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel import prange
cimport cython

# Function to generate normally distributed random numbers using Box-Muller transform
cdef double generate_normal_random() nogil:
    cdef double u1 = rand() / RAND_MAX
    cdef double u2 = rand() / RAND_MAX
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

# Disable Python interaction and bounds checks for speed
@cython.boundscheck(False)
@cython.wraparound(False)
def create_autocorrelated_data(int length, double sigma, double corr=0.1):
    cdef:
        double eps = sqrt((sigma * sigma) * (1 - corr * corr))
        np.ndarray[np.double_t, ndim=1] vals = np.zeros(length, dtype=np.double)
        int i

    # Generate random values using parallel processing
    for i in prange(length, nogil=True):
        vals[i] = eps * generate_normal_random()

    # Apply autocorrelation (this can't be parallelized because of the dependency on previous values)
    for i in range(1, length):
        vals[i] += corr * vals[i - 1]

    return vals