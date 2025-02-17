# cython: binding=True
# cython: language_level=3, boundscheck=False, wraparound=False

################################################################
# Solve a pentadiagonal system Ax=b where A is a symmetric strongly nonsingular matrix
# 
# If A is not a pentadiagonal matrix, results will be wrong
#
# Reference: G. Engeln-Muellges, F. Uhlig, "Numerical Algorithms with C"
#               Chapter 4. Springer-Verlag Berlin (1996)
#            Greg von Winckel, https://www.mathworks.com/matlabcentral/fileexchange/4671-fast-pentadiagonal-system-solver
#
# Written by James Klimavicz 2025
################################################################


import numpy as np
cimport numpy as np
cimport cython

cdef sym_pent_solve(double[::1] d, 
                   double[::1] f,  
                   double[::1] e,
                   double[::1] b):
    """
    Solve a pentadiagonal system Ax=b where A is a symmetric strongly nonsingular matrix.
    
    Args:
        d (double[::1]): Main diagonal of the pentadiagonal matrix, length N
        f (double[::1]): First diagonal above the main diagonal, length N-1
        e (double[::1]): Second diagonal above the main diagonal, length N-2
        b (double[::1]): Right-hand side vector, length N
        
    Returns:
        np.ndarray[double, ndim=1]: Solution to the pentadiagonal system.
    """
    cdef Py_ssize_t N = b.shape[0]
    cdef Py_ssize_t i, k

    # Allocate working arrays for the factorization.
    # alpha will store the diagonal of D,
    # gamma and delta correspond to multipliers.
    cdef np.ndarray[double, ndim=1] alpha = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] gamma = np.empty(N-1, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] delta = np.empty(N-2, dtype=np.float64)

    # Arrays for forward substitution and the final solution.
    cdef np.ndarray[double, ndim=1] z = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] c_arr = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] x = np.empty(N, dtype=np.float64)

    # === LDL' Factorization ===
    alpha[0] = d[0]
    gamma[0] = f[0] / alpha[0]
    delta[0] = e[0] / alpha[0]

    alpha[1] = d[1] - f[0] * gamma[0]
    gamma[1] = (f[1] - e[0] * gamma[0]) / alpha[1]
    delta[1] = e[1] / alpha[1]

    for k in range(2, N-2):
        alpha[k] = d[k] - e[k-2] * delta[k-2] - alpha[k-1] * gamma[k-1] * gamma[k-1]
        gamma[k] = (f[k] - e[k-1] * gamma[k-1]) / alpha[k]
        delta[k] = e[k] / alpha[k]
    
    # Handle the last two rows separately.
    alpha[N-2] = d[N-2] - e[N-4] * delta[N-4] - alpha[N-3] * gamma[N-3] * gamma[N-3]
    gamma[N-2] = (f[N-2] - e[N-3] * gamma[N-3]) / alpha[N-2]
    alpha[N-1] = d[N-1] - e[N-2] * delta[N-2] - alpha[N-2] * gamma[N-2] * gamma[N-2]
    
    # === Forward Substitution: Solve L * z = b ===
    z[0] = b[0]
    z[1] = b[1] - gamma[0] * z[0]
    for k in range(2, N):
        z[k] = b[k] - gamma[k-1] * z[k-1] - delta[k-2] * z[k-2]
    
    # Compute c = z ./ alpha
    for i in range(N):
        c_arr[i] = z[i] / alpha[i]
    
    # === Back Substitution: Solve L' * x = c ===
    x[N-1] = c_arr[N-1]
    x[N-2] = c_arr[N-2] - gamma[N-2] * x[N-1]
    for k in range(N-3, -1, -1):
        x[k] = c_arr[k] - gamma[k] * x[k+1] - delta[k] * x[k+2]
    
    return x