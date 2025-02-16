# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from libc.math cimport exp, fabs
cimport cython

cdef double loss_function_outside(double[:] weights,
                   double[:] residuals,
                   double[:] z,
                   double s):
    cdef int n = residuals.shape[0]
    cdef double S = 0.0
    cdef double temp
    cdef int i

    for i in range(n):
        S += weights[i] * residuals[i] * residuals[i]
        if i == 0:
            temp = z[i] - z[i + 1]
        elif i == n - 1:
            temp = z[i] - z[i - 1]
        else:
            temp = 2 * z[i] - z[i - 1] - z[i + 1]
        S += s * temp * temp
    return S

def als_psalsa(np.ndarray[double, ndim=1] raw_time,
               np.ndarray[double, ndim=1] raw_signal,
               int sr=5, double p=0.001, double s=1, double k=2, double rel_tol=1e-6):
    """
    Peaked Signal’s Asymmetric Least Squares Algorithm in Cython.
    
    Args:
        raw_time (np.ndarray[double]): Chromatogram time values.
        raw_signal (np.ndarray[double]): Chromatogram signal values.
        sr (int): Sampling rate to downsample the raw time and raw signal.
        p (double): Weight parameter, typically 0.001 <= p <= 0.1.
        s (double): Smoothness penalty factor.
        k (double): Parameter controlling the exponential decay of weight for peaks.
        rel_tol (double): Relative tolerance for convergence.
    
    Returns:
        tuple: (time, z) where both are NumPy arrays.
    """
    cdef int interval, size, iterations
    cdef double dt, interval_float, prev_loss, curr_loss, rel_loss, constant_val
    cdef bint converged
    cdef object D, D2_s, W  

    # Determine interval for downsampling
    dt = raw_time[1] - raw_time[0]
    interval_float = sr / (60.0 * dt)
    interval = <int>(round(interval_float))
    
    # Downsample the raw arrays (make copies so we have contiguous data)
    cdef np.ndarray[double, ndim=1] time = np.array(raw_time[::interval], dtype=np.double)
    cdef np.ndarray[double, ndim=1] signal = np.array(raw_signal[::interval], dtype=np.double)
    size = time.shape[0]
    
    # Initialize z to a constant value computed from the first and last signal entries.
    constant_val = signal[0] * signal[size]
    cdef np.ndarray[double, ndim=1] z = np.empty(size, dtype=np.double)
    z.fill(constant_val)
    cdef np.ndarray[double, ndim=1] residuals = signal - z

    # Initialize weights
    cdef np.ndarray[double, ndim=1] weights = np.ones(size, dtype=np.double)
    converged = False
    iterations = 0

    # Build the sparse matrix D and its smoothness penalized version D2_s.
    cdef np.ndarray[double, ndim=1] a = -1.0 * np.ones(size - 1, dtype=np.double)
    cdef np.ndarray[double, ndim=1] b = 2.0 * np.ones(size, dtype=np.double)
    b[0] = 1.0
    b[size] = 1.0
    D = diags([a, b, a], [-1, 0, 1])
    D2_s = s * (D @ D)

    prev_loss = 0.0

    # Main iteration loop
    while (not converged) and (iterations < 100):
        iterations += 1
        # Build the diagonal weight matrix W (assignment only; W was declared earlier)
        W = diags([weights], [0])
        # Solve for z: (W + D2_s) z = weights * signal
        z = spsolve(W + D2_s, weights * signal)
        # Update residuals
        residuals = signal - z
        curr_loss = loss_function_outside(weights, residuals, z, s)
        rel_loss = fabs(curr_loss - prev_loss) / curr_loss
        if rel_loss < rel_tol:
            converged = True
            break
        prev_loss = curr_loss

        # Update weights using vectorized np.where.
        for i in range(size):
            if residuals[i] > 0:
                weights[i] = p * exp(-residuals[i] / k)
            else:
                weights[i] = 1 - p
    
    return time, z
