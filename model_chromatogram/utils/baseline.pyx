# cython: binding=True
# cython: language_level=3, boundscheck=False, wraparound=False

################################################################
# Calculates the baseline of a chromatogram with peaks using the asymmetric least
# squares algorithm defined in the reference. 
#
# Reference:    S. Oller-Moreno, A. Pardo, J. M. Jiménez-Soto, J. Samitier and S. Marco, 
#               "Adaptive Asymmetric Least Squares baseline estimation for analytical instruments," 
#               2014 IEEE 11th International Multi-Conference on Systems, Signals & Devices (SSD14), 
#               Barcelona, Spain, 2014, pp. 1-5, doi: 10.1109/SSD.2014.6808837.
#               https://diposit.ub.edu/dspace/bitstream/2445/188026/1/2014_IEEE_Adaptive_MarcoS_postprint.pdf
#
# Written by James Klimavicz 2025
################################################################


import numpy as np
cimport numpy as np
from scipy.linalg import solve_banded
from libc.math cimport exp, fabs, isnan
cimport cython

from .pentadiagonal cimport sym_pent_solve

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

cdef inline bint has_nan(double[::1] arr):
    cdef Py_ssize_t i, n = arr.shape[0]
    for i in range(n):
        if isnan(arr[i]):
            return True
    return False

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


    d2_2up = np.array([*[s] * (size - 2)])
    d2_1up = np.array([-3.0 * s, *[-4.0 * s] * (size - 3), -3.0 * s])
    d2_0 = np.array([2.0 * s, *[6.0 * s] * (size - 2), 2.0 * s])

    prev_loss = 1e12

    '''
    Main iteration loop using the pentadiagonal solver.
    This solver requires the matrix to be symmetric and strongly nonsingular. 
    Note that the matrix (W + D2_s) is symmetric, but is not diagonally dominant, and
    therefore may not be strongly nonsingular. This manifests as the presense of nan values
    in the solution z. If this occurs, we fall back to the scipy banded solver, which takes
    a little longer to solve the system, but is more robust.
    '''
    while (not converged) and (iterations < 200):
        iterations += 1
        # Solve for z: (W + D2_s) z = weights * signal
        z = sym_pent_solve(d2_0 + weights, d2_1up, d2_2up, weights * signal)

        if iterations == 1 and has_nan(z):
            break
        # Update residuals
        residuals = signal - z
        curr_loss = loss_function_outside(weights, residuals, z, s)
        rel_loss = fabs(curr_loss - prev_loss) / curr_loss
        if rel_loss < rel_tol:
            converged = True
            return time, z
        prev_loss = curr_loss

        # Update weights using vectorized np.where.
        for i in range(size):
            if residuals[i] > 0:
                weights[i] = p * exp(-residuals[i] / k)
            else:
                weights[i] = 1 - p
    
    if not converged:

        d2_2u = np.array([0.0, 0.0, *[s] * (size - 2)])
        d2_1u = np.array([0.0, -3.0 * s, *[-4.0 * s] * (size - 3), -3.0 * s])
        d2_0 = np.array([2.0 * s, *[6.0 * s] * (size - 2), 2.0 * s])
        d2_1l = np.array([-3.0 * s, *[-4.0 * s] * (size - 3), -3.0 * s, 0.0])
        d2_2l = np.array([*[s] * (size - 2), 0.0, 0.0])

        D2_diag = np.vstack([d2_2u, d2_1u, d2_0, d2_1l, d2_2l])

        while (not converged) and (iterations < 200):
            iterations += 1
            # Solve for z: (W + D2_s) z = weights * signal
            D2_diag[2, :] = weights + d2_0
            z = solve_banded((2, 2), D2_diag, weights * signal)
            # Update residuals
            residuals = signal - z
            curr_loss = loss_function_outside(weights, residuals, z, s)
            rel_loss = fabs(curr_loss - prev_loss) / curr_loss
            if rel_loss < rel_tol:
                converged = True
                return time, z
            prev_loss = curr_loss

            # Update weights using vectorized np.where.
            for i in range(size):
                if residuals[i] > 0:
                    weights[i] = p * exp(-residuals[i] / k)
                else:
                    weights[i] = 1 - p


    return time, z
