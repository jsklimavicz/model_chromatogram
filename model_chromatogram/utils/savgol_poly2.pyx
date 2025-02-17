# savgol_poly2.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport fabs
cimport cython

#----------------------------------------------------------------------
# Module-level static arrays for recommended window sizes.
#----------------------------------------------------------------------
cdef double coeffs_5[5]
cdef double coeffs_7[7]

# Module-level flag to ensure initialization happens only once.
cdef bint _static_coeffs_initialized = False

# Module-level cache for fallback coefficient arrays.
cdef dict fallback_cache = {}

#----------------------------------------------------------------------
# Initialization function for static coefficient arrays.
#----------------------------------------------------------------------
cdef void init_static_coeffs():
    global _static_coeffs_initialized
    if _static_coeffs_initialized:
        return
    # Initialize coefficients for window size 5.
    coeffs_5[0] = -3.0/35.0
    coeffs_5[1] = 12.0/35.0
    coeffs_5[2] = 17.0/35.0
    coeffs_5[3] = 12.0/35.0
    coeffs_5[4] = -3.0/35.0

    # Initialize coefficients for window size 7.
    coeffs_7[0] = -2.0/21.0    # j = -3
    coeffs_7[1] =  1.0/7.0     # j = -2
    coeffs_7[2] =  2.0/7.0     # j = -1
    coeffs_7[3] =  1.0/3.0     # j =  0
    coeffs_7[4] =  2.0/7.0     # j =  1
    coeffs_7[5] =  1.0/7.0     # j =  2
    coeffs_7[6] = -2.0/21.0    # j =  3

    _static_coeffs_initialized = True

#----------------------------------------------------------------------
# Compute the two constants Q0 and Q2 for poly_order=2.
# For a given odd window_size, the convolution coefficients are:
#   c_j = Q0 + Q2 * j^2,  for j = -m,...,m  where m = (window_size-1)//2.
#----------------------------------------------------------------------
cdef inline void compute_poly2_coeffs(int window_size, double *Q0, double *Q2):
    cdef int m = (window_size - 1) // 2
    cdef double S0, S2, S4, D, m2_1
    # S0 = sum_{j=-m}^{m} 1 = 2*m+1
    S0 = window_size
    m2_1 = m * (m + 1)
    # S2 = sum_{j=-m}^{m} j^2 = m*(m+1)*(2*m+1)/3
    S2 = m2_1 * (2*m + 1) / 3.0
    # S4 = sum_{j=-m}^{m} j^4 = m*(m+1)*(2*m+1)*(3*m*m+3*m-1)/15
    S4 = (S2 * (3*m2_1 - 1)) / 5.0
    # Denominator: D = S2^2 - S0 * S4.
    D = S2 * S2 - S0 * S4
    Q0[0] = -S4 / D
    Q2[0] = S2 / D

#----------------------------------------------------------------------
# Helper function to retrieve a pointer to the precomputed coefficients.
# If window_size is one of the recommended values (5 or 7), the corresponding
# static array is returned. Otherwise, the coefficients are computed on the fly,
# cached, and a pointer to the cached array is returned.
#----------------------------------------------------------------------
cdef inline double* get_precomputed_coeffs(int window_size):
    cdef int m, j
    cdef double Q0, Q2
    cdef np.ndarray[double, ndim=1] coeff_arr, arr
    cdef double[::1] coeffs_view

    if window_size == 5:
        init_static_coeffs()
        return &coeffs_5[0]
    elif window_size == 7:
        init_static_coeffs()
        return &coeffs_7[0]
    else:
        if window_size in fallback_cache:
            arr = fallback_cache[window_size]
            return <double*> arr.data
        else:
            coeff_arr = np.empty(window_size, dtype=np.float64)
            coeffs_view = coeff_arr
            m = (window_size - 1) // 2
            compute_poly2_coeffs(window_size, &Q0, &Q2)
            for j in range(-m, m+1):
                coeffs_view[j + m] = Q0 + Q2 * (j * j)
            fallback_cache[window_size] = coeff_arr
            return <double*> coeff_arr.data

#----------------------------------------------------------------------
# The Savitzky-Golay filter (poly_order=2) implementation.
# This version precomputes coefficients and uses pointer arithmetic and
# splits the convolution into regions for speed.
#----------------------------------------------------------------------
cdef  double[::1] savgol_filter_poly2(double[::1] s, int window_size):
    """
    Apply a Savitzky-Golay filter (poly_order=2) to the 1D memoryview 's'
    using precomputed coefficients if available.
    
    Parameters:
      s : 1D memoryview of doubles.
      window_size : odd integer specifying the filter window size.
    
    Returns:
      A new NumPy array containing the filtered signal.
    
    Notes:
      - Nearest-boundary extension is used.
    """
    cdef int n = s.shape[0]
    cdef int i, j, m, idx
    cdef double conv
    cdef np.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] y = out
    cdef double* coeffs

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    m = (window_size - 1) // 2

    # Retrieve coefficients (either from static arrays or computed on the fly)
    coeffs = get_precomputed_coeffs(window_size)

    # Optimize the convolution by splitting into regions.
    # Obtain a C pointer to the input array.
    cdef double *s_ptr = &s[0]
    cdef double *y_ptr = &y[0]

    # --- Bulk region: indices i from m to n-m-1, where the window is fully inside.
    for i in range(m, n - m):
        conv = 0.0
        for j in range(window_size):
            conv += s_ptr[i + j - m] * coeffs[j]
        y_ptr[i] = conv

    # --- Left boundary: i from 0 to m-1.
    for i in range(m):
        conv = 0.0
        for j in range(window_size):
            idx = i + j - m
            if idx < 0:
                idx = 0
            conv += s_ptr[idx] * coeffs[j]
        y_ptr[i] = conv

    # --- Right boundary: i from n-m to n-1.
    for i in range(n - m, n):
        conv = 0.0
        for j in range(window_size):
            idx = i + j - m
            if idx >= n:
                idx = n - 1
            conv += s_ptr[idx] * coeffs[j]
        y_ptr[i] = conv

    return y
