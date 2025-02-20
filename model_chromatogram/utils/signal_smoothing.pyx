# signal_smoothing_optimized_opt.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3, infer_types=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, floor
from cython.parallel import prange   # Uncomment if you want to use parallel loops

cdef double coeffs_5[5]
cdef double coeffs_7[7]
cdef bint static_coeffs_initialized = False

cdef inline void init_static_coeffs() noexcept nogil:
    global static_coeffs_initialized, coeffs_5, coeffs_7
    if static_coeffs_initialized:
        return
    # Window size 5 coefficients.
    coeffs_5[0] = -0.08571428571428572
    coeffs_5[1] = 0.34285714285714286
    coeffs_5[2] = 0.4857142857142857
    coeffs_5[3] = 0.34285714285714286
    coeffs_5[4] = -0.08571428571428572
    # Window size 7 coefficients.
    coeffs_7[0] = -0.09523809523809523
    coeffs_7[1] =  0.14285714285714285
    coeffs_7[2] =  0.2857142857142857
    coeffs_7[3] =  0.3333333333333333
    coeffs_7[4] =  0.2857142857142857
    coeffs_7[5] =  0.14285714285714285
    coeffs_7[6] = -0.09523809523809523
    static_coeffs_initialized = True

#-----------------------------------------------------------
# C-level Savitzky–Golay filter (order 2) using pointers and malloc.
#-----------------------------------------------------------
cdef inline void compute_poly2_coeffs_opt(
    int window_size, 
    double* coeffs
    ) noexcept nogil:
    """
    Compute and store the coefficients for an order‑2 Savitzky–Golay filter.
    """
    cdef int m = (window_size - 1) // 2
    cdef double S0 = window_size
    cdef double m2_1 = m * (m + 1)
    cdef double S2 = m2_1 * (2*m + 1) / 3.0
    cdef double S4 = (S2 * (3*m2_1 - 1)) / 5.0
    cdef double D = S2 * S2 - S0 * S4
    cdef double Q0 = -S4 / D
    cdef double Q2 = S2 / D
    cdef int j
    for j in range(-m, m+1):
        coeffs[j + m] = Q0 + Q2 * (j * j)

cdef void savgol_filter_poly2(
    double* s, 
    double* y, 
    int n, 
    int window_size
    ) noexcept nogil:
    """
    Pointer‐based Savitzky–Golay filtering (order 2).
    s[0..n-1] is the input signal; y[0..n-1] is the output.
    """
    cdef int m = (window_size - 1) // 2
    cdef int i, j, idx
    cdef double conv

    # Allocate coefficients array on the C heap.
    if window_size == 5:
        init_static_coeffs()
        coeffs = coeffs_5
    elif window_size == 7:
        init_static_coeffs()
        coeffs = coeffs_7
    else:
        # For other sizes, allocate and compute on the fly.
        coeffs = <double*> malloc(window_size * sizeof(double))
        if coeffs == NULL:
            return
        compute_poly2_coeffs_opt(window_size, coeffs)

    # Bulk region.
    for i in range(m, n - m):
        conv = 0.0
        for j in range(window_size):
            conv += s[i + j - m] * coeffs[j]
        y[i] = conv

    # Left boundary.
    for i in range(m):
        conv = 0.0
        for j in range(window_size):
            idx = i + j - m
            if idx < 0:
                idx = 0
            conv += s[idx] * coeffs[j]
        y[i] = conv

    # Right boundary.
    for i in range(n - m, n):
        conv = 0.0
        for j in range(window_size):
            idx = i + j - m
            if idx >= n:
                idx = n - 1
            conv += s[idx] * coeffs[j]
        y[i] = conv

    if window_size != 5 and window_size != 7:
        free(coeffs)

#-----------------------------------------------------------
# C-level uniform_filter_1d using pointers.
#-----------------------------------------------------------
cdef inline void uniform_filter_1d_opt(
    double* in_arr, 
    double* out_arr, 
    int n, 
    int size
    ) noexcept nogil:
    """
    1D uniform filter using pointers.

    in_arr: input signal pointer
    out_arr: output signal pointer
    n: length of the input signal
    size: window size
    """
    cdef int half = size // 2
    cdef int i, j, start, end
    cdef double sum_val
    cdef double window_size = size
    for i in range(n):
        start = i - half
        if start < 0:
            start = 0
            window_size = end - 1
        end = i + half
        if end >= n:
            end = n - 1
            window_size = end - start + 1
        sum_val = 0.0
        for j in range(start, end + 1):
            sum_val += in_arr[j]
        out_arr[i] = sum_val / window_size

#-----------------------------------------------------------
# C-level computation of local variance.
#-----------------------------------------------------------
cdef inline void compute_local_variance_opt(
    double* signal, 
    double* local_mean, 
    double* var_arr, 
    int n
    ) noexcept nogil:

    cdef int i
    for i in range(n):
        var_arr[i] = (signal[i] - local_mean[i]) * (signal[i] - local_mean[i])

#-----------------------------------------------------------
# The heavy‐lifting core function.
#
# This function allocates and computes the final smoothed signal into a C
# array (using malloc) and returns an error code (0 if OK). It is declared
# nogil and uses only C pointers and loops.
#
# out_signal: on success, *out_signal will point to an array of n doubles,
# which must later be copied to a NumPy array and freed.
#-----------------------------------------------------------
cdef int _signal_smoothing_core(
    double* signal, 
    int n,                            
    int bg_min, 
    int bg_max,
    int min_window, 
    int max_window,
    int var_window_size,
    double k, 
    double** out_signal
    ) noexcept nogil:

    cdef int i, ws, idx, cur_ws
    cdef double val, temp, prev_size, diff_window, rms_noise, factor
    cdef int n_filters = ((max_window - min_window) // 2) + 1

    # Allocate temporary C arrays.
    cdef double* local_mean = <double*> malloc(n * sizeof(double))
    cdef double* var_arr    = <double*> malloc(n * sizeof(double))
    cdef double* local_var  = <double*> malloc(n * sizeof(double))
    cdef double* window_sizes = <double*> malloc(n * sizeof(double))
    cdef double* smoothed_signal = <double*> malloc(n * sizeof(double))
    if (local_mean == NULL or var_arr == NULL or local_var == NULL or
        window_sizes == NULL or smoothed_signal == NULL):
        # Free any allocated memory.
        if (local_mean): free(local_mean)
        if (var_arr): free(var_arr)
        if (local_var): free(local_var)
        if (window_sizes): free(window_sizes)
        if (smoothed_signal): free(smoothed_signal)
        return 1  # error code
    # Compute local mean.
    uniform_filter_1d_opt(signal, local_mean, n, var_window_size)
    # Compute local variance then smooth it.
    compute_local_variance_opt(signal, local_mean, var_arr, n)
    uniform_filter_1d_opt(var_arr, local_var, n, var_window_size)

    # Compute rms noise from background region.
    rms_noise = 0.0
    for i in range(bg_min, bg_max+1):
        rms_noise += local_var[i]
    rms_noise /= (bg_max - bg_min + 1)

    factor = k * k * rms_noise
    diff_window = (max_window - min_window) / factor

    # Compute adaptive window sizes.
    val = max_window - diff_window * local_var[0]
    if val < min_window:
        val = min_window
    elif val > max_window:
        val = max_window
    cur_ws = <int>(floor(val + 0.5))
    if cur_ws % 2 == 0:
        cur_ws += 1
    window_sizes[0] = cur_ws
    prev_size = window_sizes[0]
    for i in range(1, n):
        val = max_window - diff_window * local_var[i]
        if val < min_window:
            val = min_window
        elif val > max_window:
            val = max_window
        if val > prev_size + 0.5:
            val = prev_size + 1.0
            if val > max_window:
                val = max_window
        elif val < prev_size - 0.5:
            val = prev_size - 1.0
            if val < min_window:
                val = min_window
        prev_size = val
        temp = floor(val + 0.5)
        if (<int>temp) % 2 == 0:
            temp += 1
        window_sizes[i] = temp

    # Precompute filtered signals for each distinct window size.
    cdef double** filter_signals = <double**> malloc(n_filters * sizeof(double*))
    if filter_signals == NULL:
        free(local_mean); 
        free(var_arr); 
        free(local_var); 
        free(window_sizes); 
        free(smoothed_signal)
        return 2
    for idx in prange(n_filters, nogil=True):
        ws = min_window + 2 * idx
        filter_signals[idx] = <double*> malloc(n * sizeof(double))
        if filter_signals[idx] == NULL:
            for i in range(idx):
                free(filter_signals[i])
            free(filter_signals)
            free(local_mean); 
            free(var_arr); 
            free(local_var); 
            free(window_sizes); 
            free(smoothed_signal)
            return 3
        savgol_filter_poly2(signal, filter_signals[idx], n, ws)

    # Build the final smoothed signal.
    for i in range(n):
        ws = <int>window_sizes[i]
        idx = (ws - min_window) // 2
        smoothed_signal[i] = filter_signals[idx][i]

    # Final smoothing pass with fixed window = 5.
    cdef double* final_signal = <double*> malloc(n * sizeof(double))
    if final_signal == NULL:
        for idx in range(n_filters):
            free(filter_signals[idx])
        free(filter_signals)
        free(local_mean); free(var_arr); free(local_var); free(window_sizes); free(smoothed_signal)
        return 4
    savgol_filter_poly2(smoothed_signal, final_signal, n, 5)

    # Free temporary arrays.
    free(smoothed_signal)
    for idx in range(n_filters):
        free(filter_signals[idx])
    free(filter_signals)
    free(local_mean)
    free(var_arr)
    free(local_var)
    free(window_sizes)

    # Return the final signal through out_signal.
    out_signal[0] = final_signal

    return 0

#-----------------------------------------------------------
# Python-visible wrapper.
#
# This function runs with the GIL and calls the nogil core to get a C array.
# Then it copies the C data into a NumPy array, frees the C memory, and returns the result.
#-----------------------------------------------------------
def signal_smoothing(np.ndarray signal,
                     int bg_min,
                     int bg_max,
                     int min_window=5,
                     int max_window=41,
                     int var_window_size=31,
                     double k=7.5):
    """
    Adaptive signal smoothing using an optimized adaptive Savitzky–Golay filter.
    
    Parameters:
      signal : 1D NumPy array of doubles.
      bg_min, bg_max : indices defining the background region for noise estimation.
      min_window, max_window : allowed window sizes (odd integers).
      var_window_size : window size for computing local variance.
      k : scaling factor.
    
    Returns:
      A new NumPy array containing the smoothed signal.
    """
    # Ensure the input is a contiguous double array.
    cdef np.ndarray[np.float64_t, ndim=1] sig = np.ascontiguousarray(signal, dtype=np.float64)
    cdef int n = sig.shape[0]
    cdef double* s_ptr = <double*> sig.data

    cdef double* final_signal = NULL
    cdef int err
    # Call the core computation (releasing the GIL during heavy computation).
    err = _signal_smoothing_core(s_ptr, n, bg_min, bg_max,
                                 min_window, max_window,
                                 var_window_size, k,
                                 &final_signal)
    if err != 0 or final_signal == NULL:
        raise MemoryError("Error during signal smoothing computation.")

    # Create a new NumPy array and copy the data.
    cdef np.ndarray[np.float64_t, ndim=1] out_arr = np.empty(n, dtype=np.float64)
    cdef int i
    cdef double* out_ptr = <double*> out_arr.data
    for i in range(n):
        out_ptr[i] = final_signal[i]

    free(final_signal)
    return out_arr
