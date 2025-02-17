# signal_smoothing.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, linetrace=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from scipy.ndimage import uniform_filter1d
from .savgol_poly2 cimport savgol_filter_poly2

def signal_smoothing(double[::1] signal,
                     int bg_min,
                     int bg_max,
                     int min_window=5,
                     int max_window=41,
                     int var_window_size=31,
                     double k=7.5):
    cdef int n = signal.shape[0]
    cdef int i, j, seg_start, seg_end, seg_count, cur_ws, ws, idx, n_filters
    cdef double prev_size, val, temp, factor, rms_noise, cur_val, diff_window
    cdef np.ndarray[double, ndim=1] window_sizes_arr, local_variance_arr, local_mean
    cdef double[::1] window_sizes, local_variance, smoothed_signal_mv

    # --- Initial smoothing and local variance calculation ---
    signal = savgol_filter_poly2(signal, 5)
    local_mean = uniform_filter1d(signal, size=var_window_size)
    local_variance_arr = uniform_filter1d((signal - local_mean)**2, size=var_window_size)
    local_variance = local_variance_arr
    rms_noise = 0.0
    for i in range(bg_min, bg_max+1):
        rms_noise += local_variance[i]
    rms_noise /= (bg_max - bg_min + 1)

    # --- Compute window sizes ---
    factor = k * k * rms_noise
    window_sizes_arr = np.empty(n, dtype=np.float64)
    window_sizes = window_sizes_arr
    diff_window = (max_window - min_window) / factor

    cur_val = max_window - diff_window * local_variance[0]
    if cur_val < min_window:
        cur_val = min_window
    elif cur_val > max_window:
        cur_val = max_window
    cur_ws = <int>round(cur_val)
    if cur_ws % 2 == 0:
        cur_ws += 1
    window_sizes[0] = cur_ws
    prev_size = window_sizes[0]

    for i in range(1, n):
        val = max_window - diff_window * local_variance[i]
        if val < min_window:
            val = min_window
        elif val > max_window:
            val = max_window
        if val > prev_size + 1.0:
            val = prev_size + 2.0
            if val > max_window:
                val = max_window
        elif val < prev_size - 1.0:
            val = prev_size - 2.0
            if val < min_window:
                val = min_window
        prev_size = val
        temp = round(val)
        if <int>temp % 2 == 0:
            temp += 1
        window_sizes[i] = temp

    cdef np.ndarray[np.int32_t, ndim=1] window_sizes_int = window_sizes_arr.astype(np.int32)
    cdef int[::1] win_sizes = window_sizes_int

    # --- Precompute filters using a C array instead of a dict ---
    n_filters = ((max_window - min_window) // 2) + 1
    # Allocate an object array of size n_filters.
    cdef object[:] filter_arr = np.empty(n_filters, dtype=object)
    for idx in range(n_filters):
        ws = min_window + 2 * idx
        filter_arr[idx] = savgol_filter_poly2(signal, ws)
    
    # --- Determine boundaries directly using the memoryview ---
    cdef np.ndarray[np.int32_t, ndim=1] boundaries_arr = np.empty(n+1, dtype=np.int32)
    cdef int[::1] boundaries = boundaries_arr  # memoryview
    seg_count = 0
    boundaries[seg_count] = 0
    seg_count += 1
    for i in range(1, n):
        if win_sizes[i] != win_sizes[i-1]:
            boundaries[seg_count] = i
            seg_count += 1
    boundaries[seg_count] = n

    # --- Build the final smoothed signal using a C loop over boundaries ---
    cdef np.ndarray[double, ndim=1] smoothed_signal = np.zeros(n, dtype=np.float64)
    smoothed_signal_mv = smoothed_signal  # memoryview
    cdef double[::1] temp_filtered_mv
    for i in range(seg_count - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i+1]
        ws = win_sizes[seg_start]
        idx = (ws - min_window) // 2
        # Look up filter from our C array
        temp_filtered_mv = filter_arr[idx]
        for j in range(seg_start, seg_end):
            smoothed_signal_mv[j] = temp_filtered_mv[j]

    # --- Final smoothing pass ---
    smoothed_signal_mv = savgol_filter_poly2(smoothed_signal_mv, 5)
    return smoothed_signal

