# find_peaks.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython

#----------------------------------------------------------------------
# cdef function to find initial regions.
# Returns a Python list of (start, end) tuples.
#----------------------------------------------------------------------
cdef list _find_initial_regions(double[::1] d2_signal,
                                double[::1] processed_signal,
                                double low_cutoff,
                                double d2_sigma,
                                int start_idx,
                                int end_idx,
                                Py_ssize_t linear_limit):
    cdef int i = start_idx
    cdef int n = end_idx
    cdef int start, end
    cdef list regions = []
    while i < n:
        if d2_signal[i] < low_cutoff * d2_sigma or processed_signal[i] > linear_limit:
            start = i
            while i < n and (d2_signal[i] < 0 or processed_signal[i] > linear_limit):
                i += 1
            regions.append((start, i - 1))
        i += 1
    return regions

#----------------------------------------------------------------------
# cdef function to expand regions.
# Returns a Python list of (start, end) tuples.
#----------------------------------------------------------------------
cdef list _expand_regions(list regions,
                          double[::1] d1_signal,
                          double[::1] processed_signal,
                          double signal_noise,
                          int raw_len):
    cdef list expanded = []
    cdef int start, end
    for (start, end) in regions:
        while start > 0 and d1_signal[start] > 0 and processed_signal[start] > signal_noise:
            start -= 1
        while end < raw_len and d1_signal[end] < 0 and processed_signal[end] > signal_noise:
            end += 1
        # Make end inclusive.
        expanded.append((start, end - 1))
    return expanded

#----------------------------------------------------------------------
# cdef function to filter regions by minimum length.
#----------------------------------------------------------------------
cdef list _filter_regions(list regions, int min_length):
    cdef list filtered = []
    cdef int start, end
    for (start, end) in regions:
        if end - start > min_length:
            filtered.append((start, end))
    return filtered

#----------------------------------------------------------------------
# The main find_peaks function that accepts memoryviews and scalars.
#----------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def find_peaks(double[::1] d2_signal,
               double[::1] processed_signal,
               double[::1] d1_signal,
               double[::1] raw_signal,
               double noise_threshold_multiplier,
               double d2_ave_noise,
               double signal_noise,
               double peak_limit,
               Py_ssize_t linear_limit):
    """
    Find peaks using a three-step process:
      1. Find contiguous regions where d2_signal falls below a threshold or processed_signal is high.
      2. Expand these regions based on d1_signal and processed_signal.
      3. Filter out regions that are too short.
    
    Parameters:
      d2_signal : 1D memoryview of doubles.
      processed_signal : 1D memoryview of doubles.
      d1_signal : 1D memoryview of doubles.
      raw_signal : 1D memoryview of doubles.
      noise_threshold_multiplier : double multiplier for d2 noise threshold.
      d2_ave_noise : double, average noise in d2.
      signal_noise : double, noise level in signal.
      peak_limit : double, used for a low cutoff.
    
    Returns:
      A list of (start, end) index tuples representing detected regions.
    """
    cdef double d2_sigma = noise_threshold_multiplier * d2_ave_noise
    # signal_sigma is computed but not used in the algorithm.
    cdef double low_cutoff = -peak_limit
    cdef int n = d2_signal.shape[0]
    cdef int raw_len = raw_signal.shape[0]
    
    cdef list initial_regions, expanded_regions, final_regions, tmp_regions

    # Step 1: Find initial regions.
    initial_regions = _find_initial_regions(d2_signal, processed_signal, low_cutoff, d2_sigma, 3, n - 3, linear_limit)
    
    # Remove regions that are too short (length <= 5)
    tmp_regions = []
    for (start, end) in initial_regions:
        if end - start > 5:
            tmp_regions.append((start, end))
    initial_regions = tmp_regions

    # Step 2: Expand regions.
    expanded_regions = _expand_regions(initial_regions, d1_signal, processed_signal, signal_noise, raw_len)
    
    # Step 3: Filter out regions that are too short (length <= 15)
    final_regions = _filter_regions(expanded_regions, 15)
    
    return final_regions
