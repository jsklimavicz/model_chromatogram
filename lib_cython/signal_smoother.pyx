import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound
from scipy.signal import savgol_filter

@boundscheck(False)
@wraparound(False)
def dynamic_smoothing(np.ndarray[np.float64_t, ndim=1] signal, 
                      int min_window=3, int max_window=52, 
                      int bg_min=0, int bg_max=100):
    # Define the range of window sizes (must be odd)
    cdef np.ndarray[np.int64_t, ndim=1] window_sizes = np.arange(min_window, max_window, 2, dtype=np.int64)
    cdef int n_points = signal.shape[0]

    # Initialize smoothed_arrays as a 2D array
    cdef np.ndarray[np.float64_t, ndim=2] smoothed_arrays = np.empty((len(window_sizes), n_points), dtype=np.float64)

    # Smooth the signal with different window sizes
    cdef int i
    for i in range(len(window_sizes)):
        smoothed_arrays[i, :] = savgol_filter(signal, window_length=window_sizes[i], polyorder=2, mode="constant")

    # Calculate noise
    cdef np.float64_t noise = np.sqrt(np.mean(signal[bg_min:bg_max] ** 2))

    # Return the result from smooth_signal
    return smooth_signal(signal, window_sizes, smoothed_arrays, n_points, noise)

@boundscheck(False)
@wraparound(False)
def smooth_signal(np.ndarray[np.float64_t, ndim=1] signal, 
                  np.ndarray[np.int64_t, ndim=1] window_sizes, 
                  np.ndarray[np.float64_t, ndim=2] smoothed_arrays, 
                  int n_points, 
                  np.float64_t noise):
    cdef int prev_window = window_sizes.shape[0] - 1
    cdef np.float64_t k = 10.0 * noise

    # Initialize the new_signal array
    cdef np.ndarray[np.float64_t, ndim=1] new_signal = np.zeros_like(signal)

    cdef int max_jump = 1
    cdef int i, min_window, max_window, pad_size, min_pad, max_pad, best_window_idx
    cdef np.ndarray[np.float64_t, ndim=2] curr_windows
    cdef np.ndarray[np.float64_t, ndim=1] vars, mult, means_before, means_after, sig_diff, local_variances

    for i in range(n_points):
        min_window = max(0, prev_window - max_jump)
        max_window = min(smoothed_arrays.shape[0] - 1, prev_window + max_jump + 1)

        pad_size = window_sizes[-1] // 2
        min_pad = max(0, i - pad_size)
        max_pad = min(i + pad_size, n_points - 1)

        # Use memory views for slicing smoothed_arrays
        curr_windows = smoothed_arrays[min_window:max_window + 1, min_pad:max_pad + 1]

        vars = np.var(curr_windows, axis=1)
        mult = np.array([j for j in window_sizes[min_window:max_window + 1]], dtype=np.float64)

        if pad_size < i < n_points - 4:
            means_before = np.mean(curr_windows[:, :pad_size + 1], axis=1)
            means_after = np.mean(curr_windows[:, pad_size:], axis=1)
            sig_diff = (
                2.0 * np.mean(smoothed_arrays[min_window:max_window + 1, i - 3:i + 4] - signal[i - 3:i + 4], axis=1)
            ) / noise
            local_variances = (
                vars / np.sqrt(mult)
                * (1 + np.power(k - means_before, 2))
                * (1 + np.power(k - means_after, 2))
                * (1 + np.power(sig_diff, 2))
            )
        else:
            means = np.mean(curr_windows, axis=1)
            local_variances = vars * (1 + (k - means)) / mult

        best_window_idx = max(0, np.argmin(local_variances) + prev_window - max_jump)
        new_signal[i] = smoothed_arrays[best_window_idx, i]
        prev_window = best_window_idx

    return new_signal
