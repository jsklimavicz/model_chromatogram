# utils.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3, noexceptioncheck=True


cdef Py_ssize_t binary_search(double[:] cumsum, Py_ssize_t i, double threshold):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = i  # search in indices [0, i]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = (lo + hi) // 2
        if cumsum[mid] <= threshold:
            lo = mid + 1
        else:
            hi = mid
    return lo

cdef Py_ssize_t binary_search_ptr(double* cumsum, Py_ssize_t length, double threshold) except * nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = length  # search indices [0, length)
    cdef Py_ssize_t mid
    while lo < hi:
        mid = (lo + hi) // 2
        if cumsum[mid] <= threshold:
            lo = mid + 1
        else:
            hi = mid
    return lo
