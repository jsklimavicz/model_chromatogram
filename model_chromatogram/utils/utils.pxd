# utils.pxd

cdef Py_ssize_t binary_search(double[:] cumsum, Py_ssize_t i, double threshold)
cdef Py_ssize_t binary_search_ptr(double* cumsum, Py_ssize_t length, double threshold) except * nogil