# viscosity.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c

import numpy as np
cimport numpy as np
from libc.math cimport exp, log

# ===========================================================================
# Module-level parameters: declared as plain Python objects.
# ===========================================================================
MEOH_PARAMS = np.array([
    -4320.221828656787, -0.04317321464026093, -3.2183386517271124e-5, -4.7505432844620525e-5,
    371.38798800124863, 0.0004538706442512652, 0.0007376663726093331, 12.090142949338516,
    546.0845649098856, -8.857208509233104, -82.38958460101334, 60.96296363045879,
    -8.911526703699176, 50.12515565908149, -0.02186510631628006, -110239.53634455631,
    -0.0016034949067753946, -15261.6589032647, 626.7021731564391
], dtype=np.float64)

ACN_PARAMS = np.array([
    1000.5224252994153, 0.014999370610366661, -2.553166508492207e-5, -2.6172687530980885e-6,
    -515.7737458861527, -0.0077298065190134005, -0.0004490251419335607, 0.001530358383339248,
    0.007019136108015722, -0.008560398753365471, 0.0055932598208976924, -2.573000096219173,
    2.5699730005382704, -0.9417237250709205, 2.52427453434938e-6, -0.8166156207094488,
    -2.0742521820349215e-5, 9.338993089732396, 57.90262079548248
], dtype=np.float64)

THF_PARAMS = np.array([
    1.6563181482576743, 0.04474692598596613, -8.864950216771027e-5, 4.192451366876064e-6,
    -1024.3441136864744, -0.0007084160629411966, 5.411585590552736e-5, 0.002836980044460593,
    -0.3174381545313952, 0.592778406913796, 0.04923437614619724, 20.50005804790994,
    41.75880466355388, -39.91674068240163, -6.658731934939264e-9, 119.257853718558,
    -0.0003326722864187177, -3637.946261739626, 292.4245159089498
], dtype=np.float64)

# ===========================================================================
# The core model function.
#
# data is a 3 x N array where:
#   - row 0: temperature in Celsius (will be shifted to Kelvin)
#   - row 1: molar fraction (χ)
#   - row 2: pressure (p)
#
# params is a parameter vector (one of the above) with the ordering:
#
#   [ B, C, D, E, G, H, I, k0, k1, k2, k3, k_1, k_2, k_3, kp1, kt1, kp_1, kt_1, F ]
#
# ===========================================================================
cpdef np.ndarray[double, ndim=1] model(np.ndarray[double, ndim=2] data, object params):
    cdef Py_ssize_t N = data.shape[1]
    cdef np.ndarray[double, ndim=1] result = np.empty(N, dtype=np.float64)
    cdef double[:] res = result

    # Get a typed memoryview for the input data.
    cdef double[:, :] d = data

    # Convert the parameter vector to a local array and memoryview.
    cdef np.ndarray[double, ndim=1] p_arr = params
    cdef double[:] pview = p_arr

    # Unpack parameters (using 0-indexing):
    cdef double B   = pview[0]
    cdef double C   = pview[1]
    cdef double D   = pview[2]
    cdef double E   = pview[3]
    cdef double G   = pview[4]
    cdef double H   = pview[5]
    cdef double I   = pview[6]
    cdef double k0  = pview[7]
    cdef double k1  = pview[8]
    cdef double k2  = pview[9]
    cdef double k3  = pview[10]
    cdef double k_1 = pview[11]
    cdef double k_2 = pview[12]
    cdef double k_3 = pview[13]
    cdef double kp1 = pview[14]
    cdef double kt1 = pview[15]
    cdef double kp_1 = pview[16]
    cdef double kt_1 = pview[17]
    cdef double F   = pview[18]

    cdef Py_ssize_t i
    cdef double tau, chi, pres, chi2, chi3, temp_exp, pres_exp, mix, comp, num, den

    for i in range(N):
        # Temperature: convert from Celsius to Kelvin.
        tau = d[0, i] + 273.15
        chi = d[1, i]
        pres = d[2, i]

        chi2 = chi * chi
        chi3 = chi2 * chi

        temp_exp = B / (tau - F) + C * tau + D * (tau * tau)
        pres_exp = E * pres
        mix = (G / tau + H * tau + I * pres) * chi

        comp = exp(temp_exp + pres_exp + mix)

        num = k0 + (k1 + kt1 / tau + kp1 * pres) * chi + k2 * chi2 + k3 * chi3
        den = 1.0 + (k_1 + kt_1 / tau + kp_1 * pres) * chi + k_2 * chi2 + k_3 * chi3

        res[i] = comp * num / den
    return result

# ===========================================================================
# Viscosity for scalar inputs.
# ===========================================================================
cpdef double viscosity_scalar(str solvent_model, double T, double chi, double p):
    cdef np.ndarray[double, ndim=2] data = np.empty((3, 1), dtype=np.float64)
    data[0, 0] = T
    data[1, 0] = chi
    data[2, 0] = p
    cdef np.ndarray[double, ndim=1] result
    if "meoh" in solvent_model.lower():
        result = model(data, MEOH_PARAMS)
    elif "acn" in solvent_model.lower():
        result = model(data, ACN_PARAMS)
    elif "thf" in solvent_model.lower():
        result = model(data, THF_PARAMS)
    else:
        raise ValueError("Unknown solvent model: " + solvent_model)
    return result[0]

# ===========================================================================
# Viscosity for vector inputs.
# T, chi, and p are 1D arrays of the same length.
# ===========================================================================
cpdef np.ndarray[double, ndim=1] viscosity_vector(str solvent_model,
                                                   np.ndarray[double, ndim=1] T,
                                                   np.ndarray[double, ndim=1] chi,
                                                   np.ndarray[double, ndim=1] p):
    cdef Py_ssize_t n = T.shape[0]
    cdef np.ndarray[double, ndim=2] data = np.empty((3, n), dtype=np.float64)
    data[0, :] = T
    data[1, :] = chi
    data[2, :] = p
    cdef np.ndarray[double, ndim=1] result
    if "meoh" in solvent_model.lower():
        result = model(data, MEOH_PARAMS)
    elif "acn" in solvent_model.lower():
        result = model(data, ACN_PARAMS)
    elif "thf" in solvent_model.lower():
        result = model(data, THF_PARAMS)
    else:
        raise ValueError("Unknown solvent model: " + solvent_model)
    return result
