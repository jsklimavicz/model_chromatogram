# viscosity.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport exp, log

# ===========================================================================
# Module-level parameters: declared as static C arrays.
#
# Each parameter array has 19 elements with the ordering:
#   [ B, C, D, E, G, H, I, k0, k1, k2, k3, k_1, k_2, k_3, kp1, kt1, kp_1, kt_1, F ]
# ===========================================================================

cdef double MEOH_PARAMS_arr[19]
cdef double ACN_PARAMS_arr[19]
cdef double THF_PARAMS_arr[19]

# Initialize the arrays at module initialization.
# (This code is executed when the module is imported.)
MEOH_PARAMS_arr[0] = -4320.221828656787         # B
MEOH_PARAMS_arr[1] = -0.04317321464026093       # C
MEOH_PARAMS_arr[2] = -3.2183386517271124e-5     # D
MEOH_PARAMS_arr[3] = -4.7505432844620525e-5     # E
MEOH_PARAMS_arr[4] = 371.38798800124863         # G
MEOH_PARAMS_arr[5] = 0.0004538706442512652      # H
MEOH_PARAMS_arr[6] = 0.0007376663726093331      # I
MEOH_PARAMS_arr[7] = 12.090142949338516         # k0
MEOH_PARAMS_arr[8] = 546.0845649098856          # k1
MEOH_PARAMS_arr[9] = -8.857208509233104         # k2
MEOH_PARAMS_arr[10] = -82.38958460101334        # k3
MEOH_PARAMS_arr[11] = 60.96296363045879         # k_1
MEOH_PARAMS_arr[12] = -8.911526703699176        # k_2
MEOH_PARAMS_arr[13] = 50.12515565908149         # k_3
MEOH_PARAMS_arr[14] = -0.02186510631628006      # kp1
MEOH_PARAMS_arr[15] = -110239.53634455631       # kt1
MEOH_PARAMS_arr[16] = -0.0016034949067753946    # kp_1
MEOH_PARAMS_arr[17] = -15261.6589032647         # kt_1
MEOH_PARAMS_arr[18] = 626.7021731564391         # F

ACN_PARAMS_arr[0] = 1000.5224252994153          # B
ACN_PARAMS_arr[1] = 0.014999370610366661        # C
ACN_PARAMS_arr[2] = -2.553166508492207e-5       # D
ACN_PARAMS_arr[3] = -2.6172687530980885e-6      # E
ACN_PARAMS_arr[4] = -515.7737458861527          # G
ACN_PARAMS_arr[5] = -0.0077298065190134005      # H
ACN_PARAMS_arr[6] = -0.0004490251419335607      # I
ACN_PARAMS_arr[7] = 0.001530358383339248        # k0
ACN_PARAMS_arr[8] = 0.007019136108015722        # k1
ACN_PARAMS_arr[9] = -0.008560398753365471       # k2
ACN_PARAMS_arr[10] = 0.0055932598208976924      # k3
ACN_PARAMS_arr[11] = -2.573000096219173         # k_1
ACN_PARAMS_arr[12] = 2.5699730005382704         # k_2
ACN_PARAMS_arr[13] = -0.9417237250709205        # k_3
ACN_PARAMS_arr[14] = 2.52427453434938e-6        # kp1
ACN_PARAMS_arr[15] = -0.8166156207094488        # kt1
ACN_PARAMS_arr[16] = -2.0742521820349215e-5     # kp_1
ACN_PARAMS_arr[17] = 9.338993089732396          # kt_1
ACN_PARAMS_arr[18] = 57.90262079548248          # F

THF_PARAMS_arr[0] = 1.6563181482576743          # B
THF_PARAMS_arr[1] = 0.04474692598596613         # C
THF_PARAMS_arr[2] = -8.864950216771027e-5       # D
THF_PARAMS_arr[3] = 4.192451366876064e-6        # E 
THF_PARAMS_arr[4] = -1024.3441136864744         # G
THF_PARAMS_arr[5] = -0.0007084160629411966      # H
THF_PARAMS_arr[6] = 5.411585590552736e-5        # I
THF_PARAMS_arr[7] = 0.002836980044460593        # k0
THF_PARAMS_arr[8] = -0.3174381545313952         # k1
THF_PARAMS_arr[9] = 0.592778406913796           # k2
THF_PARAMS_arr[10] = 0.04923437614619724        # k3
THF_PARAMS_arr[11] = 20.50005804790994          # k_1
THF_PARAMS_arr[12] = 41.75880466355388          # k_2
THF_PARAMS_arr[13] = -39.91674068240163         # k_3
THF_PARAMS_arr[14] = -6.658731934939264e-9      # kp1
THF_PARAMS_arr[15] = 119.257853718558           # kt1
THF_PARAMS_arr[16] = -0.0003326722864187177     # kp_1
THF_PARAMS_arr[17] = -3637.946261739626         # kt_1
THF_PARAMS_arr[18] = 292.4245159089498          # F

# ===========================================================================
# The core model function.
#
# data is a 3 x N array where:
#   - row 0: temperature in Celsius (will be shifted to Kelvin)
#   - row 1: molar fraction (χ)
#   - row 2: pressure (p)
#
# model_type is a string that should be one of "meoh", "acn", or "thf"
# ===========================================================================
cpdef np.ndarray[double, ndim=1] model(np.ndarray[double, ndim=2] data, str model_type):
    cdef Py_ssize_t N = data.shape[1]
    cdef np.ndarray[double, ndim=1] result = np.empty(N, dtype=np.float64)
    cdef double[:] res = result

    # Get a 2D memoryview for the data array (assumed C-contiguous).
    cdef double[:, :] d = data
    # Cache pointers for each row:
    cdef double* temp_ptr = &d[0, 0]   # Temperature (Celsius)
    cdef double* chi_ptr  = &d[1, 0]   # Molar fraction (χ)
    cdef double* pres_ptr = &d[2, 0]   # Pressure (p)

    # Select the appropriate parameter set based on model_type.
    cdef double* pview
    if "meoh" in model_type.lower():
        pview = MEOH_PARAMS_arr
    elif "acn" in model_type.lower():
        pview = ACN_PARAMS_arr
    elif "thf" in model_type.lower():
        pview = THF_PARAMS_arr
    else:
        raise ValueError("Unknown model type: " + model_type)

    # Unpack parameters from the static array:
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
    cdef double tau, chi, pres, chi2, chi3, temp_exp, pres_exp, mix, comp, num, den, tau_inv

    for i in range(N):
        # Convert temperature from Celsius to Kelvin.
        tau = temp_ptr[i] + 273.15
        chi = chi_ptr[i]
        pres = pres_ptr[i]
        tau_inv = 1.0 / tau

        chi2 = chi * chi
        chi3 = chi2 * chi

        temp_exp = B / (tau - F) + C * tau + D * (tau * tau)
        pres_exp = E * pres
        mix = (G * tau_inv + H * tau + I * pres) * chi

        comp = exp(temp_exp + pres_exp + mix)
        num = k0 + (k1 + kt1 * tau_inv + kp1 * pres) * chi + k2 * chi2 + k3 * chi3
        den = 1.0 + (k_1 + kt_1 * tau_inv + kp_1 * pres) * chi + k_2 * chi2 + k_3 * chi3

        res[i] = comp * num / den
    return result


cpdef double model_scalar(double temp, double chi, double pres, str model_type):
    # Select the appropriate parameter set based on model_type.
    cdef double* pview
    if "meoh" in model_type.lower():
        pview = MEOH_PARAMS_arr
    elif "acn" in model_type.lower():
        pview = ACN_PARAMS_arr
    elif "thf" in model_type.lower():
        pview = THF_PARAMS_arr
    else:
        raise ValueError("Unknown model type: " + model_type)

    # Unpack parameters from the static array:
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

    cdef double tau, chi2, chi3, temp_exp, pres_exp, mix, comp, num, den, tau_inv


    # Convert temperature from Celsius to Kelvin.
    tau = temp + 273.15
    tau_inv = 1.0 / tau

    chi2 = chi * chi
    chi3 = chi2 * chi

    temp_exp = B / (tau - F) + C * tau + D * (tau * tau)
    pres_exp = E * pres
    mix = (G * tau_inv + H * tau + I * pres) * chi

    comp = exp(temp_exp + pres_exp + mix)
    num = k0 + (k1 + kt1 * tau_inv + kp1 * pres) * chi + k2 * chi2 + k3 * chi3
    den = 1.0 + (k_1 + kt_1 * tau_inv + kp_1 * pres) * chi + k_2 * chi2 + k_3 * chi3

    return comp * num / den


# ===========================================================================
# Viscosity for scalar inputs.
# ===========================================================================
cpdef double viscosity_scalar(str solvent_model, double T, double chi, double p):
    return model_scalar(T, chi, p, solvent_model)

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
    cdef double[:] result_view = model(data, solvent_model)
    return result_view
