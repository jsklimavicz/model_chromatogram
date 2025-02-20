# compound_calculations.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3, noexceptioncheck=True

################################################################
# Calculates compound retention time and related parameters for compounds 
# based on compound and column properties.
#
# Written by James Klimavicz 2025
################################################################

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log, exp, fabs, pow
from libc.stdlib cimport malloc, free
from .utils cimport binary_search_ptr   # assume binary_search is implemented in pure C, declared with nogil

###############################################################################
# Pure C function: _calculate_logD_inner
###############################################################################
cdef void _calculate_logD_inner(double pH_value,
                                double* pka, Py_ssize_t npka,
                                double* pkb, Py_ssize_t npkb,
                                double intrinsic_log_p,
                                double* out_avg_charge, 
                                double* out_broadening, 
                                double* out_logD) noexcept nogil:
    cdef int n_groups = npka + npkb
    cdef Py_ssize_t total_states = 1 << n_groups  # 2^(n_groups)
    cdef Py_ssize_t s, i
    cdef double state_prob, weight
    cdef int state_charge, bit, state_value
    cdef double logD_sum = 0.0, avg_charge = 0.0, sum_sq = 0.0
    # Precompute constant: log10 conversion factor.
    cdef double log10_const = log(10.0)
    # If n_groups is known to be small, you might allocate fixed arrays on the stack.
    cdef double* all_pka = <double*> malloc(n_groups * sizeof(double))
    cdef double* frac = <double*> malloc(n_groups * sizeof(double))
    if not all_pka or not frac:
        if all_pka:
            free(all_pka)
        if frac:
            free(frac)
        out_avg_charge[0] = 0.0
        out_logD[0] = intrinsic_log_p
        out_broadening[0] = 0.0
        return
    # Copy pka and pkb into one contiguous array.
    for i in range(npka):
        all_pka[i] = pka[i]
    for i in range(npkb):
        all_pka[npka + i] = pkb[i]
    # Compute fractions; instead of pow(10, x) we compute exp(x*log(10))
    for i in range(n_groups):
        frac[i] = 1.0 / (1.0 + exp((pH_value - all_pka[i]) * log10_const))
    # Enumerate states.
    for s in range(total_states):
        state_prob = 1.0
        state_charge = 0
        for i in range(n_groups):
            bit = (s >> i) & 1
            if i < npka:
                if bit == 0:
                    state_prob *= frac[i]
                    state_value = 0
                else:
                    state_prob *= (1.0 - frac[i])
                    state_value = -1
            else:
                if bit == 0:
                    state_prob *= frac[i]
                    state_value = 1
                else:
                    state_prob *= (1.0 - frac[i])
                    state_value = 0
            state_charge += state_value
        if state_charge < 0:
            weight = - (state_charge * state_charge) * 0.8 + intrinsic_log_p
        else:
            weight = - (state_charge * state_charge) * 0.75 + intrinsic_log_p
        logD_sum += state_prob * weight
        avg_charge += state_prob * state_charge
        sum_sq += state_prob * state_prob
    free(all_pka)
    free(frac)
    out_avg_charge[0] = avg_charge
    out_logD[0] = logD_sum
    out_broadening[0] = 1.0 / sqrt(sum_sq)

###############################################################################
# Pure C function: _apply_temperature_correction_nogil
###############################################################################
cdef void _apply_temperature_correction_nogil(double* Rf_in,
                                              double* t, Py_ssize_t n,
                                              double* Rf_out) noexcept nogil:
    cdef Py_ssize_t i
    cdef double dt_temp
    for i in range(n):
        dt_temp = 10.0 * (1.0 / t[i] - 1.0 / 298.0)
        Rf_out[i] = Rf_in[i] * exp(dt_temp)

###############################################################################
# Python wrapper for calculate_logD.
###############################################################################
cpdef tuple calculate_logD(double pH_value,
                           double[::1] pka_list,
                           double[::1] pkb_list,
                           double intrinsic_log_p):
    cdef Py_ssize_t npka = pka_list.shape[0]
    cdef Py_ssize_t npkb = pkb_list.shape[0]
    cdef double avg_charge, broadening, logD_val
    cdef double* pka_ptr = &pka_list[0]
    cdef double* pkb_ptr = &pkb_list[0]
    with nogil:
         _calculate_logD_inner(pH_value, pka_ptr, npka,
                               pkb_ptr, npkb, intrinsic_log_p,
                               &avg_charge, &broadening, &logD_val)
    return avg_charge, broadening, logD_val

###############################################################################
# Pure C function: _find_retention_factor_nogil
###############################################################################
cdef void _find_retention_factor_nogil(double* Rf_out,
                                       double* hb_acidity,
                                       double* hb_basicity,
                                       double* polarity,
                                       double* dielectric,
                                       int n,
                                       double solvent_ph,
                                       double mw,
                                       double tpsa,
                                       double logD,
                                       double average_charge,
                                       double h_acceptors,
                                       double h_donors,
                                       double col_a,
                                       double col_b,
                                       double col_c7,
                                       double col_c28,
                                       double col_eb,
                                       double col_h,
                                       double col_s_star) noexcept nogil:
    cdef int i
    cdef double vol_ratio = pow(mw, 1.0/3.0)
    cdef double ratio_tpsa = sqrt(tpsa) / vol_ratio
    cdef double curr_c = col_c28 + (col_c7 - col_c28) / 4.2 * (solvent_ph - 2.8)
    cdef double base = log(col_eb)
    cdef double ha, hb_val, pol, diel, temp
    cdef double sqrt_ha_cola = sqrt(h_acceptors) * col_a
    cdef double sqrt_hd_colb = sqrt(h_donors) * col_b
    cdef double logd_colh = - logD * col_h
    cdef double curr_c_ratio_tpsa = curr_c * ratio_tpsa
    cdef double col_s_star_div_10 = col_s_star / 10.0

    for i in range(n):
        Rf_out[i] = base
        ha = hb_acidity[i]
        hb_val = hb_basicity[i]
        pol = polarity[i]
        diel = dielectric[i]
        temp = sqrt_ha_cola / (1.0 + vol_ratio * ha)
        temp += sqrt_hd_colb / (1.0 + vol_ratio * hb_val)
        temp += logd_colh / (10.0 + pol)
        temp += curr_c_ratio_tpsa / (1.0 + diel / 10.0)
        temp += col_s_star_div_10
        Rf_out[i] += - 4.0 * temp
        Rf_out[i] = exp(Rf_out[i]) + 1.0

###############################################################################
# Python wrapper for find_retention_factor.
###############################################################################
cdef double[::1] find_retention_factor(double[::1] hb_acidity,
                                       double[::1] hb_basicity,
                                       double[::1] polarity,
                                       double[::1] dielectric,
                                       double solvent_ph,
                                       double mw,
                                       double tpsa,
                                       double logD,
                                       double average_charge,
                                       double h_acceptors,
                                       double h_donors,
                                       double column_a,
                                       double column_b,
                                       double column_c7,
                                       double column_c28,
                                       double column_eb,
                                       double column_h,
                                       double column_s_star):
    cdef int n = hb_acidity.shape[0]
    cdef np.ndarray[double, ndim=1] Rf_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] Rf_mv = Rf_arr
    cdef double* Rf_ptr = &Rf_mv[0]
    cdef double* ha_ptr = &hb_acidity[0]
    cdef double* hb_ptr = &hb_basicity[0]
    cdef double* pol_ptr = &polarity[0]
    cdef double* diel_ptr = &dielectric[0]
    with nogil:
        _find_retention_factor_nogil(Rf_ptr,
            ha_ptr, hb_ptr, pol_ptr, diel_ptr,
            n, solvent_ph, mw, tpsa, logD, average_charge,
            h_acceptors, h_donors,
            column_a, column_b, column_c7, column_c28, column_eb, column_h, column_s_star)
    return Rf_mv

###############################################################################
# cpdef function: set_retention_time
###############################################################################
cpdef tuple set_retention_time(double[::1] time,
                               double[::1] flow,
                               double[::1] hb_acidity,
                               double[::1] hb_basicity,
                               double[::1] polarity,
                               double[::1] dielectric,
                               double[::1] temperature,
                               double solvent_ph,
                               object column,
                               double mw, 
                               double tpsa,
                               double[::1] pka_list, 
                               double[::1] pkb_list,
                               double intrinsic_log_p,
                               double h_acceptors, 
                               double h_donors):
    cdef Py_ssize_t n = temperature.shape[0], i
    cdef double avg_charge, broadening_factor, logD_val, retention_time, v0, v1, t0, t1, dt_temp
    # Compute logD and average charge.
    avg_charge, broadening_factor, logD_val = calculate_logD(7.0, pka_list, pkb_list, intrinsic_log_p)
    
    # Extract column parameters (with the GIL).
    cdef double col_a      = column.parameters.a
    cdef double col_b      = column.parameters.b
    cdef double col_c7     = column.parameters.c7
    cdef double col_c28    = column.parameters.c28
    cdef double col_eb     = column.parameters.eb
    cdef double col_h      = column.parameters.h
    cdef double col_s_star = column.parameters.s_star
    cdef double col_volume = column.volume

    # Get the retention factor (Rf) as a memoryview.
    cdef double[::1] Rf = find_retention_factor(hb_acidity, hb_basicity, polarity, dielectric,
                                                 solvent_ph, mw, tpsa,
                                                 logD_val, avg_charge, h_acceptors, h_donors,
                                                 col_a, col_b, col_c7, col_c28, col_eb, col_h, col_s_star)
    
    # Preallocate array for temperature-corrected Rf.
    cdef np.ndarray[double, ndim=1] Rf_corr_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] Rf_corr_mv = Rf_corr_arr
    cdef double* Rf_corr_ptr = &Rf_corr_mv[0]
    cdef double* temp_ptr = &temperature[0]
    cdef double* Rf_ptr = &Rf[0]
    with nogil:
        _apply_temperature_correction_nogil(Rf_ptr, temp_ptr, n, Rf_corr_ptr)
    
    # Preallocate and obtain a memoryview for move_ratio.
    cdef np.ndarray[double, ndim=1] move_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] move_mv = move_arr
    cdef double* move_ptr = &move_mv[0]
    cdef double* time_ptr = &time[0]
    cdef double* flow_ptr = &flow[0]
    move_ptr[0] = -1.0
    cdef double prev = -1.0
    dt_temp = time_ptr[1] - time_ptr[0]
    for i in range(1, n):
        move_ptr[i] = prev + (flow_ptr[i] / (Rf_corr_ptr[i] * col_volume)) * dt_temp
        prev = move_ptr[i]
    
    # Call binary_search. (Assume binary_search_ptr accepts a pointer to double and returns an int.)
    cdef int last_neg_ind = binary_search_ptr(move_ptr, n//2, 1.0)
    
    if last_neg_ind < n - 1:
        v0 = move_ptr[last_neg_ind]
        v1 = move_ptr[last_neg_ind + 1]
        t0 = time_ptr[last_neg_ind]
        t1 = time_ptr[last_neg_ind + 1]
        if v1 != v0:
            retention_time = t0 + (-v0) * (t1 - t0) / (v1 - v0)
        else:
            retention_time = t0
    else:
        retention_time = time_ptr[n - 1] + 5.0

    return retention_time, avg_charge, broadening_factor, logD_val
