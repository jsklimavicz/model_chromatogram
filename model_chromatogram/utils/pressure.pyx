# pressure.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c

import numpy as np
cimport numpy as np
import math
from libc.math cimport exp, log, fabs, pi
cimport cython
import pandas as pd

from .viscosity import viscosity_scalar

#############################
# Global solvent dictionaries
#############################
cdef dict METHANOL = {"density": 0.792, "mw": 32.04}
cdef dict WATER = {"density": 0.997, "mw": 18.01528}
cdef dict ACETONITRILE = {"density": 0.786, "mw": 41.05}
cdef dict THF = {"density": 0.886, "mw": 72.11}

#############################
# Column class (like Julia's struct)
#############################
cdef class Column:
    cdef public double length, particle_diameter, particle_sphericity, porosity, volume, inner_diameter
    def __init__(self, double length, double particle_diameter, double particle_sphericity,
                 double porosity, double volume, double inner_diameter):
        self.length = length
        self.particle_diameter = particle_diameter
        self.particle_sphericity = particle_sphericity
        self.porosity = porosity
        self.volume = volume
        self.inner_diameter = inner_diameter

########################################
# calculate_molar_fraction
########################################

def calculate_molar_fraction(str org_type, np.ndarray[double, ndim=1] org_percent):
    """
    Given a 1D NumPy array of organic percentages (org_percent) and an organic type,
    compute the molar fraction as:
        a = org_percent * density * mw
        b = (100 - org_percent) * WATER[density]*WATER[mw]
    and return a/(a+b) as a new NumPy array.
    """
    cdef double density, mw
    if org_type == "methanol":
        density = METHANOL["density"]
        mw = METHANOL["mw"]
    elif org_type == "acetonitrile":
        density = ACETONITRILE["density"]
        mw = ACETONITRILE["mw"]
    elif org_type == "thf":
        density = THF["density"]
        mw = THF["mw"]
    else:
        raise ValueError("Unknown organic type: " + org_type)
    
    cdef np.ndarray[double, ndim=1] a_arr = np.ascontiguousarray(org_percent * density * mw, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] b_arr = np.ascontiguousarray((100.0 - org_percent) * WATER["density"] * WATER["mw"], dtype=np.float64)
    cdef Py_ssize_t n = a_arr.shape[0]
    cdef np.ndarray[double, ndim=1] out_arr = np.empty(n, dtype=np.float64)
    cdef double[:] a = a_arr, b = b_arr, out = out_arr
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = a[i] / (a[i] + b[i])
    return out_arr

########################################
# update_solvent_profile!
########################################

def update_solvent_profile(object sp_df, Column column):
    """
    Update the solvent profile DataFrame (sp_df) in-place.
    Assumes sp_df has columns:
      "time", "flow", "percent_meoh", "percent_acn", "percent_thf", "temperature".
    Adds new columns:
      "incremental_CV", "incremental_length", "superficial_velocity",
      "meoh_x", "acn_x", "thf_x".
    """
    cdef np.ndarray[double, ndim=1] time_arr = np.ascontiguousarray(sp_df["time"].to_numpy(np.float64))
    cdef double[:] time_view = time_arr
    cdef double time_delta = time_view[1] - time_view[0]
    
    sp_df.loc[:, "incremental_CV"] = sp_df.loc[:, "flow"] * time_delta / column.volume
    sp_df.loc[:, "incremental_length"] = sp_df.loc[:, "incremental_CV"] * column.length
    sp_df.loc[:, "superficial_velocity"] = sp_df.loc[:, "flow"] / ((column.inner_diameter**2 * pi) / 400.0) / (100.0 * 60.0)
    
    cdef np.ndarray[double, ndim=1] perc_meoh = np.ascontiguousarray(sp_df.loc[:, "percent_meoh"].to_numpy(np.float64))
    cdef np.ndarray[double, ndim=1] perc_acn = np.ascontiguousarray(sp_df.loc[:, "percent_acn"].to_numpy(np.float64))
    cdef np.ndarray[double, ndim=1] perc_thf = np.ascontiguousarray(sp_df.loc[:, "percent_thf"].to_numpy(np.float64))
    
    sp_df.loc[:, "meoh_x"] = calculate_molar_fraction("methanol", perc_meoh)
    sp_df.loc[:, "acn_x"] = calculate_molar_fraction("acetonitrile", perc_acn)
    sp_df.loc[:, "thf_x"] = calculate_molar_fraction("thf", perc_thf)

########################################
# total_viscosity (vector version)
########################################

def total_viscosity_vector(np.ndarray[double, ndim=1] meoh_x,
                           np.ndarray[double, ndim=1] acn_x,
                           np.ndarray[double, ndim=1] thf_x,
                           np.ndarray[double, ndim=1] t,
                           double p):
    """
    Given 1D arrays for methanol, acetonitrile, THF molar fractions (meoh_x, acn_x, thf_x),
    temperature t, and pressure p, compute the total viscosity for each index.
    """
    cdef Py_ssize_t n = t.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[:] meoh = meoh_x, acn = acn_x, thf = thf_x, temp = t, out = result
    cdef Py_ssize_t i
    cdef double threshold = 1e-6, sum_mix, water_meoh_x, meoh_visc, acn_visc, thf_visc
    for i in range(n):
        sum_mix = meoh[i] + acn[i] + thf[i]
        if sum_mix < threshold:
            out[i] = viscosity_scalar("meoh", temp[i], meoh[i], p)
        else:
            water_meoh_x = 1.0 - (acn[i] + thf[i])
            meoh_visc = viscosity_scalar("meoh", temp[i], meoh[i], p)
            if acn[i] > threshold:
                acn_visc = viscosity_scalar("acn", temp[i], acn[i], p)
            else:
                acn_visc = 1.0
            if thf[i] > threshold:
                thf_visc = viscosity_scalar("thf", temp[i], thf[i], p)
            else:
                thf_visc = 1.0
            out[i] = exp(water_meoh_x * log(meoh_visc) + acn[i] * log(acn_visc) + thf[i] * log(thf_visc))
    return result

########################################
# total_viscosity (scalar version)
########################################

def total_viscosity_scalar(double meoh_x, double acn_x, double thf_x, double t, double p):
    cdef double threshold = 1e-6, sum_mix, water_meoh_x, meoh_visc, acn_visc, thf_visc
    sum_mix = meoh_x + acn_x + thf_x
    if sum_mix < threshold:
        return viscosity_scalar("meoh", t, meoh_x, p)
    else:
        water_meoh_x = 1.0 - (acn_x + thf_x)
        meoh_visc = viscosity_scalar("meoh", t, meoh_x, p)
        if acn_x > threshold:
            acn_visc = viscosity_scalar("acn", t, acn_x, p)
        else:
            acn_visc = 1.0
        if thf_x > threshold:
            thf_visc = viscosity_scalar("thf", t, thf_x, p)
        else:
            thf_visc = 1.0
        return exp(water_meoh_x * log(meoh_visc) + acn_x * log(acn_visc) + thf_x * log(thf_visc))

########################################
# kozeny_carman_model (vector version)
########################################

def kozeny_carman_model_vector(double kozeny_carman, 
                               double[:] v,
                               double[:] eta,
                               double[:] l):
    """
    Compute incremental pressure using the Kozeny–Carman model:
        pressure = kozeny_carman * v * eta * l * 1e-8.
    """
    cdef Py_ssize_t n = v.shape[0]
    cdef np.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[:] o = out
    cdef Py_ssize_t i
    for i in range(n):
        o[i] = kozeny_carman * v[i] * eta[i] * l[i] * 1e-8
    return out

########################################
# kozeny_carman_model (scalar version)
########################################

def kozeny_carman_model_scalar(double kozeny_carman, double v, double eta, double l):
    return kozeny_carman * v * eta * l * 1e-8

########################################
# pressure_driver
########################################

def pressure_driver(object solvent_profile, column_input,
                    double column_permeability_factor=20.0,
                    double tolerance=1e-5,
                    bint recalculate_viscosities=False,
                    double initial_pressure_guess=200.0):
    """
    Given a solvent_profile (a DataFrame or convertible object) and column_input (an iterable of 6 floats),
    compute and return a 1D NumPy array of pressures.
    """
    cdef Column col = Column(column_input[0], column_input[1], column_input[2],
                             column_input[3], column_input[4], column_input[5])
    
    if not isinstance(solvent_profile, pd.DataFrame):
        sp_df = pd.DataFrame(solvent_profile)
    else:
        sp_df = solvent_profile.copy()
    
    cdef double kozeny_carman = (column_permeability_factor * 150.0 /
        ((col.particle_diameter * col.particle_sphericity)**2)) * ((1.0 - col.porosity)**2) / (col.porosity**3)
    
    update_solvent_profile(sp_df, col)
    
    # Pre-convert DataFrame columns into contiguous arrays and memoryviews.
    cdef np.ndarray[double, ndim=1] time_arr = np.ascontiguousarray(sp_df["time"].to_numpy(np.float64))
    cdef double[:] inc_CV = np.ascontiguousarray(sp_df["incremental_CV"].to_numpy(np.float64))
    cdef double[:] inc_length = np.ascontiguousarray(sp_df["incremental_length"].to_numpy(np.float64))
    cdef double[:] sv = np.ascontiguousarray(sp_df["superficial_velocity"].to_numpy(np.float64))
    cdef double[:] meoh_x = np.ascontiguousarray(sp_df["meoh_x"].to_numpy(np.float64))
    cdef double[:] acn_x = np.ascontiguousarray(sp_df["acn_x"].to_numpy(np.float64))
    cdef double[:] thf_x = np.ascontiguousarray(sp_df["thf_x"].to_numpy(np.float64))
    cdef double[:] temperature = np.ascontiguousarray(sp_df["temperature"].to_numpy(np.float64))
    
    cdef Py_ssize_t N = time_arr.shape[0]
    cdef np.ndarray[double, ndim=1] pressures = np.zeros(N, dtype=np.float64)
    
    cdef Py_ssize_t i, lower_index, j, slice_len
    cdef double delta, pressure_val, fraction_accounted, col_unaccounted, col_overaccounted, local_sum
    cdef np.ndarray[double, ndim=1] current_eta_arr, incr_press_arr
    cdef double[:] current_eta, incr_press
    cdef np.ndarray[double, ndim=1] viscosities = np.zeros(N, dtype=np.float64)


    for i in range(N):
        delta = 1e12
        fraction_accounted = 0.0
        lower_index = i
        col_unaccounted = 0.0
        col_overaccounted = 0.0
        # Find the lower index so that cumulative inc_CV >= 1.
        while fraction_accounted < 1.0:
            fraction_accounted += inc_CV[lower_index]
            if fraction_accounted >= 1.0:
                col_overaccounted = fraction_accounted - 1.0
                break
            if lower_index == 0:
                col_unaccounted = 1.0 - fraction_accounted
                break
            lower_index -= 1
        
        pressure_val = 0.0
        # Iterative update loop.
        while delta > tolerance:
            if recalculate_viscosities:
                current_eta_arr = total_viscosity_vector(meoh_x[lower_index:i+1],
                                                         acn_x[lower_index:i+1],
                                                         thf_x[lower_index:i+1],
                                                         temperature[lower_index:i+1],
                                                         initial_pressure_guess)
            else:
                # Use scalar viscosity for index i.
                eta = np.array([total_viscosity_scalar(meoh_x[i],
                                                                    acn_x[i],
                                                                    thf_x[i],
                                                                    temperature[i],
                                                                    initial_pressure_guess)], dtype=np.float64)
                viscosities[i] = eta[0]
                current_eta_arr = viscosities[lower_index:i+1]

            #current_eta = current_eta_arr  # memoryview of current_eta_arr

            incr_press_arr = kozeny_carman_model_vector(kozeny_carman,
                                                         sv[lower_index:i+1],
                                                         current_eta_arr,
                                                         inc_length[lower_index:i+1])
            incr_press = incr_press_arr
            # Manually sum incr_press using a C loop.
            local_sum = 0.0
            slice_len = incr_press_arr.shape[0]
            for j in range(slice_len):
                local_sum += incr_press[j]
            pressure_val = local_sum
            if col_unaccounted > 0:
                pressure_val += col_unaccounted / inc_CV[lower_index] * incr_press[0]
            elif col_overaccounted > 0:
                pressure_val -= col_overaccounted / inc_CV[lower_index] * incr_press[slice_len-1]
            delta = fabs(pressure_val - initial_pressure_guess)
            initial_pressure_guess = pressure_val
        pressures[i] = pressure_val
    return pressures
