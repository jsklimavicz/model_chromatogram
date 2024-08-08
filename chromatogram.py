import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import exponnorm
from user_parameters import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
from peak_finding import *

## Create Baseline with noise

# Create Baseline
samples_per_minute = 60 * SAMPLE_RATE

times = np.arange(0, RUN_LENGTH, 1./samples_per_minute)
signal = np.random.normal(loc = 0, scale = BASELINE_NOISE, size = times.shape)
signal = savgol_filter(signal, 15, 3)
## Load peaks table
from peaks import peaks as peaks_list
peaks_list
## Create Signal for Peaks
for peak in peaks_list:
    signal += peak["height"] * exponnorm.pdf(times, K = peak["asymmetry"], loc = peak["time"], scale = peak["width"])

# plt.plot(times, signal)
# plt.show()

from scipy.signal import find_peaks, peak_widths
peaks, _ = find_peaks(signal, height=signal.max()/100, prominence = signal.max()/200, distance = SAMPLE_RATE * 10)
# plt.plot(times, signal)

# plt.plot(times[peaks], signal[peaks], "x")

# plt.plot(times, np.zeros_like(signal), "--", color="gray")

# plt.show()


peak_widths(signal, peaks)

results_half = peak_widths(signal, peaks, rel_height=0.5)
results_full = peak_widths(signal, peaks, rel_height=1)
# plt.plot(times, signal)

# plt.plot(times[peaks], signal[peaks], "x")

num_points = SAMPLE_RATE * 60

# plt.hlines(results_half[1], results_half[2]/num_points, results_half[3]/num_points, color="C2")

# plt.hlines(results_full[1], results_full[2]/num_points, results_full[3]/num_points, color="C3")
# # plt.hlines(*results_full[1:], color="C3")

# plt.show()
## Finding good minima
inv_signal = signal.max() - signal

minima, _ = find_peaks(inv_signal, 
                      height=signal.max()/100, 
                      prominence = signal.max()/1000, 
                      distance = SAMPLE_RATE * 10)
# plt.plot(times, signal)

# plt.plot(times[minima], signal[minima], "x")

# plt.show()
# heights = signal[peaks]
# means = times[peaks]
# sigmas = ((results_half[3] - results_half[2])/num_points)/(2 * math.sqrt(2 * math.log(2)))

# from scipy.stats import norm

# def gaussian_pdf(x, height = 1, scale = 1, loc = 0):
#     return height / (scale * math.sqrt(2*math.pi)) * math.exp(-.5 * ((x - loc)/scale) ** 2)

# def fitted_spectrum(x, *args):
#     """
#     Args expected in form of [*heights, *times, *sigmas]
#     """
#     y = np.zeros_like(x)
#     num_gaussians = int(len(args)/3)
#     for i in range(0, num_gaussians):
#         h, t, s  = args[i], args[i+ num_gaussians], args[i + 2*num_gaussians]
#         y += np.array([gaussian_pdf(a, height = h, scale = s, loc = t) for a in x])
#     return y

# initial_guess = np.array([*heights, *means, *sigmas])

# from scipy.optimize import curve_fit
# popt, pcov = curve_fit(f = fitted_spectrum, xdata = times, ydata = signal, p0 = initial_guess )

# plt.plot(times, signal)
# plt.plot(times, fitted_spectrum(times, *popt), label="Fitted Curve", color='red')
# plt.show()


d_signal1 = savgol_filter(signal[:-2]-signal[1:-1], 25, 3)
d_signal2 = savgol_filter(signal[1:-1]-signal[2:], 25, 3)
d2_signal = savgol_filter(d_signal1 - d_signal2, 25, 3)
d2_signal = savgol_filter(d2_signal, 25, 3)
dt = times[1:-1]

height_param = 20

min_t = 11 * samples_per_minute
max_t = 12 * samples_per_minute
signals_in_range = d2_signal[min_t:max_t]
noise_sd = np.std(signals_in_range)

threshhold = 5*noise_sd

crossings = []

for i in range(int(samples_per_minute/60),len(d2_signal)-int(samples_per_minute/60)):
    if (d2_signal[i] > threshhold and d2_signal[i-1] <= threshhold) or (d2_signal[i] < threshhold and d2_signal[i-1] >= threshhold):
        crossings.append((i/samples_per_minute, 1))
    if (d2_signal[i] > -threshhold and d2_signal[i-1] <= -threshhold) or (d2_signal[i] < -threshhold and d2_signal[i-1] >= -threshhold):
        crossings.append((i/samples_per_minute, -1))


filtered_crossings = [(round(a,3),b) for (a,b) in filter_tuples(crossings)]

def find_peak_segments(filtered_crossings):
    peak_start = filtered_crossings[0][0]
    results = []
    series = filtered_crossings[0]

    for tup in filtered_crossings:
        a, b = tup

        # Check if the current tuple breaks the series
        if (b == 1 and series[-1][1] == -1) or (b == -1 and series[-1][1] == 1):
            if len(series) == 4 and series[0][1] == 1:
                results.extend(series[:2] + series[-2:])
            else:
                results.extend(series)
            series = []

        series.append(tup)

    return results