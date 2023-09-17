import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import exponnorm
from general_parameters import signal_globals as sg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
## Create Baseline with noise

### Create Baseline

max_time = sg.RUN_LENGTH

samples_per_minute = 60 * sg.SAMPLE_RATE


times = np.arange(0, max_time, 1./samples_per_minute)
signal = np.random.normal(loc = 0, scale = sg.BASELINE_NOISE, size = times.shape)
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
peaks, _ = find_peaks(signal, height=signal.max()/100, prominence = signal.max()/200, distance = sg.SAMPLE_RATE * 10)
# plt.plot(times, signal)

# plt.plot(times[peaks], signal[peaks], "x")

# plt.plot(times, np.zeros_like(signal), "--", color="gray")

# plt.show()


peak_widths(signal, peaks)

results_half = peak_widths(signal, peaks, rel_height=0.5)
results_full = peak_widths(signal, peaks, rel_height=1)
# plt.plot(times, signal)

# plt.plot(times[peaks], signal[peaks], "x")

num_points = sg.SAMPLE_RATE * 60

# plt.hlines(results_half[1], results_half[2]/num_points, results_half[3]/num_points, color="C2")

# plt.hlines(results_full[1], results_full[2]/num_points, results_full[3]/num_points, color="C3")
# # plt.hlines(*results_full[1:], color="C3")

# plt.show()
## Finding good minima
inv_signal = signal.max() - signal

minima, _ = find_peaks(inv_signal, 
                      height=signal.max()/100, 
                      prominence = signal.max()/1000, 
                      distance = sg.SAMPLE_RATE * 10)
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

def filter_tuples(input_list):
    results = []
    series = []
    encountered_1 = False

    for tup in input_list:
        _, b = tup

        if not series:
            # If series is empty, add the tuple with b == 1 to the series
            if b == 1:
                series.append(tup)
                encountered_1 = True
            continue

        # Check if the current tuple breaks the series
        if (b == 1 and series[-1][1] == -1) or (b == -1 and series[-1][1] == 1):
            if not encountered_1:
                # If we haven't encountered 1 yet, omit the series
                series = []
            else:
                # Otherwise, apply the rules and add to results
                if len(series) > 4 and series[0][1] == 1:
                    results.extend(series[:2] + series[-2:])
                else:
                    results.extend(series)
                series = []

        series.append(tup)

    # Handle the last series
    if series:
        if len(series) > 4 and series[0][1] == 1:
            results.extend(series[:2] + series[-2:])
        else:
            results.extend(series)

    # Handle the case where the list starts with a series of b == 1 
    initial_ones_count = 0 
    for tup in input_list:
        _, b = tup
        if b == 1:
            initial_ones_count += 1
        else: break
    if initial_ones_count > 2:
        results = results[initial_ones_count - 2:]

    return results


# Test cases
input1 = [(0.022, 1), (0.023, 1), (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)]
output1 = filter_tuples(input1)
assert(output1 == [ (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)])

input2 = [(0.022, -1), (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)]
output2 = filter_tuples(input2)
assert(output2 == [ (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)])

input3 = [(0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1),
          (1.487, 1), (1.497, 1), (1.513, 1), (1.707, 1), (1.71, -1), (1.883, -1), (1.89, 1), (2.403, 1),
          (2.405, -1), (2.583, -1), (2.585, 1), (2.962, 1), (2.988, 1), (3.002, 1), (4.81, 1), (5.045, 1),
          (5.047, -1), (5.228, -1), (5.23, 1), (5.477, 1), (5.478, -1), (5.657, -1), (5.658, 1), (5.952, 1),
          (6.028, -1), (6.157, -1), (6.442, 1), (6.455, 1), (8.853, 1), (9.06, 1), (9.067, -1), (9.252, -1),
          (9.26, 1), (9.54, 1)]
output3 = filter_tuples(input3)
assert(output3 == [(0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1),
          (1.513, 1), (1.707, 1), (1.71, -1), (1.883, -1), (1.89, 1), (2.403, 1),
          (2.405, -1), (2.583, -1), (2.585, 1), (2.962, 1), (4.81, 1), (5.045, 1),
          (5.047, -1), (5.228, -1), (5.23, 1), (5.477, 1), (5.478, -1), (5.657, -1), (5.658, 1), (5.952, 1),
          (6.028, -1), (6.157, -1), (6.442, 1), (6.455, 1), (8.853, 1), (9.06, 1), (9.067, -1), (9.252, -1),
          (9.26, 1), (9.54, 1)])

