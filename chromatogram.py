import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import exponnorm
from general_parameters import signal_globals as sg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
## Create Baseline with noise

gradient_table = pd.read_csv("./gradients.csv")
gradient_table.head()

### Create Baseline

max_time = gradient_table.time.max()

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

min_t = 11
max_t = 12
signals_in_range = d2_signal[round(min_t * samples_per_minute):round(max_t * samples_per_minute)]
noise_sd = np.std(signals_in_range)

threshhold = 5*noise_sd

crossings = []

for i in range(int(samples_per_minute/60),len(d2_signal)-int(samples_per_minute/60)):
    if (d2_signal[i] > threshhold and d2_signal[i-1] <= threshhold) or (d2_signal[i] < threshhold and d2_signal[i-1] >= threshhold):
        crossings.append((i/samples_per_minute, 1))
    if (d2_signal[i] > -threshhold and d2_signal[i-1] <= -threshhold) or (d2_signal[i] < -threshhold and d2_signal[i-1] >= -threshhold):
        crossings.append((i/samples_per_minute, -1))
        
print(crossings)
        
crossing_count = 0
cleaned_crossings  = []
ind = 0
while ind < len(crossings):
    crossing = crossings[ind]
    crossing_type = crossing[1]
    if crossing_count == 2 and crossing_type == 1:
        #count ones ahead
        ahead_count = 0
        ahead_ind = ind + 1
        while ahead_ind < length(crossings):
            if crossings[ahead_ind][1] == 1:
                ahead_ind += 1
                ahead_count += 1
            else:
                break 
        if ahead_count > 2:
            ind = ahead_ind - 2
    else:
        ind += 1
    cleaned_crossings.append(crossing)
        
cleaned_crossings    
