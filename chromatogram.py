import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import exponnorm
import math

from user_parameters import *
from peaks import peaks as peaks_list
from analysis.peak_finding import *
from analysis.EMG_curvefit import expmodgauss

max_time = RUN_LENGTH

samples_per_minute = 60 * SAMPLE_RATE

times = np.arange(0, max_time, 1.0 / samples_per_minute)
signal = np.random.normal(loc=0, scale=BASELINE_NOISE, size=times.shape)
signal = savgol_filter(signal, 15, 3)


for peak in peaks_list:
    signal += peak["height"] * exponnorm.pdf(
        times, K=peak["asymmetry"], loc=peak["time"], scale=peak["width"]
    )

# plt.plot(times, signal)
# plt.xlabel("Time (min)")
# plt.ylabel("Absorbance (mAU)")
# plt.show()

peaks, _ = find_peaks(
    signal,
    height=signal.max() / 100,
    prominence=signal.max() / 200,
    distance=SAMPLE_RATE * 10,
)
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

inv_signal = signal.max() - signal

minima, _ = find_peaks(
    inv_signal,
    height=signal.max() / 100,
    prominence=signal.max() / 1000,
    distance=SAMPLE_RATE * 10,
)
# plt.plot(times, signal)

# plt.plot(times[minima], signal[minima], "x")

# plt.show()

means = times[peaks]
sigmas = ((results_half[3] - results_half[2]) / num_points) / (
    2 * math.sqrt(2 * math.log(2))
)
heights = signal[peaks] / (math.sqrt(2 * math.pi)) * 0.67


def gaussian_pdf(x, height=1, scale=1, loc=0):
    return (
        height
        / (scale * math.sqrt(2 * math.pi))
        * math.exp(-0.5 * ((x - loc) / scale) ** 2)
    )


def fitted_spectrum(x, *args):
    """
    Args expected in form of [*heights, *times, *sigmas]
    """
    y = np.zeros_like(x)
    num_gaussians = int(len(args) / 3)
    for i in range(0, num_gaussians):
        h, t, s = args[i], args[i + num_gaussians], args[i + 2 * num_gaussians]
        y += np.array([gaussian_pdf(a, height=h, scale=s, loc=t) for a in x])
    return y


initial_guess = np.array([*heights, *means, *sigmas])


popt, pcov = curve_fit(f=fitted_spectrum, xdata=times, ydata=signal, p0=initial_guess)

# plt.plot(times, signal)
# plt.plot(times, fitted_spectrum(times, *popt), label="Fitted Curve", color='red')
# plt.show()


def fitted_spectrum(x, *args):
    """
    Args expected in form of [*heights, *times, *sigmas]
    """
    y = np.zeros_like(x)
    num_gaussians = int(len(args) / 3)
    for i in range(0, num_gaussians):
        h, t, s = args[i], args[i + num_gaussians], args[i + 2 * num_gaussians]
        y += expmodgauss(x, h, mu=t, sigma=s, tau=1)
    return y


initial_guess = np.array([*heights, *means, *sigmas])


popt, pcov = curve_fit(f=fitted_spectrum, xdata=times, ydata=signal, p0=initial_guess)

plt.plot(times, signal)
plt.plot(times, fitted_spectrum(times, *popt), label="Fitted Curve", color="red")
plt.show()
