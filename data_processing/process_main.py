import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import exponnorm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, sosfiltfilt, find_peaks, peak_widths, butter
import math

from signal_processing import als_psalsa

df = pd.read_parquet("./data_processing/test.parquet")
raw_signal = df["y"].to_numpy()
times = df["x"].to_numpy()
dt = times[1] - times[0]

data = np.array([[x, y] for x, y in zip(times, raw_signal)])

# from rdp import rdp
# mask = rdp(data, epsilon=0.1, algo="rec")
# plt.plot(times, raw_signal)
# plt.plot(mask[:, 0], mask[:, 1], c="red")


sos = butter(1, 60 * dt, output="sos")
butter_signal = sosfiltfilt(sos, raw_signal)
butter_signal = sosfiltfilt(sos, butter_signal)
plt.plot(times, raw_signal)
plt.plot(times, butter_signal, c="red")


plt.show()

signal = savgol_filter(signal, 11, 5)

# b, a = butter(1, 0.1)
# signal = filtfilt(b, a, signal)

sos = butter(1, 0.01, output="sos")
butter_signal = sosfiltfilt(sos, raw_signal)


baseline, baseline_time = als_psalsa(times, butter_signal)


spline = CubicSpline(baseline_time, baseline)

baselined_signal = raw_signal - spline(times)


peaks, _ = find_peaks(
    baselined_signal,
    height=3,
    prominence=2.5,
    distance=10,
)


plt.plot(times, baselined_signal)
# plt.plot(times, butter_signal - spline(times), c="red")

plt.plot(times[peaks], baselined_signal[peaks], "x")
plt.plot(times, np.zeros_like(times), "--", color="gray")

dt = times[1] - times[0]

results_full = peak_widths(baselined_signal, peaks, rel_height=0.65)

num_points = int(round(1 / dt))

plt.hlines(
    results_full[1],
    results_full[2] / num_points,
    results_full[3] / num_points,
    color="C2",
)

# d_signal1 = savgol_filter(signal[:-2] - signal[1:-1], 5, 3)
# d_signal2 = savgol_filter(signal[1:-1] - signal[2:], 5, 3)
# d2_signal = savgol_filter(d_signal1 - d_signal2, 5, 3)
# dt = times[1:-1]
# plt.plot(dt, d2_signal)
plt.show()
