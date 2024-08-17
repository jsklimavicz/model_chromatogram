import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import exponnorm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.signal import (
    savgol_filter,
    sosfiltfilt,
    find_peaks,
    peak_widths,
    butter,
    argrelextrema,
    argrelmin,
    argrelmax,
)
import math

from signal_processing import als_psalsa


df = pd.read_parquet("./data_processing/test.parquet")
raw_signal = df["y"].to_numpy()
times = df["x"].to_numpy()
dt = times[1] - times[0]

data = np.array([[x, y] for x, y in zip(times, raw_signal)])

signal = savgol_filter(raw_signal, 25, 7)
signal = savgol_filter(signal, 50, 4)

sos = butter(1, 30 * dt, output="sos")
butter_signal = sosfiltfilt(sos, signal)

baseline, baseline_time = als_psalsa(times, butter_signal, sr=5)


spline = CubicSpline(baseline_time, baseline)

baselined_signal = raw_signal - spline(times)


peaks, _ = find_peaks(
    baselined_signal,
    height=3,
    prominence=2.5,
    distance=10,
)


dt = times[1] - times[0]


baselined_butter = butter_signal - spline(times)
d_signal1 = baselined_butter[:-2] - baselined_butter[1:-1]
d_signal2 = baselined_butter[1:-1] - baselined_butter[2:]
d2_signal = d_signal1 - d_signal2
dt2 = times[1:-1]

ave_noise = np.mean(np.abs(d2_signal[:100]))


# plt.plot(dt, d2_signal)
# plt.hlines(
#     [-3 * ave_noise, 2 * ave_noise], times[1], times[-1], linestyles="--", color="red"
# )
# plt.show()


def find_peaks(time, d2_chrom, sigma):
    n = len(time)
    regions = []

    # Step 1: Find all contiguous regions where d2_chrom < -3 * sigma
    i = 0
    while i < n:
        if d2_chrom[i] < -3 * sigma:
            start = i
            while i < n and d2_chrom[i] < 2 * sigma:
                i += 1
            regions.append((start, i - 1))
        i += 1

    expanded_regions = []

    # Step 2: Expand each region to the left and right to where d2_chrom > 2 * sigma and then back down to 2 * sigma
    count = 0
    for start, end in regions:
        count += 1
        if count == 8:
            pass
        # Expand left
        left = start
        while left > 0 and (
            d2_chrom[left] < 2 * sigma or d2_chrom[left - 1] < 2 * sigma
        ):
            left -= 1
        else:
            while left > 0 and (
                d2_chrom[left] > 2 * sigma or d2_chrom[left - 1] > 2 * sigma
            ):
                left -= 1

        # Expand right
        right = end
        while right < n - 1 and (
            d2_chrom[right] < 2 * sigma or d2_chrom[right + 1] < 2 * sigma
        ):
            right += 1
        else:
            while right < n - 1 and (
                d2_chrom[right] > 2 * sigma or d2_chrom[right + 1] > 2 * sigma
            ):
                right += 1

        expanded_regions.append((left, right))
    # Step 3: Trim overlapping regions at local minima
    trimmed_regions = []
    expanded_regions.sort()  # Sort by start time
    current_start, current_end = expanded_regions[0]

    for next_start, next_end in expanded_regions[1:]:
        if next_start <= current_end:  # Overlapping regions
            # two possible cases:
            #   poorly resolved shoulder, only a local max, or
            #   better resolved shoulter, with a local min, still above threshold

            rel_min = (
                argrelmin(d2_chrom[next_start : current_end + 1], order=10)[0]
                + next_start
            )
            if len(rel_min) == 1:
                trimmed_regions.append([current_start, rel_min[0]])
                current_start, current_end = rel_min[0], next_end
                continue
            elif len(rel_min) > 1:
                ave = np.int32(np.mean(rel_min))
                trimmed_regions.append([current_start, ave])
                current_start, current_end = ave, next_end
                continue

            rel_max = (
                argrelmax(d2_chrom[next_start : current_end + 1], order=10)[0]
                + next_start
            )
            if len(rel_max) == 1:
                trimmed_regions.append([current_start, rel_max[0]])
                current_start, current_end = rel_max[0], next_end
            else:
                max_idx = np.argmax(d2_chrom[next_start : current_end + 1]) + next_start
                trimmed_regions.append([current_start, max_idx])
                current_start, current_end = max_idx, next_end
        else:
            trimmed_regions.append([current_start, current_end])
            current_start, current_end = next_start, next_end

    # Add the last region
    trimmed_regions.append([current_start, current_end])

    # Convert regions from indices to time intervals
    peak_intervals = [(time[start], time[end]) for start, end in trimmed_regions]
    indices = [np.array(ind) + 1 for ind in trimmed_regions]

    return peak_intervals, indices


def peak_area(signal, dt, index_start, index_end):
    area = np.sum(signal[index_start:index_end]) * dt
    return area


def peak_height(signal, index_start, index_end):
    interval = signal[index_start:index_end]
    return np.max(interval), np.argmax(interval)


peaks, indices = find_peaks(dt2, d2_signal, ave_noise)

filtered_peaks = []
filtered_indices = []
for ind, index in enumerate(indices):
    index_start, index_end = index
    height, _ = peak_height(baselined_signal, index_start, index_end)
    area = round(peak_area(baselined_signal, dt, index_start, index_end), 3)
    if area >= 0.1 and height >= 2:
        filtered_peaks.append(peaks[ind])
        filtered_indices.append(indices[ind])


indices = filtered_indices
peaks = filtered_peaks

count = 0
for peak, index in zip(peaks, indices):
    count += 1
    index_start, index_end = index
    height, _ = peak_height(baselined_signal, index_start, index_end)
    area = round(peak_area(baselined_signal, dt, index_start, index_end), 3)
    start = round(peak[0], 3)
    end = round(peak[1], 3)
    print(
        f"peak {count}: \t{start}\t--  {end}\t Area: {area} \t Height: {round(height,3)}"
    )

import matplotlib.pyplot as plt

# Plot the final result
plt.figure(figsize=(14, 8))
plt.plot(times, baselined_signal, color="black")
plt.plot(dt2, d2_signal / dt, color="gray")

# Colors for adjacent peaks
for idx, (start, end) in enumerate(peaks):
    color = "orange" if idx % 2 == 0 else "blue"
    # plt.axvspan(start, end, color=color, alpha=0.5, label=f"Peak {idx + 1}")

    plt.axvspan(start, end, color=color, alpha=0.5)

plt.xlabel("Time")
plt.ylabel("Second Derivative")
plt.legend()
plt.show()
