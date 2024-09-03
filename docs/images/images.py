import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
import numpy.random as random

random.seed(42)

baseline_slope = 3
sample_rate = 5  # Hz
start_time = 8
end_time = 8.75
n_samples = int(60 * sample_rate * (end_time - start_time) + 1)
baseline_slope = 4


def create_noisy_peak(times, height, K, loc, scale, noise_sd=0.5):
    signal = height * exponnorm.pdf(times, K, loc=loc, scale=scale)
    signal += random.normal(loc=0, scale=noise_sd, size=len(signal))
    return signal


def calculate_retention_time(times, signal):
    # Step 1: Find the index of the highest point of the peak
    max_idx = np.argmax(signal)

    # Step 2: Define the range around the highest point (3 points on each side, if possible)
    start_idx = max(0, max_idx - 10)
    end_idx = min(len(signal) - 1, max_idx + 10)

    # Extract the relevant times and signal values
    fit_times = times[start_idx : end_idx + 1]
    fit_signal = signal[start_idx : end_idx + 1]

    # Step 3: Fit a quadratic to these points
    coeffs = np.polyfit(fit_times, fit_signal, 2)

    # The vertex of the quadratic is given by -b/(2a), where the polynomial is ax^2 + bx + c
    a, b, _ = coeffs
    retention_time = -b / (2 * a)

    return retention_time, coeffs


times = np.linspace(start_time, end_time, n_samples)
noisy_peak = create_noisy_peak(times, 10, K=1.2, loc=8.2, scale=0.05, noise_sd=1)
retention_time, coeffs = calculate_retention_time(times, noisy_peak)
baseline = times * baseline_slope - 20
noisy_peak_baseline = noisy_peak + baseline


pre_retention_idx = np.where((times < retention_time) & (noisy_peak < 0))[0]
if len(pre_retention_idx) > 0:
    start_idx = pre_retention_idx[-1] + 1
else:
    start_idx = 0  # If no crossing is found, start from the beginning

# Find the index of the first baseline crossing after the retention time
post_retention_idx = np.where((times > retention_time) & (noisy_peak < 0))[0]
if len(post_retention_idx) > 0:
    end_idx = post_retention_idx[0] - 1
else:
    end_idx = len(times) - 1  # If no crossing is found, go to the end


# Plot the noisy peak, baseline, and shaded area with retention time marked
plt.figure(figsize=(7.5, 4.5))
plt.plot(times, noisy_peak_baseline, label="Signal", c="blue")
plt.plot(times, baseline, label="Baseline", color="black")
plt.fill_between(
    times[start_idx : end_idx + 1],
    baseline[start_idx : end_idx + 1],
    noisy_peak_baseline[start_idx : end_idx + 1],
    where=noisy_peak_baseline[start_idx : end_idx + 1]
    > baseline[start_idx : end_idx + 1],
    color="blue",
    alpha=0.2,
    label="Peak area",
)

ymax = np.interp(retention_time, times, noisy_peak_baseline)
ymin = np.interp(retention_time, times, baseline)

# Mark the retention time on the plot
plt.vlines(
    x=retention_time,
    ymax=ymax,
    ymin=ymin,
    linestyle="-.",
    color="black",
    label=f"Retention time and height",
)


ymin = baseline[start_idx]
ymax = ymin + 10

plt.text(
    times[start_idx],
    ymax + 1,
    "start time",
    color="black",
    fontsize=12,
    ha="center",
)
plt.vlines(
    times[start_idx],
    ymax=ymax,
    ymin=ymin,
    linestyle="--",
    color="black",
)

ymin = baseline[end_idx]
ymax = ymin + 10
plt.text(
    times[end_idx],
    ymax + 1,
    "end time",
    color="black",
    fontsize=12,
    ha="center",
)
plt.vlines(
    times[end_idx],
    ymax=ymax,
    ymin=ymin,
    linestyle="--",
    color="black",
)


plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.savefig(
    "./docs/images/baseline_and_retention_time.png", dpi=300, bbox_inches="tight"
)


# Further expand the region to include two more points on either side of the peak
further_expanded_region_start_idx = max(0, np.searchsorted(times, retention_time) - 10)
further_expanded_region_end_idx = min(
    len(times) - 1, np.searchsorted(times, retention_time) + 10
)

further_expanded_fit_times = times[
    further_expanded_region_start_idx : further_expanded_region_end_idx + 1
]
further_expanded_fit_signal = noisy_peak[
    further_expanded_region_start_idx : further_expanded_region_end_idx + 1
]

# Create a smoother curve by evaluating the quadratic at finer intervals
further_smooth_fit_times = np.linspace(
    further_expanded_fit_times[0], further_expanded_fit_times[-1], 200
)
further_smooth_quadratic_fit = np.polyval(coeffs, further_smooth_fit_times)

# Plot the further expanded view with the quadratic fit in a different color
plt.figure(figsize=(7.5, 4.5))
plt.plot(
    further_expanded_fit_times,
    further_expanded_fit_signal,
    "o-",
    color="blue",
    label="Data Points near Peak",
)
plt.plot(
    further_smooth_fit_times,
    further_smooth_quadratic_fit,
    "-",
    color="black",
    label="Quadratic Fit",
)
plt.axvline(
    x=retention_time,
    color="black",
    linestyle="--",
    label=f"Retention Time = {retention_time:.3f}",
)

plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.savefig("./docs/images/quadratic_fit.png", dpi=300, bbox_inches="tight")
