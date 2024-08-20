import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from operator import lt, gt
import matplotlib.pyplot as plt
from scipy.signal import (
    savgol_filter,
    sosfiltfilt,
    butter,
    argrelmin,
    argrelmax,
)
from data_processing import PeakList, als_psalsa

from user_parameters import (
    NOISE_THRESHOLD_MULTIPLIER,
    SG_FILTER_SIZE,
    BUTTER_FILTER_SIZE,
    MINIMUM_HEIGHT,
    MINIMUM_AREA,
    BACKGROUND_NOISE_RANGE,
    LINEAR_LIMIT,
    PEAK_LIMIT,
)


class PeakFinder:
    """
    Class to find peaks in a noisy spectrum. Peaks are located using a smoothed spectrum and its second derivative.
    """

    def __init__(self, timepoints: np.array, signal: np.array) -> None:
        """
        Initiates the PeakFinder. Instantiating this performs the following tasks:
            1. smoothing the signal using Savitzky-Golay smoothing and a Butterworth filter
            2. finding the baseline using asymmetric least squares fitting
            3. subtracting this baseline from the processes signal
            4. calculating the second derivative of the smoothed, baselined signal
            5. finding the peaks in the spectrum:
            a. calculate background noise in the second derivative
            b. determine when the second derivative crosses below a threshold multiple of the noise
            c. expanding the above range until the second derivative crosses above a positive threshold and back down.
            d. removing any duplicate peaks
            6. refining peaks to remove those with areas or heights that are too small.

        Args:
            timepoints (np.array): Array of times.
            signal (np.array): The raw signal.
        """
        self.timepoints = timepoints
        self.raw_signal = signal
        self.processed_signal = np.copy(signal)
        self.dt = (timepoints[-1] - timepoints[0]) / len(timepoints)
        self.__initial_peak_finding()

    def __initial_peak_finding(self):
        """
        Internal method to drive the functions outlined in __init__.
        """
        self.__smooth_signal()
        self.__find_baseline()
        self.__apply_baseline()
        self.__get_second_derivative()
        self.find_peaks()
        self.refine_peaks()

    def __smooth_signal(self):
        """
        Smooths the raw signal using:
        1. a Butterworth filter with size specified by user
        2. Three sequential Savitzky-Golay smoothing steps with increasing filter size
        """
        sos = butter(1, (BUTTER_FILTER_SIZE) * self.dt, output="sos")
        self.smoothed_signal = sosfiltfilt(sos, self.processed_signal)
        for i in range(3):
            self.smoothed_signal = savgol_filter(
                self.smoothed_signal, SG_FILTER_SIZE + 15 * i, 5
            )

    def __find_baseline(self):
        """
        Calculates a baseline approximation using the psalsa asymmetric least squares algorithm, and stores a cubic spline of the baseline.
        """

        baseline_time, baseline_vals = als_psalsa(
            self.timepoints, self.smoothed_signal, sr=5
        )
        self.baseline_spline = CubicSpline(baseline_time, baseline_vals)

    def __apply_baseline(self):
        """
        Applies the als-psalsa baseline to the processed and smoothed signals.
        """
        self.processed_signal -= self.baseline_spline(self.timepoints)
        self.smoothed_signal -= self.baseline_spline(self.timepoints)
        self.signal_noise = np.mean(
            np.abs(
                self.processed_signal[
                    BACKGROUND_NOISE_RANGE[0] : BACKGROUND_NOISE_RANGE[1]
                ]
            )
        )

    def __get_second_derivative(self):
        """
        Calculates the second derivative of the smoothed signal via finite differences.
        """
        d1 = self.smoothed_signal - self.baseline_spline(self.timepoints)
        d_signal1 = d1[:-2] - d1[1:-1]
        d_signal2 = d1[1:-1] - d1[2:]
        self.d2_signal = d_signal1 - d_signal2
        self.d2_ave_noise = np.mean(
            np.abs(
                self.d2_signal[BACKGROUND_NOISE_RANGE[0] : BACKGROUND_NOISE_RANGE[1]]
            )
        )

    def find_peaks(self, noise_multiplier: float = NOISE_THRESHOLD_MULTIPLIER):
        """
        Works in three steps:
            1. find regions when the second derivative of the smoothed signal drops below a negative threshold multiple of the noise in the second derivative
            2. expand these regions to where the second derivative goes above positive multiple of the threshold and back down again
            3. contract overlapping regions to a local max or min in the second derivative
        This function populates the `self.peaks` array.

        Args:
            noise_multiplier (float): multiplier for threshold of signal detection. Higher values result in fewer peaks detected.

        """
        self.d2_sigma = noise_multiplier * self.d2_ave_noise
        self.signal_sigma = noise_multiplier * self.signal_noise
        low_cutoff = -PEAK_LIMIT
        high_cutoff = 2
        n = len(self.d2_signal)

        # Step 1: Find all contiguous regions where self.d2_signal < -3.5 * self.d2_sigma
        initial_regions = []
        i = 3
        while i < n - 3:
            if (
                self.d2_signal[i] < low_cutoff * self.d2_sigma
                or self.processed_signal[i + 1] > LINEAR_LIMIT
            ):
                start = i
                while i < n and (
                    self.d2_signal[i] < 0 or self.processed_signal[i + 1] > LINEAR_LIMIT
                ):
                    i += 1
                initial_regions.append((start, i - 1))
            i += 1

        def expand_region(regions):
            """
            Method for expanding regions as found above.
            """
            nonlocal high_cutoff, low_cutoff

            def compare_d2_to_noise(index, sign):
                """
                Compares 2nd derivative to multiple of 2nd derivative noise.

                Args:
                    sign (func): lt or gt operator

                Returns:
                    (bool): returns True if either the current, previous, or next signal index is (sign) the noise threshold.
                """
                d2_1 = self.d2_signal[index + 1]
                d2_0 = self.d2_signal[index]
                d2_T = self.d2_signal[index - 1]
                d2_n = high_cutoff * self.d2_sigma
                return sign(d2_1, d2_n) or sign(d2_0, d2_n) or sign(d2_T, d2_n)

            def compare_signal_to_noise(index):
                """
                Compares signal values to multiple of signal noise.

                Returns:
                    (bool): returns True if any of the n-2, n-1, n, n+1, or n+2 values are greater than the noise threshold.
                """
                s_0 = self.smoothed_signal[index - 2 : index + 3]
                n = high_cutoff * self.signal_sigma
                return np.any(s_0 > n)

            def expand(index, comp_func, x_limit, addn_value):
                """
                expands regions while the second derivative and signal meet the requirements
                """
                while (
                    comp_func(index, x_limit)
                    and compare_d2_to_noise(index, lt)
                    and compare_signal_to_noise(index)
                ):
                    index += addn_value
                else:
                    while (
                        comp_func(index, x_limit)
                        and compare_d2_to_noise(index, gt)
                        and compare_signal_to_noise(index)
                    ):
                        index += addn_value
                return index

            expanded_regions = []
            for start, end in regions:
                # expand left and right bounds
                left = expand(start, gt, 0, -1)  # Expand left
                right = expand(end, lt, n - 1, 1)  # Expand right
                expanded_regions.append((left, right))

            return expanded_regions

        # TODO optimize this and use it
        def remove_dupulicate_regions(regions):
            current_start, current_end = regions[0]
            filtered_regions = [[current_start, current_end]]
            for region in regions[1:]:
                next_start, next_end = region
                if next_start == current_start and current_end == next_end:
                    continue
                else:
                    filtered_regions.append([next_start, next_end])
                    current_start = next_start
                    current_end = next_end
            return filtered_regions

        def trim_regions(regions):
            def get_extrema(function, signal, start, end, offset=0, **kwargs):
                vals = function(signal[start + offset : end + offset + 1], **kwargs)
                try:
                    return vals[0] + next_start + offset
                except:
                    return vals + next_start + offset

            def update_start_end(new_end):
                nonlocal trimmed_regions, current_start, current_end
                trimmed_regions.append([current_start, new_end])
                current_start, current_end = new_end, next_end

            regions = [[start, end] for start, end in regions]
            trimmed_regions = []
            regions.sort()  # Sort by start time
            current_start, current_end = regions[0]

            for next_start, next_end in regions[1:]:
                if abs(next_start - 6255) < 2:
                    pass
                if next_start <= current_end:  # Overlapping regions
                    # two possible cases: poorly resolved shoulder with only a local max, or a better resolved shoulter, with a local min, still above threshold

                    rel_min_signal = get_extrema(
                        argrelmin,
                        self.processed_signal,
                        next_start,
                        current_end + 1,
                        offset=1,
                        order=10,
                    )
                    rel_min_d2 = get_extrema(
                        argrelmin,
                        self.d2_signal,
                        next_start,
                        current_end + 1,
                        order=10,
                    )
                    rel_max_d2 = get_extrema(
                        argrelmax,
                        self.d2_signal,
                        next_start,
                        current_end + 1,
                        order=10,
                    )
                    if len(rel_min_signal) == 1:
                        update_start_end(rel_min_signal[0])
                        continue
                    elif len(rel_max_d2) == 3 and len(rel_min_d2) == 2:
                        update_start_end(rel_max_d2[1])
                        continue
                    elif len(rel_min_d2) == 1:
                        update_start_end(rel_min_d2[0])
                        continue
                    elif len(rel_min_signal) > 1:
                        update_start_end(
                            get_extrema(
                                np.argmin,
                                self.processed_signal,
                                next_start,
                                current_end,
                                offset=1,
                            )
                        )
                        update_start_end(rel_min_signal[0])
                        continue
                    else:
                        update_start_end(
                            get_extrema(
                                np.argmin, self.d2_signal, next_start, current_end
                            )
                        )
                        continue
                else:
                    trimmed_regions.append([current_start, current_end])
                    current_start, current_end = next_start, next_end

            trimmed_regions.append([current_start, current_end])  # add last region
            return trimmed_regions

        expanded_regions = expand_region(initial_regions)

        # Step 3: Trim overlapping regions at local minima
        trimmed_regions = trim_regions(expanded_regions)

        regions = []
        for start, end in trimmed_regions:
            if end - start < 2:
                continue
            else:
                regions.append([start + 1, end + 1])

        self.peaks: PeakList = PeakList(
            self.timepoints, self.raw_signal, self.smoothed_signal, self.baseline_spline
        )
        self.peaks.add_peaks(regions)

    def refine_peaks(self, height_cutoff=MINIMUM_HEIGHT, area_cutoff=MINIMUM_AREA):
        self.peaks.refine_peaks(height_cutoff, area_cutoff)

    def plot_peaks(self):
        # Plot the final result
        plt.figure(figsize=(14, 8))
        plt.plot(self.timepoints, self.raw_signal, color="black")
        plt.plot(self.timepoints, self.baseline_spline(self.timepoints), color="grey")

        # Colors for adjacent peaks
        for idx, peak in enumerate(self.peaks):
            color = "orange" if idx % 2 == 0 else "blue"
            # plt.axvspan(start, end, color=color, alpha=0.5, label=f"Peak {idx + 1}")

            plt.axvspan(peak.start_time, peak.end_time, color=color, alpha=0.2)

        plt.xlabel("Time")
        plt.ylabel("Second Derivative")
        # plt.legend()
        plt.show()

    def print_peaks(self):
        peak_list = []
        total_area = np.sum([peak.area for peak in self.peaks])
        for idx, peak in enumerate(self.peaks):
            peak_dict = peak.get_peak_dict()
            peak_dict = {
                "index": idx + 1,
                **peak_dict,
                "relative_area": np.round(100 * peak.area / total_area, 2),
                "start_ind": peak.start_index,
                "end_ind": peak.end_index,
            }
            peak_list.append(peak_dict)
        peak_df = pd.DataFrame.from_dict(peak_list)
        print(peak_df)
