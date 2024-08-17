import numpy as np
from scipy.interpolate import CubicSpline
import sys, os
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from user_parameters import (
    NOISE_THRESHOLD_MULTIPLIER,
    SG_FILTER_SIZE,
    BUTTER_FILTER_SIZE,
    MINIMUM_HEIGHT,
    MINIMUM_AREA,
    BACKGROUND_NOISE_RANGE,
)

import matplotlib.pyplot as plt
from scipy.signal import (
    savgol_filter,
    sosfiltfilt,
    butter,
    argrelmin,
    argrelmax,
)

from data_processing.baseline import als_psalsa
from operator import lt, gt


class PeakFinder:
    def __init__(self, timepoints: np.array, signal: np.array) -> None:
        self.timepoints = timepoints
        self.raw_signal = self.processed_signal = signal
        self.dt = (timepoints[-1] - timepoints[0]) / len(timepoints)
        self.__initial_peak_finding()

    def __initial_peak_finding(self):
        self.__smooth_signal()
        self.__find_baseline()
        self.__apply_baseline()
        self.__get_second_derivative()
        self.find_peaks()
        self.refine_peaks()

    def __smooth_signal(self):
        signal = savgol_filter(self.processed_signal, int(round(SG_FILTER_SIZE / 2)), 7)
        signal = savgol_filter(signal, SG_FILTER_SIZE, 4)
        signal = savgol_filter(signal, SG_FILTER_SIZE, 4)

        sos = butter(1, BUTTER_FILTER_SIZE * self.dt, output="sos")
        self.smoothed_signal = sosfiltfilt(sos, signal)

    def __find_baseline(self):
        baseline, baseline_time = als_psalsa(
            self.timepoints, self.smoothed_signal, sr=5
        )
        self.baseline_spline = CubicSpline(baseline_time, baseline)

    def __apply_baseline(self):
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
        Works in five steps:
            1. find regions when the second derivative of the smoothed signal drops below a negative threshold multiple of thenoise in the second derivative
            2. expand these regions to where the second derivative goes above positive multiple of the threshold and back down again
            3. contract overlapping regions to a local max or min in the second derivative
            4. expand these regions to a local min in the processed signal, if it exists
            5. contract overlapping regions to a local max or min in the second derivative
        This function populates the `self.peaks` array.

        Args:
            noise_multiplier (float): multiplier for threshold of signal detection. Higher values result in fewer peaks detected.

        """
        self.d2_sigma = noise_multiplier * self.d2_ave_noise
        self.signal_sigma = noise_multiplier * self.signal_noise
        self.d2_upper = 2 * self.d2_sigma
        self.d2_lower = -3.5 * self.d2_sigma
        n = len(self.d2_signal)

        # Step 1: Find all contiguous regions where self.d2_signal < self.d2_lower
        regions = []
        i = 0
        while i < n:
            if self.d2_signal[i] < self.d2_lower:
                start = i
                while i < n and self.d2_signal[i] < self.d2_upper:
                    i += 1
                regions.append((start, i - 1))
            i += 1

        # Step 2: Expand each region to the left and right to where self.d2_signal > self.d2_upper and then back down to self.d2_upper
        def step_2_compare(side, sign):
            return sign(self.d2_signal[side], self.d2_upper) or sign(
                self.d2_signal[side - 1], self.d2_upper
            )

        def expand_region(side, comp_func, comp_value, addn_value):
            while comp_func(side, comp_value) and step_2_compare(side, lt):
                side += addn_value
            else:
                while comp_func(side, comp_value) and step_2_compare(side, gt):
                    side += addn_value
            return side

        expanded_regions = []
        for start, end in regions:
            left = expand_region(start, gt, 0, -1)  # Expand left
            right = expand_region(end, lt, n - 1, 1)  # Expand right
            expanded_regions.append((left, right))

        # Step 3: Expand each region to the left and right to where self.processed_signal > self.signal_sigma
        def step_3_compare(side, sign):
            return sign(self.smoothed_signal[side + 1], 3 * self.signal_sigma) or sign(
                self.smoothed_signal[side], 3 * self.signal_sigma
            )

        def expand_3_region(side, comp_func, comp_value, addn_value):
            min_val = self.smoothed_signal[side + 1]
            while comp_func(side, comp_value) and step_3_compare(side, gt):
                side += addn_value
                curr_signal = self.smoothed_signal[side + 1]
                if curr_signal < min_val:
                    min_val = curr_signal
                else:
                    break
            return side

        regions = expanded_regions
        expanded_regions = []
        for start, end in regions:
            left = expand_3_region(start, gt, 0, -1)  # Expand left
            right = expand_3_region(end, lt, n - 1, 1)  # Expand right
            expanded_regions.append((left, right))

        # Step 4: Trim overlapping regions at local minima
        trimmed_regions = []
        expanded_regions.sort()  # Sort by start time
        current_start, current_end = expanded_regions[0]

        def get_extrema(function, start, end, order=10):
            return (
                function(self.d2_signal[start : end + 1], order=order)[0] + next_start
            )

        def update_start_end(new_end):
            nonlocal trimmed_regions, current_start, current_end
            trimmed_regions.append([current_start, new_end])
            current_start, current_end = new_end, next_end

        for next_start, next_end in expanded_regions[1:]:
            if next_start <= current_end:  # Overlapping regions
                # two possible cases: poorly resolved shoulder with only a local max, or a better resolved shoulter, with a local min, still above threshold

                rel_min = get_extrema(argrelmin, next_start, current_end + 1)
                if len(rel_min) == 1:
                    update_start_end(rel_min[0])
                    continue
                elif len(rel_min) > 1:
                    ave = np.int32(np.mean(rel_min))
                    update_start_end(ave)
                    continue

                rel_max = get_extrema(argrelmax, next_start, current_end + 1)
                if len(rel_max) == 1:
                    update_start_end(rel_max[0])
                else:
                    max_idx = (
                        np.argmax(self.d2_signal[next_start : current_end + 1])
                        + next_start
                    )
                    update_start_end(max_idx)
            else:
                trimmed_regions.append([current_start, current_end])
                current_start, current_end = next_start, next_end

        # Add the last region
        trimmed_regions.append([current_start, current_end])

        self.peaks: list[Peak] = []
        for start, end in trimmed_regions:
            if start == end:
                continue
            self.peaks.append(
                Peak(
                    start_time=self.timepoints[start + 1],
                    end_time=self.timepoints[end + 1],
                    start_index=start + 1,
                    end_index=end + 1,
                    chromatogram_times=self.timepoints,
                    chromatogram_signal=self.processed_signal,
                )
            )

    def refine_peaks(self, height_cutoff=MINIMUM_HEIGHT, area_cutoff=MINIMUM_AREA):
        peaks: list[Peak] = []
        for peak in self.peaks:
            if peak.area >= area_cutoff and peak.height >= height_cutoff:
                peaks.append(peak)
        self.peaks = peaks

    def plot_peaks(self):
        # Plot the final result
        plt.figure(figsize=(14, 8))
        plt.plot(self.timepoints, self.processed_signal, color="black")
        plt.plot(self.timepoints, self.smoothed_signal, color="slategray")
        plt.plot(self.timepoints[1:-1], self.d2_signal / self.dt, color="gray")

        # Colors for adjacent peaks
        for idx, peak in enumerate(self.peaks):
            color = "orange" if idx % 2 == 0 else "blue"
            # plt.axvspan(start, end, color=color, alpha=0.5, label=f"Peak {idx + 1}")

            plt.axvspan(peak.start_time, peak.end_time, color=color, alpha=0.5)

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
            }
            peak_list.append(peak_dict)
        peak_df = pd.DataFrame.from_dict(peak_list)
        print(peak_df)


class Peak:
    def __init__(
        self,
        start_time,
        end_time,
        start_index,
        end_index,
        chromatogram_times,
        chromatogram_signal,
    ) -> None:
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.start_index: int = start_index
        self.end_index: int = end_index
        self.time_array: np.array = chromatogram_times[start_index:end_index]
        self.signal_array: np.array = chromatogram_signal[start_index:end_index]
        self.__initiate_values()

    def __initiate_values(self):
        self.get_area()
        self.get_height()

    def get_area(self):
        self.area = np.sum(self.signal_array) * (
            self.time_array[1] - self.time_array[0]
        )
        return self.area

    def get_height(self):
        self.height = np.max(self.signal_array)
        self.retention_time = self.time_array[np.argmax(self.signal_array)]
        return self.height

    def get_peak_dict(self):
        this_dict = {
            "retention_time": self.retention_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "area": self.area,
            "height": self.height,
        }
        return this_dict
