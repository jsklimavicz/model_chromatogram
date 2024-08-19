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
    LINEAR_LIMIT,
    PEAK_LIMIT,
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
        self.raw_signal = signal
        self.processed_signal = np.copy(signal)
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
        sos = butter(1, (BUTTER_FILTER_SIZE) * self.dt, output="sos")
        self.smoothed_signal = sosfiltfilt(sos, self.processed_signal)
        for i in range(3):
            self.smoothed_signal = savgol_filter(
                self.smoothed_signal, SG_FILTER_SIZE + 15 * i, 5
            )

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
        low_cutoff = -PEAK_LIMIT
        high_cutoff = 2
        n = len(self.d2_signal)

        # Step 1: Find all contiguous regions where self.d2_signal < -3.5 * self.d2_sigma
        initial_regions = []
        i = 3
        # TODO: split peak if 2nd derivative rises and goes back down while signal is less than LINEAR_LIMIT
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
            nonlocal high_cutoff, low_cutoff

            def compare_d2_to_noise(index, sign):
                d2_0 = self.d2_signal[index]
                d2_1 = self.d2_signal[index - 1]
                d2_n = high_cutoff * self.d2_sigma
                return sign(d2_1, d2_n) or sign(d2_0, d2_n)

            def compare_signal_to_noise(index):
                # TODO expand to range of 3 with np.any
                s_0 = self.smoothed_signal[index + 1]
                s_1 = self.smoothed_signal[index]
                n = high_cutoff * self.signal_sigma
                return s_0 > n or s_1 > n

            def expand(index, comp_func, x_limit, addn_value):
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

        self.peaks: list[Peak] = []
        for start, end in trimmed_regions:
            if end - start < 2:
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


class PeakList:
    def __init__(self, times, chromatogram) -> None:
        self.peaks: list[Peak] = []
        self.signal = chromatogram
        self.times = times

    # TODO implement sorting for peaks after each addition
    # TODO class methods for percent height and area
    # TODO peak width methods; undefined for some peak types
    # TODO resolution methods
    # TODO asymmetry methods
    # TODO plate count methods
    # TODO peak baseline type: baseline, adjacent peak, etc

    def add_peak(self, start_index, end_index) -> None:

        peak = Peak()


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
