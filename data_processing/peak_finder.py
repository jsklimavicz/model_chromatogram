import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal
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
from data_processing import PeakList, als_psalsa, Peak
from scipy.stats import exponnorm
from user_parameters import (
    NOISE_THRESHOLD_MULTIPLIER,
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

        sample_rate = int(round((1 / self.dt) / 60))
        sos = butter(1, (BUTTER_FILTER_SIZE) * self.dt, output="sos")
        self.smoothed_signal = sosfiltfilt(sos, self.processed_signal)
        self.smoothed_signal = savgol_filter(
            self.smoothed_signal, 3 * sample_rate + 10, 3
        )

        half_size = int(round(sample_rate * 3 / 2))
        window_size = 2 * half_size
        window = signal.windows.gaussian(window_size, std=2 * sample_rate)
        s = np.sum(window)
        K = 0.2
        window_2 = exponnorm.pdf(
            range(-half_size, half_size),
            loc=-sample_rate * K,
            scale=sample_rate,
            K=K,
        )
        m = np.sum(window_2)
        window_2 *= s / m

        self.smoothed_signal = np.pad(
            self.smoothed_signal,
            (half_size, half_size - 1),
            "constant",
            constant_values=(self.smoothed_signal[0], self.smoothed_signal[-1]),
        )
        self.smoothed_signal = signal.convolve(
            self.smoothed_signal, window_2, mode="valid"
        ) / sum(window_2)

        self.smoothed_signal = np.pad(
            self.smoothed_signal,
            (half_size, half_size - 1),
            "constant",
            constant_values=(self.smoothed_signal[0], self.smoothed_signal[-1]),
        )
        self.smoothed_signal = signal.convolve(
            self.smoothed_signal, window_2, mode="valid"
        ) / sum(window_2)

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
        self.signal_noise = np.sqrt(
            np.mean(
                self.processed_signal[
                    BACKGROUND_NOISE_RANGE[0] : BACKGROUND_NOISE_RANGE[1]
                ]
                ** 2
            )
        )

    def __get_second_derivative(self):
        """
        Calculates the second derivative of the smoothed signal via finite differences.
        """
        # d1 = self.smoothed_signal - self.baseline_spline(self.timepoints)
        d1 = self.smoothed_signal
        d_signal1 = d1[:-2] - d1[1:-1]
        d_signal2 = d1[1:-1] - d1[2:]
        self.d2_signal = d_signal1 - d_signal2
        self.d2_ave_noise = np.sqrt(
            np.mean(
                self.d2_signal[BACKGROUND_NOISE_RANGE[0] : BACKGROUND_NOISE_RANGE[1]]
            )
            ** 2
        )
        self.d2_signal = np.pad(self.d2_signal, (1, 1))

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
        self.signal_sigma = noise_multiplier * self.signal_noise * 1.5
        low_cutoff = -PEAK_LIMIT
        high_cutoff = 1.5
        n = len(self.d2_signal)

        # Step 1: Find all contiguous regions where self.d2_signal < -3.5 * self.d2_sigma
        initial_regions = []
        i = 3

        while i < n - 3:
            if i > 5650:
                pass
            curr_sig = self.d2_signal[i]
            if (
                self.d2_signal[i] < low_cutoff * self.d2_sigma
                or self.processed_signal[i] > LINEAR_LIMIT
            ):
                start = i
                while i < n and (
                    self.d2_signal[i] < 0 or self.processed_signal[i] > LINEAR_LIMIT
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
                    (bool): returns True if at least two of the n-4 to n+4 values are greater than the noise threshold.
                """
                s_0 = self.smoothed_signal[index - 4 : index + 5]
                n = high_cutoff * self.signal_sigma
                return sum(s_0 > n) >= 2

            def expand(index, comp_func, x_limit, addn_value):
                """
                expands regions while the second derivative and signal meet the requirements
                """
                while (
                    comp_func(index, x_limit)  # index in bounds
                    and compare_d2_to_noise(index, lt)  # d2 < noise cutoff
                    and compare_signal_to_noise(index)  # signal > noise cutoff
                ):
                    index += addn_value
                else:
                    while comp_func(index, x_limit) and (
                        compare_d2_to_noise(index, gt)
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

        def trim_regions(regions):
            def split_regions(start, end, split_points):
                sub_regions = [(start, split_points[0])]
                for i in range(len(split_points) - 1):
                    sub_regions.append((split_points[i], split_points[i + 1]))
                sub_regions.append((split_points[-1], end))
                return sub_regions

            # Step 1: Get unique regions along with their multiplicity
            unique_regions = {}
            for region in regions:
                region_tuple = tuple(region)
                if region_tuple in unique_regions:
                    unique_regions[region_tuple] += 1
                else:
                    unique_regions[region_tuple] = 1

            # Step 2: Process each region-multiplicity pair
            final_regions = []
            for region, k in unique_regions.items():
                start, end = region

                if k == 1:
                    # If multiplicity is 1, just add the region as is
                    final_regions.append((start, end))
                    continue

                # Find local minima in D2
                local_minima = (
                    argrelmin(self.d2_signal[start : end + 1], order=10)[0] + start
                )

                # Step 2a: If the number of D2 local minima is equal to multiplicity k: k peaks
                if len(local_minima) == k:
                    # Split the region into k sub-regions at each local maximum between minima
                    split_points = []
                    for i in range(len(local_minima) - 1):
                        min_idx = argrelmin(
                            self.smoothed_signal[
                                local_minima[i] : local_minima[i + 1] + 1
                            ],
                            order=10,
                        )[0]
                        if len(min_idx) == 1:
                            split_points.append(local_minima[i] + min_idx[0])
                        elif len(min_idx) == 0:
                            # we have a shoulder with no local min. Split based on d2:
                            max_idx = np.argmax(
                                self.d2_signal[
                                    local_minima[i] : local_minima[i + 1] + 1
                                ]
                            )
                            split_points.append(local_minima[i] + max_idx)
                        else:
                            min_idx = np.argmin(
                                self.smoothed_signal[
                                    local_minima[i] : local_minima[i + 1] + 1
                                ]
                            )
                            split_points.append(local_minima[i] + min_idx)

                    # Create sub-regions
                    sub_regions = split_regions(start, end, split_points)
                    final_regions.extend(sub_regions)

                # Step 2b: Raise an exception if the number of minima doesn't match k
                else:
                    # Step 2.b.1: Handle case where multiplicity is k and exactly k maxima exist
                    local_maxima = (
                        argrelmax(self.smoothed_signal[start : end + 1], order=10)[0]
                        + start
                    )
                    if len(local_maxima) == k:
                        split_points = []
                        for i in range(len(local_maxima) - 1):
                            # Split at the minimum value between each consecutive maxima
                            min_idx = (
                                np.argmin(
                                    self.smoothed_signal[
                                        local_maxima[i] : local_maxima[i + 1] + 1
                                    ]
                                )
                                + local_maxima[i]
                            )
                            split_points.append(min_idx)

                        # Create sub-regions
                        sub_regions = split_regions(start, end, split_points)
                        final_regions.extend(sub_regions)

                    # Step 2.b.2: Handle shoulder case where len(local_maxima) < k
                    elif len(local_maxima) < k:
                        split_points = []
                        for i in range(len(local_maxima) - 1):
                            min_idx = (
                                np.argmin(
                                    self.processed_signal[
                                        local_maxima[i] : local_maxima[i + 1] + 1
                                    ]
                                )
                                + local_maxima[i]
                            )
                            split_points.append(min_idx)

                        # Create sub-regions for len(local_maxima)
                        shoulder_regions = split_regions(start, end, split_points)

                        # Step 2.b.2: Further split regions if conditions are met
                        for region_start, region_end in shoulder_regions:
                            d2_maxima = (
                                argrelmax(
                                    self.d2_signal[region_start : region_end + 1],
                                    order=10,
                                )[0]
                                + region_start
                            )
                            d2_minima = (
                                argrelmin(
                                    self.d2_signal[region_start : region_end + 1],
                                    order=10,
                                )[0]
                                + region_start
                            )

                            if len(d2_maxima) == 3 and len(d2_minima) >= 2:
                                # Split at the middle maximum
                                middle_max_idx = d2_maxima[1]
                                final_regions.append((region_start, middle_max_idx))
                                final_regions.append((middle_max_idx, region_end))
                            else:
                                final_regions.append((region_start, region_end))
                    else:
                        raise ValueError(
                            f"Region {region} has {len(local_minima)} local minima, but expected {k - 1}."
                        )

            # prev_start, prev_end = final_regions[0]

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

            trimmed_regions = []
            current_start, current_end = final_regions[0]
            for next_start, next_end in regions[1:]:
                if next_start <= current_end:  # Overlapping regions
                    rel_min_signal = get_extrema(
                        argrelmin,
                        self.smoothed_signal,
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
                    elif len(rel_max_d2) == 1:
                        update_start_end(rel_max_d2[0])
                        continue
                    elif len(rel_max_d2) == 3 and len(rel_min_d2) == 2:
                        update_start_end(rel_max_d2[1])
                        continue
                    elif len(rel_min_d2) == 1:
                        update_start_end(rel_min_d2[0])
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
            trimmed_regions.append([current_start, current_end])

            # for region in final_regions[1:]:
            #     curr_start, curr_end = region
            #     if curr_start < prev_end:
            #         local_max = (
            #             argrelmax(self.d2_signal[curr_start : prev_end + 1], order=10)[
            #                 0
            #             ]
            #             + curr_start
            #         )[0]
            #         trimmed_regions.append([prev_start, local_max])
            #         prev_start, prev_end = prev_start, local_max
            #         continue
            #     prev_start, prev_end = curr_start, curr_end
            #     trimmed_regions.append([prev_start, prev_end])

            return trimmed_regions

        expanded_regions = expand_region(initial_regions)

        # Step 3: Trim overlapping regions at local minima
        trimmed_regions = trim_regions(expanded_regions)

        regions = []
        for start, end in trimmed_regions:
            if end - start <= 1:
                continue
            else:
                regions.append([start + 1, end + 1])

        self.peaks: PeakList = PeakList(
            self.timepoints,
            self.raw_signal,
            self.smoothed_signal,
            self.baseline_spline,
            self.d2_signal,
            self.signal_sigma,
        )
        self.peaks.add_peaks(regions)

    def refine_peaks(self, height_cutoff=MINIMUM_HEIGHT, area_cutoff=MINIMUM_AREA):
        self.peaks.filter_peaks(height_cutoff, area_cutoff)

    def plot_peaks(self, smoothed=False, second_derivative=False, noise=False):
        # Plot the final result
        plt.figure(figsize=(14, 8))
        t = self.timepoints
        plt.plot(t, self.raw_signal, color="black")
        plt.plot(t, self.baseline_spline(t), color="grey")

        spline = self.baseline_spline(t)

        if smoothed:
            plt.plot(t, self.smoothed_signal + spline, color="limegreen")

        if second_derivative:
            plt.plot(t, self.d2_signal / self.dt + spline, color="blue")

        if noise:
            ones = np.ones_like(t)
            plt.plot(t, 2 * self.d2_sigma / self.dt * ones + spline, color="green")
            plt.plot(t, 2 * self.signal_sigma * ones + spline, color="red")
            plt.plot(t, -2 * self.d2_sigma / self.dt * ones + spline, color="green")
            plt.plot(t, -2 * self.signal_sigma * ones + spline, color="red")

        # Colors for adjacent peaks
        for idx, peak in enumerate(self.peaks):
            color = "orange" if idx % 2 == 0 else "blue"
            plt.axvspan(peak.start_time, peak.end_time, color=color, alpha=0.2)

        plt.xlabel("Time (min)")
        plt.ylabel("Absorbance (mAU)")
        plt.show()

    def __peak_list_to_dataframe(self) -> pd.DataFrame:
        peak_dict_list = []
        for peak in self.peaks:
            peak_dict_list.append(peak.get_properties())
        return pd.DataFrame.from_dict(peak_dict_list)

    def get_peaks(self, dataframe=False) -> PeakList:
        if dataframe:
            return self.__peak_list_to_dataframe()
        else:
            return self.peaks

    def print_peaks(self):
        print(self.get_peaks())

    def save_peaks(self, filename="output.csv"):
        df: pd.DataFrame = self.__peak_list_to_dataframe()
        df.to_csv(filename, index=False)
