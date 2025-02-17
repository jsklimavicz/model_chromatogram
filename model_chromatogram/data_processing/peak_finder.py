import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from model_chromatogram.methods import ProcessingMethod
from model_chromatogram.data_processing import Peak, PeakList  # , als_psalsa

from model_chromatogram.utils import als_psalsa, signal_smoothing, find_peaks
from model_chromatogram.user_parameters import (
    NOISE_THRESHOLD_MULTIPLIER,
    MINIMUM_HEIGHT,
    MINIMUM_AREA,
    BACKGROUND_NOISE_RANGE,
    LINEAR_LIMIT,
    PEAK_LIMIT,
    MINIMUM_HEIGHT_METHOD,
)
from pydash import get as get_, set_
from scipy.stats import linregress


class PeakFinder:
    """
    Class to find peaks in a noisy spectrum. Peaks are located using a smoothed spectrum and its second derivative.
    """

    def __init__(
        self,
        timepoints: np.array,
        signal: np.array,
        processing_method: ProcessingMethod,
        sample_introduction: dict,
        channel_name: str = None,
    ) -> None:
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
        self.processing_method = processing_method
        self.channel_name = channel_name
        self.sample_introduction = sample_introduction
        self.__parse_processing_method()
        self.timepoints = timepoints
        self.n_points = len(timepoints)
        self.raw_signal = signal
        self.processed_signal = np.copy(signal)
        self.dt = (timepoints[-1] - timepoints[0]) / (len(timepoints) - 1)
        self.sample_rate = (1 / self.dt) / 60
        self.__initial_peak_finding()

    def __parse_processing_method(self):
        def set_param(mapping, default):
            fetched_val = get_(self.processing_method.kwargs, mapping)
            if fetched_val is None:
                fetched_val = default
            return fetched_val

        self.bg_min = set_param(
            "detection_parameters.background_noise_range.minimum",
            BACKGROUND_NOISE_RANGE[0],
        )
        self.bg_max = set_param(
            "detection_parameters.background_noise_range.maximum",
            BACKGROUND_NOISE_RANGE[1],
        )
        self.min_area = set_param("detection_parameters.minimum_area", MINIMUM_AREA)
        self.min_height = set_param(
            "detection_parameters.minimum_height.value", MINIMUM_HEIGHT
        )
        self.peak_limit = set_param("detection_parameters.peak_limit", PEAK_LIMIT)
        self.min_height_method = set_param(
            "detection_parameters.minimum_height.type", MINIMUM_HEIGHT_METHOD
        )
        self.resolution_reference = set_param("resolution_reference", "prev")
        self.noise_threshold_multiplier = set_param(
            "detection_parameters.noise_threshold_multiplier",
            NOISE_THRESHOLD_MULTIPLIER,
        )

    def __initial_peak_finding(self):
        """
        Internal method to drive the functions outlined in __init__.
        """
        self.__smooth_signal()
        self.__find_baseline()
        self.__apply_baseline()
        self.__get_derivatives()
        self.find_peaks()
        self.refine_peaks()

    def __signal_smoothing(
        self, signal, min_window=5, max_window=41, var_window_size=31, k=7.5
    ):
        """
        Smooths the signal using a variable window size Savitzky-Golay filter.

        Args:
            signal (np.array): The raw signal to be smoothed.
            min_window (int): Minimum window size for the Savitzky-Golay filter.
            max_window (int): Maximum window size for the Savitzky-Golay filter.
            var_window_size (int): Window size for calculating local variance.
            k (float): Scaling factor for the noise threshold.
        """

        return signal_smoothing(
            signal,
            bg_min=self.bg_min,
            bg_max=self.bg_max,
            min_window=min_window,
            max_window=max_window,
            var_window_size=var_window_size,
            k=k,
        )

    def __smoothing_driver(self, signal, min_window=5, max_window=41, deriv=0):
        if deriv == 0:
            return self.__signal_smoothing(
                signal,
                min_window=min_window,
                max_window=max_window,
                var_window_size=31,
            )
        else:
            return savgol_filter(signal, 31, 2, deriv=deriv)

    def __smooth_signal(self):
        self.smoothed_signal = self.__smoothing_driver(self.raw_signal)

    def __find_baseline(self):
        """
        Calculates a baseline approximation using the psalsa asymmetric least squares algorithm, and stores a cubic
        spline of the baseline.
        """

        baseline_params = get_(
            self.processing_method.kwargs, "detection_parameters.baseline"
        )
        baseline_time, baseline_vals = als_psalsa(
            self.timepoints, self.smoothed_signal, **baseline_params
        )
        self.baseline_spline = CubicSpline(
            baseline_time, baseline_vals, bc_type="natural"
        )

    def __apply_baseline(self):
        """
        Applies the als-psalsa baseline to the processed and smoothed signals.
        """
        self.processed_signal -= self.baseline_spline(self.timepoints)
        self.smoothed_signal -= self.baseline_spline(self.timepoints)
        self.signal_noise = np.sqrt(
            np.mean(self.processed_signal[self.bg_min : self.bg_max] ** 2)
        )

    def __get_derivatives(self):
        """
        Calculates the first and second derivative of the smoothed signal via the savgol filter.
        """

        self.d1_signal = self.__smoothing_driver(
            self.smoothed_signal, min_window=31, max_window=32, deriv=1
        )

        self.d2_signal = self.__smoothing_driver(
            self.smoothed_signal, min_window=31, max_window=32, deriv=2
        )

        self.d2_ave_noise = np.sqrt(
            np.mean(self.d2_signal[self.bg_min : self.bg_max]) ** 2
        )

    def find_peaks(self):
        """
        Works in three steps:
            1. find regions when the second derivative of the smoothed signal drops below a negative threshold multiple
                of the noise in the second derivative
            2. expand these regions to where the second derivative goes above positive multiple of the threshold and
                back down again
            3. contract overlapping regions to a local max or min in the second derivative
        This function populates the `self.peaks` array.

        Args:
            noise_multiplier (float): multiplier for threshold of signal detection. Higher values result in fewer peaks
                detected.

        """

        final_regions = find_peaks(
            self.d2_signal,
            self.processed_signal,
            self.d1_signal,
            self.raw_signal,
            self.noise_threshold_multiplier,
            self.d2_ave_noise,
            self.signal_noise,
            self.peak_limit,
            LINEAR_LIMIT,
        )

        self.peaks: PeakList = PeakList(
            self.timepoints,
            self.raw_signal,
            self.smoothed_signal,
            self.baseline_spline,
            self.d2_signal,
            self.signal_noise,
        )
        self.peaks.add_peaks(final_regions)

    def refine_peaks(self):
        if self.min_height_method in ["baseline_noise_multiplier", "multiplier"]:
            self.min_height *= self.signal_noise

        self.peaks.filter_peaks(self.min_height, self.min_area)
        self.name_peaks()
        self.peaks.calculate_peak_properties(
            globals=True, resolution_reference=self.resolution_reference
        )

    def plot_peaks(
        self,
        smoothed=False,
        first_derivative=False,
        second_derivative=False,
        noise=False,
        highlight_peaks=False,
        show_spline=False,
    ):
        # Plot the final result
        plt.figure(figsize=(14, 8))
        t = self.timepoints
        plt.plot(t, self.raw_signal, color="black")

        spline = self.baseline_spline(t)

        if show_spline:
            plt.plot(t, spline, color="grey")

        if smoothed:
            plt.plot(t, self.smoothed_signal + spline, color="limegreen")

        if first_derivative:
            plt.plot(t, self.d1_signal / np.sqrt(self.dt) + spline, color="slateblue")

        if second_derivative:
            plt.plot(t, self.d2_signal / self.dt + spline, color="blue")

        if noise:
            ones = np.ones_like(t)
            plt.plot(t, 2 * self.d2_sigma / self.dt * ones + spline, color="green")
            plt.plot(t, self.signal_sigma * ones + spline, color="red")
            plt.plot(t, -2 * self.d2_sigma / self.dt * ones + spline, color="green")
            plt.plot(t, -self.signal_sigma * ones + spline, color="red")

        # Colors for adjacent peaks
        if highlight_peaks:
            for idx, peak in enumerate(self.peaks):
                color = "orange" if idx % 2 == 0 else "blue"
                plt.axvspan(peak.start_time, peak.end_time, color=color, alpha=0.2)

        plt.xlabel("Time (min)")
        plt.ylabel("Absorbance (mAU)")
        return plt

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
        print(self.get_peaks(dataframe=True))

    def save_peaks(self, filename="output.csv"):
        df: pd.DataFrame = self.__peak_list_to_dataframe()
        df.to_csv(filename, index=False)

    def __iter__(self):
        return iter(self.peaks)

    def __getitem__(self, index):
        return self.peaks[index]

    def name_peaks(self):
        peak_names = get_(self.processing_method, "peak_identification")

        for identification in peak_names:
            min_time = identification["min_time"]
            max_time = identification["max_time"]
            name = identification["name"]
            calibration_sets = identification.get("calibration", None)
            method = identification["method"].lower()

            # Find the largest peak within the specified time range
            filtered_peaks: list[Peak] = [
                peak
                for peak in self.peaks
                if min_time <= peak.retention_time <= max_time
            ]

            if filtered_peaks:
                if method == "largest":
                    named_peak = max(filtered_peaks, key=lambda p: p.area)
                elif method in ["first", "earliest"]:
                    named_peak = min(filtered_peaks, key=lambda p: p.retention_time)
                elif method in ["last", "latest"]:
                    named_peak = max(filtered_peaks, key=lambda p: p.retention_time)
                else:
                    break

                named_peak.name = name

                # If calibration data is present, calculate the amount
                if calibration_sets:
                    for calibration in calibration_sets:
                        if calibration["channel"] == self.channel_name:
                            try:
                                areas = [
                                    point["area"] for point in calibration["points"]
                                ]
                                amounts = [
                                    point["amount"] for point in calibration["points"]
                                ]
                                if areas is None or amounts is None:
                                    continue
                            except Exception:
                                continue

                            calibration_fit(
                                calibration,
                                named_peak,
                                self.sample_introduction,
                            )


def calibration_fit(calibration, peak: Peak, sample_introduction):
    areas = [point["area"] for point in calibration["points"]]
    amounts = [point["amount"] for point in calibration["points"]]
    if calibration["type"] == "linear":
        fit_dict = lin_fit(areas, amounts, peak)
    else:
        fit_dict = {}

    for key, value in fit_dict.items():
        set_(calibration, key, value)
    peak.amount *= (
        sample_introduction["dilution_factor"] / sample_introduction["injection_volume"]
    )
    peak.amount_unit = calibration["amount_unit"]


def lin_fit(areas, amounts, peak: Peak):
    slope, intercept, rvalue, _, _ = linregress(areas, amounts)
    fit_dict = {
        "coefficient_A": slope,
        "coefficient_B": intercept,
        "formula": "Ax+B",
        "rvalue": rvalue,
    }
    peak.amount = slope * peak.area + intercept
    return fit_dict
