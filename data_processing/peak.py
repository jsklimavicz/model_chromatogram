import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import exponnorm
from scipy.optimize import curve_fit


class Peak:

    def __init__(
        self,
        start_index,
        end_index,
        times,
        raw_chromatogram,
        smoothed_chromatogram,
        baseline_spline,
        second_derivative,
        signal_noise,
    ) -> None:
        self.start_index: int = start_index
        self.end_index: int = end_index
        self.times: np.array = times
        self.raw_signal: np.array = raw_chromatogram
        self.smoothed_chromatogram: np.array = smoothed_chromatogram
        self.baseline_spline: np.array = baseline_spline
        self.d2_signal = second_derivative
        self.signal_noise = signal_noise
        self.dt = (self.times[-1] - self.times[0]) / len(self.times)
        self.n_data_points = end_index - start_index + 1
        self.start_signal = raw_chromatogram[start_index]
        self.end_signal = raw_chromatogram[end_index]
        self.start_time = times[start_index]
        self.end_time = times[end_index]

    def calculate_properties(
        self, prev_peak: "Peak|None" = None, next_peak: "Peak | None" = None
    ):
        self.__set_peak_type(prev_peak, next_peak)

        if self.start_peak_type.lower() == "b":
            start_y = self.start_signal
        else:
            start_y = self.baseline_spline(self.start_index)

        if self.start_peak_type.lower() == "b":
            end_y = self.end_signal
        else:
            end_y = self.baseline_spline(self.end_index)

        self.peak_times = self.times[self.start_index : self.end_index + 1]
        base_signals = np.interp(
            self.peak_times, (self.peak_times[0], self.peak_times[-1]), (start_y, end_y)
        )
        self.peak_signal = self.raw_signal[self.start_index : self.end_index + 1]
        self.baselined_peak_signal = self.peak_signal - base_signals

        self.__calculate_height_and_retention_time()
        self.__calculate_area()
        self.__calculate_standard_widths()
        self.__calculate_statistical_moments()
        self.__calculate_asymmetry()
        self.__calculate_resolution()

    def get_properties(self):
        self.dict = {
            "start_time": self.start_time,
            "retention_time": self.retention_time,
            "end_time": self.end_time,
            "area": self.area,
            "relative_area": self.relative_area,
            "height": self.height,
            "relative_height": self.relative_height,
            "peak_type": self.peak_type,
            "width_50_full": self.width_50_full,
            "width_50_left": self.width_50_left,
            "width_50_right": self.width_50_right,
            "width_10_full": self.width_10_full,
            "width_10_left": self.width_10_left,
            "width_10_right": self.width_10_right,
            "width_5_full": self.width_5_full,
            "width_5_left": self.width_5_left,
            "width_5_right": self.width_5_right,
            "width_baseline_full": self.width_baseline_full,
            "width_baseline_left": self.width_baseline_left,
            "width_baseline_right": self.width_baseline_right,
            "asymmetry_USP": self.asymmetry_USP,
            "asymmetry_AIA": self.asymmetry_AIA,
        }

    def __set_peak_type(self, prev_peak: "Peak", next_peak: "Peak"):
        """
        Calculates the peak type based on whether the next peak is well-separated or not.
        """
        # peak start
        if not prev_peak or (prev_peak.end_index < self.start_index):
            self.start_peak_type = "B"
        elif (
            self.start_signal
            < self.baseline_spline[self.start_index] + 2 * self.signal_noise
        ):
            self.start_peak_type = "b"
        else:
            self.start_peak_type = "M"

        # peak end
        if not next_peak or (next_peak.start_index > self.end_index):
            self.end_peak_type = "B"
        elif (
            self.end_signal
            < self.baseline_spline[self.end_index] + 2 * self.signal_noise
        ):
            self.end_peak_type = "b"
        else:
            self.end_peak_type = "M"

        # overall peaks
        self.peak_type = self.start_peak_type + self.end_peak_type
        mapping = {"BB": "BMB", "Bb": "BMb", "bB": "bMB", "bb": "bMb", "MM": "M"}
        if self.peak_type in mapping.keys():
            self.peak_type = mapping[self.peak_type]

    def __calculate_area(self):
        self.area = np.sum(self.baselined_peak_signal) * self.dt

    def __calculate_height_and_retention_time(self):
        self.height = np.max(self.baselined_peak_signal)
        self.retention_index = np.argmax(self.baselined_peak_signal) + self.start_index
        self.retention_time = self.times[self.retention_index]

    def __calculate_width_at_height(self, height):
        val = height * self.height / 100
        midpoint = self.retention_index - self.start_index
        l_below = np.where(self.baselined_peak_signal[:midpoint] < val)[0]
        l_below = l_below[-1]
        r_below = np.where(self.baselined_peak_signal[midpoint:] < val)[0] + midpoint
        r_below = r_below[0]
        l_y0 = self.baselined_peak_signal[l_below]
        l_x0 = self.peak_times[l_below]
        l_y1 = self.baselined_peak_signal[l_below + 1]
        l_x1 = self.peak_times[l_below + 1]

        left_time = np.interp(val, (l_y0, l_y1), (l_x0, l_x1))

        r_y0 = self.baselined_peak_signal[r_below - 1]
        r_x0 = self.peak_times[r_below - 1]
        r_y1 = self.baselined_peak_signal[r_below]
        r_x1 = self.peak_times[r_below]

        right_time = np.interp(val, (r_y0, r_y1), (r_x0, r_x1))

        lw = self.retention_time - left_time
        rw = right_time - self.retention_time
        fw = right_time - left_time
        return lw, rw, fw

    def _exponnorm_curve(self, t):
        h, K, loc, scale = self.curve_params
        return h * exponnorm.pdf(t, K=K, loc=loc, scale=scale)

    def __fit_EMG_curve(self, t_array, signal_array):
        def exponnorm_curve(t, h, K, loc, scale):
            return h * exponnorm.pdf(t, K=K, loc=loc, scale=scale)

        self.curve_params, _ = curve_fit(
            exponnorm_curve,
            t_array,
            signal_array,
            p0=(self.height / 5, 1.01, self.retention_time, self.width_50_full),
        )

    def __calculate_width_at_baseline(self):
        d2_spline = CubicSpline(
            self.peak_times,
            self.d2_signal[self.start_index : self.end_index + 1],
        )
        r = d2_spline.roots(extrapolate=False)
        self.left_poi = r[0]
        self.right_poi = r[1]
        self.__fit_EMG_curve(self.peak_times, self.baselined_peak_signal)
        times = [
            self.left_poi - self.dt / 2,
            self.left_poi + self.dt / 2,
            self.right_poi - self.dt / 2,
            self.right_poi + self.dt / 2,
        ]
        vals = self._exponnorm_curve(times)

        def find_x_intercept(x0, y0, x1, y1):
            # Calculate the slope
            m = (y1 - y0) / (x1 - x0)
            # Calculate the x-intercept
            return x0 - y0 / m

        left_time = find_x_intercept(times[0], vals[0], times[1], vals[1])
        right_time = find_x_intercept(times[2], vals[2], times[3], vals[3])

        lw = self.retention_time - left_time
        rw = right_time - self.retention_time
        fw = right_time - left_time
        return lw, rw, fw

    def __calculate_standard_widths(self):
        self.width_50_left, self.width_50_right, self.width_50_full = (
            self.__calculate_width_at_height(height=50)
        )
        self.width_10_left, self.width_10_right, self.width_10_full = (
            self.__calculate_width_at_height(height=10)
        )
        self.width_5_left, self.width_5_right, self.width_5_full = (
            self.__calculate_width_at_height(height=5)
        )
        (
            self.width_baseline_left,
            self.width_baseline_right,
            self.width_baseline_full,
        ) = self.__calculate_width_at_baseline()

    def __calculate_statistical_moments(self):
        pass

    def __calculate_asymmetry(self):
        self.asymmetry_USP = None
        self.asymmetry_AIA = None
        if self.width_5_full and self.width_5_left:
            self.asymmetry_USP = self.width_5_full / (2 * self.width_5_left)
        if self.width_10_right and self.width_10_left:
            self.asymmetry_USP = self.width_10_right / self.width_10_left

    def __calculate_resolution(self, reference: "Peak" = None):
        pass

    def __calculate_theoretical_plates(self):
        self.plates_EP = None
        self.plates_USP = None
        self.plates_statistical_moments = None
