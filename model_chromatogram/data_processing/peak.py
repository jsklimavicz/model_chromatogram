import numpy as np
from scipy.interpolate import CubicSpline
from model_chromatogram.lib import exponnorm
from scipy.stats import norm
from scipy.optimize import curve_fit, brentq


class Peak:
    field_list = [
        "index",
        "name",
        "start_time",
        "retention_time",
        "end_time",
        "area",
        "start_signal",
        "end_signal",
        "retention_signal",
        "start_baseline",
        "end_baseline",
        "retention_baseline",
        "amount",
        "amount_unit",
        "relative_area",
        "height",
        "relative_height",
        "peak_type",
        "width_50_full",
        "width_4_sigma_full",
        "width_10_full",
        "width_5_full",
        "width_5_sigma_full",
        "width_baseline_full",
        "extended_width_calculation",
        "width_50_left",
        "width_4_sigma_left",
        "width_10_left",
        "width_5_left",
        "width_5_sigma_left",
        "width_baseline_left",
        "width_50_right",
        "width_4_sigma_right",
        "width_10_right",
        "width_5_right",
        "width_5_sigma_right",
        "width_baseline_right",
        "capillary_electrophoresis_area",
        "moment_0",
        "moment_1",
        "moment_2",
        "moment_3",
        "moment_3_standardized",
        "moment_4",
        "moment_4_standardized",
        "standard_deviation",
        "signal_to_noise",
        "asymmetry_USP",
        "asymmetry_AIA",
        "asymmetry_moments",
        "skewness",
        "resolution_ep_next_main",
        "resolution_usp_next_main",
        "resolution_statistical_next_main",
        "resolution_ep_previous_main",
        "resolution_usp_previous_main",
        "resolution_statistical_previous_main",
        "resolution_ep_previous_identified",
        "resolution_usp_previous_identified",
        "resolution_statistical_previous_identified",
        "resolution_ep_next_identified",
        "resolution_usp_next_identified",
        "resolution_statistical_next_identified",
        "resolution_ep_reference",
        "resolution_usp_reference",
        "resolution_statistical_reference",
        "plates_USP",
        "plates_EP",
        "plates_statistical",
    ]

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
        self.__set_initial_properties_to_none()
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
        self.start_spline = float(self.baseline_spline(self.start_time))
        self.end_spline = float(self.baseline_spline(self.end_time))
        self.__initial_calculations()

    def __set_initial_properties_to_none(self):
        self.__dict__.update({field: None for field in self.field_list})

    def __initial_calculations(self):

        self.peak_times = self.times[self.start_index : self.end_index + 1]
        base_signals = np.interp(
            self.peak_times,
            (self.peak_times[0], self.peak_times[-1]),
            (self.start_spline, self.end_spline),
        )
        self.peak_signal = self.raw_signal[self.start_index : self.end_index + 1]
        self.baselined_peak_signal = self.peak_signal - base_signals

        self.__calculate_height_and_retention_time()
        self.__calculate_area()
        self.__calculate_statistical_moments()
        self.__calculate_standard_widths()
        self.__calculate_asymmetry()
        self.__calculate_theoretical_plates()

    def calculate_properties(
        self, prev_peak: "Peak|None" = None, next_peak: "Peak | None" = None
    ):
        self.__set_peak_type(prev_peak, next_peak)

        if self.start_peak_type.lower() == "b":
            self.start_baseline = self.start_signal
        else:
            self.start_baseline = self.start_spline

        if self.end_peak_type.lower() == "b":
            self.end_baseline = self.end_signal
        else:
            self.end_baseline = self.end_spline

        self.peak_times = self.times[self.start_index : self.end_index + 1]
        base_signals = np.interp(
            [*self.peak_times, self.retention_time],
            (self.peak_times[0], self.peak_times[-1]),
            (self.start_baseline, self.end_baseline),
        )
        self.peak_signal = self.raw_signal[self.start_index : self.end_index + 1]
        self.baselined_peak_signal = self.peak_signal - base_signals[:-1]
        self.retention_baseline = base_signals[-1]

        self.__calculate_height_and_retention_time()
        self.__calculate_area()
        # self.__calculate_statistical_moments()

    def get_properties(self):
        return_dict = {}
        for field in self.field_list:
            val = self.__dict__[field]
            if isinstance(val, int):
                return_dict[field] = val
            else:
                return_dict[field] = val
        return return_dict

    def __set_peak_type(self, prev_peak: "Peak", next_peak: "Peak"):
        """
        Calculates the peak type based on whether the next peak is well-separated or not.
        """

        if not prev_peak or (prev_peak.end_index < self.start_index):
            self.start_peak_type = "B"
        elif self.start_signal < self.start_spline + 2 * self.signal_noise:
            self.start_peak_type = "b"
        else:
            self.start_peak_type = "M"

        # peak end
        if not next_peak or (next_peak.start_index > self.end_index):
            self.end_peak_type = "B"
        elif self.end_signal < self.end_spline + 2 * self.signal_noise:
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
        self.relative_area = None
        self.capillary_electrophoresis_area = self.area / self.retention_time

    def __calculate_height_and_retention_time(self):
        self.height = np.max(self.baselined_peak_signal)
        self.relative_height = None
        self.retention_index = np.argmax(self.baselined_peak_signal) + self.start_index
        time_window = self.times[self.retention_index - 3 : self.retention_index + 3]
        signal_window = self.raw_signal[
            self.retention_index - 3 : self.retention_index + 3
        ]

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        params, _ = curve_fit(quadratic, time_window, signal_window)
        a, b, _ = params

        self.retention_time = -b / (2 * a)
        if not (self.start_time < self.retention_time < self.end_time):
            self.retention_time = self.times[self.retention_index]
            self.retention_signal = self.height
        else:
            self.retention_signal = quadratic(self.retention_time, *params)

        self.signal_to_noise = 2 * self.height / self.signal_noise

    def __calculate_statistical_moments(self):
        signal = np.abs(self.baselined_peak_signal)
        d_times = self.peak_times - self.retention_time
        self.moment_0 = np.sum(signal) * self.dt
        self.moment_1 = np.sum(self.peak_times * signal) * self.dt / self.area
        self.moment_2 = np.sum(d_times**2 * signal) * self.dt / self.moment_0
        self.standard_deviation = np.sqrt(self.moment_2)
        self.moment_3 = np.sum(d_times**3 * signal) * self.dt / self.area
        self.moment_3_standardized = self.moment_3 / self.moment_2 ** (3 / 2)
        self.moment_4 = np.sum(d_times**4 * signal) * self.dt / self.area
        self.moment_4_standardized = self.moment_4 / self.moment_2**2

    def _calculate_widths_exception(func):
        def calculate_widths(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (IndexError, ValueError, RuntimeError) as e:
                return None, None, None

        return calculate_widths

    @_calculate_widths_exception
    def __calculate_width_with_curve_fit(self, height):
        val = height * self.height / 100

        def y_shifted_emg(t):
            return self._exponnorm_curve(t) - val

        std = np.sqrt(self.moment_2)
        slack = 10 * std
        left = brentq(y_shifted_emg, self.retention_time - slack, self.retention_time)
        right = brentq(y_shifted_emg, self.retention_time, self.retention_time + slack)
        lw = self.retention_time - left
        rw = right - self.retention_time
        fw = right - left
        return lw, rw, fw

    @_calculate_widths_exception
    def __calculate_width_with_datapoints(self, height):
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
        return h * exponnorm(t, K=K, loc=loc, scale=scale)

    def __fit_EMG_curve(self, t_array, signal_array):
        def exponnorm_curve(t, h, K, loc, scale):
            return h * exponnorm(t, K=K, loc=loc, scale=scale)

        guess = self.standard_deviation
        self.curve_params, _ = curve_fit(
            exponnorm_curve,
            t_array,
            signal_array,
            p0=(self.height / 10, 0.01, self.retention_time + 0.02, guess),
            bounds=(
                [0, 0, self.retention_time - 20 * guess, guess / 3],
                [np.inf, 5, self.retention_time + 20 * guess, 5 * guess],
            ),
        )

    @_calculate_widths_exception
    def __calculate_width_at_baseline(self):

        self.__fit_EMG_curve(self.peak_times, self.baselined_peak_signal)
        t_list = np.linspace(self.start_time - self.dt, self.end_time + self.dt, 202)
        t_vals = self._exponnorm_curve(t_list)
        vals_d2 = np.diff(np.diff(t_vals))
        d2_spline = CubicSpline(
            t_list[1:-1],
            vals_d2,
        )
        r = d2_spline.roots(extrapolate=False)
        self.left_poi = r[0]
        self.right_poi = r[1]

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
            if m == 0:
                pass
            # Calculate the x-intercept
            return x0 - y0 / m

        left_time = find_x_intercept(times[0], vals[0], times[1], vals[1])
        right_time = find_x_intercept(times[2], vals[2], times[3], vals[3])

        lw = self.retention_time - left_time
        rw = right_time - self.retention_time
        fw = right_time - left_time
        return lw, rw, fw

    def __calculate_width(self, method, height, use_EGM_fit=False):
        if method == "sigma":
            zero_h = norm.pdf(0)
            height = 100 * norm.pdf(height / 2) / zero_h
        if use_EGM_fit:
            return self.__calculate_width_with_curve_fit(height=height)
        else:
            return self.__calculate_width_with_datapoints(height=height)

    def __calculate_standard_widths(self):

        self.width_50_left, self.width_50_right, self.width_50_full = (
            self.__calculate_width_with_datapoints(height=50)
        )

        if self.width_50_full is None:
            # not enough peak to find other peak widths or fit a peak. This is a rider.
            return

        # calculate baseline width, which also fits an EMG curve
        (
            self.width_baseline_left,
            self.width_baseline_right,
            self.width_baseline_full,
        ) = self.__calculate_width_at_baseline()

        if self.width_baseline_full:

            self.extended_width_calculation = []
            use_EGM_fit = False
            heights = [4, 10, 5, 5]
            methods = ["sigma", "percent", "percent", "sigma"]

            def _width_driver(method, height, use_EGM_fit=False):
                portions = ["left", "right", "full"]
                if method == "percent":
                    keys = [f"width_{height}_{portion}" for portion in portions]

                elif method == "sigma":
                    keys = [f"width_{height}_sigma_{portion}" for portion in portions]

                self.__dict__.update(
                    dict(
                        zip(
                            keys,
                            self.__calculate_width(
                                method=method, height=height, use_EGM_fit=use_EGM_fit
                            ),
                        )
                    )
                )

                if self.__dict__[keys[-1]] is None:
                    self.__dict__.update(
                        dict(
                            zip(
                                keys,
                                self.__calculate_width(
                                    method=method,
                                    height=height,
                                    use_EGM_fit=True,
                                ),
                            )
                        )
                    )
                    return True
                else:
                    return use_EGM_fit

            for method, height in zip(methods, heights):
                use_EGM_fit = _width_driver(method, height, use_EGM_fit=use_EGM_fit)
                if use_EGM_fit:
                    self.extended_width_calculation.append(f"{height}_{method}")

    def __calculate_asymmetry(self):
        if self.width_5_full and self.width_5_left:
            self.asymmetry_USP = self.width_5_full / (2 * self.width_5_left)
        if self.width_10_right and self.width_10_left:
            self.asymmetry_AIA = self.width_10_right / self.width_10_left
            self.skewness = (self.width_10_right + self.width_10_left) / (
                2 * self.width_10_left
            )
        self.asymmetry_moments = (self.moment_1 - self.retention_time) / np.sqrt(
            self.moment_2
        )

    def calculate_resolution(self, type_, reference: "Peak" = None):
        if reference and reference != self:
            d_rt = reference.retention_time - self.retention_time
            ep = usp = statistical = None
            if self.width_50_full and reference.width_50_full:
                ep = 1.18 * abs(d_rt / (self.width_50_full + reference.width_50_full))
            if self.width_baseline_full and reference.width_baseline_full:
                usp = 2 * abs(
                    d_rt / (self.width_baseline_full + reference.width_baseline_full)
                )
            if self.moment_2 and reference.moment_2:
                statistical = abs(
                    d_rt / (2 * (np.sqrt(reference.moment_2) + np.sqrt(self.moment_2)))
                )
            for name, value in zip(
                ["ep", "usp", "statistical"], [ep, usp, statistical]
            ):
                self.__dict__.update({f"resolution_{name}_{type_}": value})

    def __calculate_theoretical_plates(self):
        if self.width_50_full:
            self.plates_EP = 5.54 * (self.retention_time / self.width_50_full) ** 2
        if self.width_baseline_full:
            self.plates_USP = 16 * (self.retention_time / self.width_baseline_full) ** 2
        self.plates_statistical = self.retention_time**2 / self.moment_2
