import numpy as np


class Peak:

    def __init__(
        self,
        start_index,
        end_index,
        times,
        raw_chromatogram,
        smoothed_chromatogram,
        baseline_spline,
        signal_noise,
    ) -> None:
        self.start_index: int = start_index
        self.end_index: int = end_index
        self.times: np.array = times
        self.raw_signal: np.array = raw_chromatogram
        self.smoothed_chromatogram: np.array = smoothed_chromatogram
        self.baseline_spline: np.array = baseline_spline
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
        self._set_peak_type(prev_peak, next_peak)

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

        self._calculate_height_and_retention_time()
        self._calculate_area()
        self._calculate_standard_widths()

    def _set_peak_type(self, prev_peak: "Peak", next_peak: "Peak"):
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

    def _calculate_area(self):
        self.area = np.sum(self.baselined_peak_signal) * self.dt

    def _calculate_height_and_retention_time(self):
        self.height = np.max(self.baselined_peak_signal)
        self.retention_index = np.argmax(self.baselined_peak_signal) + self.start_index
        self.retention_time = self.times[self.retention_index]

    def _calculate_width(self, height):
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

    def _calculate_standard_widths(self):
        self.width_50_left, self.width_50_right, self.width_50_full = (
            self._calculate_width(height=50)
        )
        self.width_10_left, self.width_10_right, self.width_10_full = (
            self._calculate_width(height=10)
        )
        self.width_5_left, self.width_5_right, self.width_5_full = (
            self._calculate_width(height=5)
        )

    def _calculate_asymmetry(self):
        pass

    def _calculate_resolution(self, reference: "Peak"):
        pass
