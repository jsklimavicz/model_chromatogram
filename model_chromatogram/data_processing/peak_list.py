from scipy.interpolate import CubicSpline
from model_chromatogram.data_processing import Peak
from model_chromatogram.user_parameters import MINIMUM_AREA, MINIMUM_HEIGHT


class PeakList:
    def __init__(
        self,
        times,
        raw_chromatogram,
        smoothed_chromatorgram,
        spline: CubicSpline,
        second_derivative,
        signal_noise,
    ) -> None:
        self.peaks: list[Peak] = []
        self.signal = raw_chromatogram
        self.smoothed = smoothed_chromatorgram
        self.times = times
        self.baseline: CubicSpline = spline
        self.second_derivative = second_derivative
        self.noise = signal_noise
        self.dt = times[1] - times[0]

    def __getitem__(self, index):
        if index < 0 or index >= len(self.peaks):
            raise PeakListIndexError(
                f"Index {index} is out of range for PeaksList of size {len(self.peaks)}."
            )
        return self.peaks[index]

    def __iter__(self):
        return iter(self.peaks)

    def add_peaks(self, peak_indices_list: list[list[int]]) -> None:
        for peak_indices in peak_indices_list:
            start_index, end_index = peak_indices
            self.add_peak(start_index, end_index)

    def add_peak(self, start_index: int, end_index: int) -> None:
        curr_peak = Peak(
            start_index,
            end_index,
            times=self.times,
            raw_chromatogram=self.signal,
            smoothed_chromatogram=self.smoothed,
            baseline_spline=self.baseline,
            second_derivative=self.second_derivative,
            signal_noise=self.noise,
        )
        self.peaks.append(curr_peak)

    def calculate_peak_properties(
        self, globals=True, resolution_reference="prev"
    ) -> None:
        """
        Drives the calculation of peak properties that are based on the values other peaks, like relative area and
        resolution.

        Args:
            globals (bool): If true (default), calculates the relative area and relative height for each peak.
            resolution_reference (None|int|str): A reference determining which peak to use for calculating the
                resolution of each peak. Permitted values are:
                    "prev": Resolution for a peak is calculated using the previous peak as a reference. Undefined for
                        the first peak.
                    "next": Resolution for a peak is calculated using the next peak as a reference. Undefined for the
                        last peak.
                    1 < int <= n_peaks: For an `int` value between 1 and the number of peaks, the 1-indexed index of the
                        peak to use as a reference for resolution. Resolution is undefined for that peak.
        """
        self.peaks.sort(key=lambda peak: peak.retention_time)

        # calculate properties and set index

        prev_peak = next_peak = None
        for ind, peak in enumerate(self.peaks):
            if ind > 0:
                prev_peak = self.peaks[ind - 1]
            if ind < len(self.peaks) - 1:
                next_peak = self.peaks[ind + 1]
            else:
                next_peak = None
            peak.calculate_properties(prev_peak=prev_peak, next_peak=next_peak)
            peak.index = ind + 1

        # calculate relative area and height
        if globals:
            total_area = sum(peak.area for peak in self.peaks)
            total_amount = sum(peak.amount if peak.amount else 0 for peak in self.peaks)
            total_height = sum(peak.height for peak in self.peaks)
            for peak in self.peaks:
                peak.relative_area = 100 * peak.area / total_area
                peak.relative_height = 100 * peak.height / total_height
                try:
                    peak.relative_amount = 100 * peak.amount / total_amount
                except Exception:
                    pass

        # calculate resolution

        def get_next_main(peak):
            index = self.peaks.index(peak)
            if index < len(self.peaks) - 1:
                return self.peaks[index + 1]
            return None

        def get_previous_main(peak):
            index = self.peaks.index(peak)
            if index > 0:
                return self.peaks[index - 1]
            return None

        def get_next_identified(peak):
            index = self.peaks.index(peak)
            for next_peak in self.peaks[index + 1 :]:
                if next_peak.name not in ["", None, "unknown"]:
                    return next_peak
            return None

        def get_previous_identified(peak):
            index = self.peaks.index(peak)
            for prev_peak in reversed(self.peaks[:index]):
                if prev_peak.name not in ["", None, "unknown"]:
                    return prev_peak
            return None

        def get_reference_peak():
            for peak in self.peaks:
                if peak.name == resolution_reference:
                    return peak
            return None

        for peak in self.peaks:
            ref_peak_dic = {
                "next_main": get_next_main(peak),
                "previous_main": get_previous_main(peak),
                "next_identified": get_next_identified(peak),
                "previous_identified": get_previous_identified(peak),
                "reference": get_reference_peak(),
            }
            for key, ref_peak in ref_peak_dic.items():
                peak.calculate_resolution(key, ref_peak)

    def filter_peaks(
        self,
        min_height=MINIMUM_HEIGHT,
        min_area=MINIMUM_AREA,
    ) -> None:
        peaks = []
        for peak in self.peaks:
            if peak.area >= min_area and peak.height >= min_height:
                peaks.append(peak)
        self.peaks = peaks


class PeakListIndexError(IndexError):
    def __init__(self, message="Index out of range in the PeaksList"):
        super().__init__(message)


class PeakListValueError(ValueError):
    def __init__(self, message="PeakList value could not be cast to specified type."):
        super().__init__(message)
