from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np


class PeakList:
    def __init__(
        self,
        times,
        raw_chromatogram,
        smoothed_chromatorgram,
        spline: CubicSpline,
        signal_noise,
    ) -> None:
        self.peaks: pd.DataFrame = pd.DataFrame()
        self.signal = raw_chromatogram
        self.smoothed = smoothed_chromatorgram
        self.times = times
        self.baseline: CubicSpline = spline
        self.noise = signal_noise
        self.dt = times[1] - times[0]

    # TODO implement sorting for peaks after each addition
    # TODO class methods for percent height and area
    # TODO peak width methods; undefined for some peak types
    # TODO resolution methods
    # TODO asymmetry methods
    # TODO plate count methods
    # TODO peak baseline type: baseline, adjacent peak, etc

    def _recalculate_overall_peak_values(func):
        def _determine_peak_types(self, *args, **kwargs):

            # Initialize the start_type and end_type columns
            self.peaks["peak_type_start"] = "B"  # Default to 'M'
            self.peaks["peak_type_end"] = "B"  # Default to 'M'

            # Iterate through the DataFrame to set start_type and end_type
            for i in range(1, len(self.peaks)):
                idx_start = self.peaks.at[i, "index_start"]
                idx_end = self.peaks.at[i, "index_end"]
                if idx_start <= self.peaks.at[i - 1, "index_end"]:
                    if (
                        self.peaks.at[i, "baseline_start"] + self.noise
                        > self.smoothed[idx_start]
                    ):
                        self.peaks.at[i, "peak_type_start"] = "b"
                    else:
                        self.peaks.at[i, "peak_type_start"] = "M"
                if (
                    i < len(self.peaks) - 1
                    and idx_end >= self.peaks.at[i + 1, "index_start"]
                ):
                    if (
                        self.peaks.at[i, "baseline_end"] + self.noise
                        > self.smoothed[idx_end]
                    ):
                        self.peaks.at[i, "peak_type_end"] = "b"
                    else:
                        self.peaks.at[i, "peak_type_end"] = "M"

            self.peaks["peak_type"] = (
                self.peaks["peak_type_start"] + self.peaks["peak_type_end"]
            )
            self.peaks.loc[self.peaks["peak_type"] == "BB", "peak_type"] = "BMB"
            self.peaks.loc[self.peaks["peak_type"] == "bB", "peak_type"] = "bMB"
            self.peaks.loc[self.peaks["peak_type"] == "Bb", "peak_type"] = "BMb"
            self.peaks.loc[self.peaks["peak_type"] == "MM", "peak_type"] = "M"

        def recalculate_globals(self, *args, **kwargs):
            func(self, *args, **kwargs)

            self.peaks = self.peaks.sort_values(by="index_start").reset_index(drop=True)
            _determine_peak_types(self, *args, **kwargs)

            def calculate_areas(row):
                start = row["index_start"]
                end = row["index_end"] + 1
                signals = self.signal[start:end]
                if row["peak_type_start"] == "M":
                    baseline_start = self.baseline(self.times[start])
                else:
                    baseline_start = row["signal_start"]

                if row["peak_type_end"] == "M":
                    baseline_end = self.baseline(self.times[end])
                else:
                    baseline_end = row["signal_end"]

                baselined_signal = signals - np.interp(
                    self.times[start:end],
                    [self.times[start], self.times[end]],
                    [baseline_start, baseline_end],
                )
                return np.sum(baselined_signal) * self.dt

            self.peaks["area"] = self.peaks.apply(calculate_areas, axis=1)
            self.peaks["relative_height"] = (
                100 * self.peaks["height"] / np.sum(self.peaks["height"])
            )
            self.peaks["relative_area"] = (
                100 * self.peaks["area"] / np.sum(self.peaks["area"])
            )

        return recalculate_globals

    @_recalculate_overall_peak_values
    def add_peaks(self, peak_indices_list: list[list[int]]) -> None:
        curr_peaks = []
        for peak_indices in peak_indices_list:
            start_ind, end_ind = peak_indices
            rt, peak_height, signal_height = self.__get_retention_time_and_height(
                start_ind, end_ind
            )
            base_val_start = self.baseline(self.times[start_ind])
            base_val_end = self.baseline(self.times[end_ind])
            peak_dict = {
                "index_start": start_ind,
                "index_end": end_ind,
                "time_start": self.times[start_ind],
                "time_retention": rt,
                "time_end": self.times[end_ind],
                "width_baseline": self.times[end_ind] - self.times[start_ind],
                "signal_start": self.signal[start_ind],
                "signal_end": self.signal[end_ind],
                "signal_retention_time": signal_height,
                "baseline_start": base_val_start,
                "baseline_end": base_val_end,
                "height": peak_height,
                "relative_height": None,
                "area": None,
                "relative_area": None,
                "peak_type_start": "",
                "peak_type_end": "",
                "peak_type": "",
                "width_50": None,
                "width_10": None,
                "width_5": None,
                "asymmetry_10": None,
                "asymmetry_5": None,
                "resolution": None,
                "plates": None,
            }
            curr_peaks.append(peak_dict)
        curr_peaks_df = pd.DataFrame(curr_peaks)

        self.peaks = pd.concat(
            [self.peaks, curr_peaks_df], join="outer", ignore_index=True
        )

    @_recalculate_overall_peak_values
    def refine_peaks(self, height_cutoff, area_cutoff):
        self.peaks = self.peaks.drop(
            self.peaks[
                (self.peaks["height"] < height_cutoff)
                | (self.peaks["area"] < area_cutoff)
            ].index
        )

    def __get_retention_time_and_height(self, start_ind, end_ind):
        rt_ind = np.argmax(self.smoothed[start_ind:end_ind]) + start_ind
        peak_height = self.smoothed[rt_ind]
        signal_height = self.signal[rt_ind]
        return self.times[rt_ind], peak_height, signal_height

    def get_peaklist(self):
        return self.peaks
