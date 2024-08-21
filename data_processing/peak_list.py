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
            self.peaks.loc[self.peaks["peak_type"] == "bb", "peak_type"] = "bMb"
            self.peaks.loc[self.peaks["peak_type"] == "MM", "peak_type"] = "M"

        def recalculate_globals(self, *args, **kwargs):
            func(self, *args, **kwargs)

            self.peaks = self.peaks.sort_values(by="index_start").reset_index(drop=True)
            _determine_peak_types(self, *args, **kwargs)

            def _get_baselined_signal(row):
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
                return baselined_signal

            def calculate_areas(row):
                baselined_signal = _get_baselined_signal(row)
                return np.sum(baselined_signal) * self.dt

            self.peaks["area"] = self.peaks.apply(calculate_areas, axis=1)
            self.peaks["relative_height"] = (
                100 * self.peaks["height"] / np.sum(self.peaks["height"])
            )
            self.peaks["relative_area"] = (
                100 * self.peaks["area"] / np.sum(self.peaks["area"])
            )

            def calculate_widths(row):
                baselined_signal = _get_baselined_signal(row)
                start = row["index_start"]
                peak_max = row["retention_index"]
                retention_time = row["time_retention"]
                fields = ["width_50", "width_10", "width_5"]
                vals = np.array([0.5, 0.1, 0.05])
                vals *= row["height"]
                return_dict = {}
                for val, field in zip(vals, fields):
                    try:
                        left_indices = (
                            np.where(baselined_signal[: peak_max - start] <= val)[0]
                            + start
                        )
                        y0 = self.signal[left_indices[-1]]
                        y1 = self.signal[left_indices[-1] + 1]
                        x0 = self.times[left_indices[-1]]
                        left_time = self.dt / (y1 - y0) * (val - y0) + x0
                        right_indices = (
                            np.where(baselined_signal[peak_max - start :] <= val)[0]
                            + peak_max
                        )
                        y0 = self.signal[right_indices[0] - 1]
                        y1 = self.signal[right_indices[0]]
                        x0 = self.times[right_indices[0] - 1]
                        right_time = self.dt / (y1 - y0) * (val - y0) + x0
                        left = retention_time - left_time
                        right = right_time - retention_time
                        total = right_time - left_time
                        return_dict[f"{field}_left"] = left
                        return_dict[f"{field}_right"] = right
                        return_dict[f"{field}_full"] = total
                    except IndexError as e:
                        for x in ["left", "right", "full"]:
                            return_dict[f"{field}_{x}"] = None
                return pd.Series(return_dict)

            new_columns: pd.Series = self.peaks.apply(calculate_widths, axis=1)
            print(new_columns)
            # Concatenate the new columns to the original DataFrame
            self.peaks = pd.concat(
                [self.peaks.drop(new_columns.columns, axis=1), new_columns], axis=1
            )

            def calculate_asymmetry(row, method="USP"):
                try:
                    if method == "USP":
                        return row["width_5_full"] / (2 * row["width_5_left"])

                    elif method == "AIA":
                        return row["width_10_right"] / row["width_10_left"]
                except:
                    return None

            self.peaks["asymmetry_USP"] = self.peaks.apply(calculate_asymmetry, axis=1)
            self.peaks["asymmetry_AIA"] = self.peaks.apply(
                calculate_asymmetry, method="AIA", axis=1
            )

        return recalculate_globals

    # @_recalculate_overall_peak_values
    def add_peaks(self, peak_indices_list: list[list[int]]) -> None:
        curr_peaks = []
        for peak_indices in peak_indices_list:
            start_ind, end_ind = peak_indices
            rt_ind, rt, peak_height, signal_height = (
                self.__get_retention_time_and_height(start_ind, end_ind)
            )
            base_val_start = self.baseline(self.times[start_ind])
            base_val_end = self.baseline(self.times[end_ind])
            peak_dict = {
                "index_start": start_ind,
                "retention_index": rt_ind,
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
                "width_50_full": None,
                "width_10_full": None,
                "width_5_full": None,
                "width_50_right": None,
                "width_10_right": None,
                "width_5_right": None,
                "width_50_left": None,
                "width_10_left": None,
                "width_5_left": None,
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
        return rt_ind, self.times[rt_ind], peak_height, signal_height

    def get_peaklist(self):
        return self.peaks
