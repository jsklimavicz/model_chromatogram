import numpy as np
from model_chromatogram.user_parameters import (
    SATURATION_SCALE,
    LINEAR_LIMIT,
    BASELINE_NOISE,
    BASELINE_AUTOCORRELATION_PARAMETER,
)

import matplotlib.pyplot as plt
import pandas as pd
from model_chromatogram.utils import create_autocorrelated_data
import uuid


class Chromatogram:
    """
    Creates and stores time and signal data for chromatograms.
    """

    def __init__(
        self, times: np.array, initial_values: np.array, wavelength: float
    ) -> None:
        """
        Creates a new Chromatogram.

        Args:
            times (np.array): array of times in minutes at which to generate chromatogram data.
            initial_values (np.array): array of initial signal values to add peaks to.
        """
        self.uuid = str(uuid.uuid4())
        self.times: np.array = times
        self.signal: np.array = initial_values
        self.saturated_signal: np.array = self.signal
        self.saturation_up_to_date = False
        self.wavelength = wavelength

    def add_compound_peak(self, signal: np.array, absorbance: float = 1) -> None:
        """
        Adds a compound peak to the chromatogram.

        Args:
            absorbance (float): Absorbance value for the peak. Acts as a multiplier for the signal. Default = 1.0
            signal (np.array):

        """
        if len(signal) != len(self.signal):
            raise ChromatogramError(
                "The provided signal does not match the length of the current chromatogram."
            )
        self.signal += absorbance * signal
        self.saturation_up_to_date = False

    def _detector_saturation(func):
        """
        Decorator function to create/update a saturated chromatogram signal. Detectors only have a linear relation
        between amount of analyte and peak area for a finite linear range, and at some point, the detector saturates.
        This decorator creates this saturation effect using the user-defined `LINEAR_LIMIT` and `SATURATION_SCALE`.
        """

        def _adjust_saturation(x):
            diff_x = x - LINEAR_LIMIT
            diff = SATURATION_SCALE * (
                1 - np.exp(-(diff_x / SATURATION_SCALE)) ** np.log(2)
            )
            return LINEAR_LIMIT + diff

        def adjust_signal(self, *args, **kwargs):
            if not self.saturation_up_to_date:
                self.saturated_signal = np.where(
                    self.signal < LINEAR_LIMIT,
                    self.signal,
                    _adjust_saturation(self.signal),
                )
                self.saturation_up_to_date = True
            return func(self, *args, **kwargs)

        return adjust_signal

    @_detector_saturation
    def plot(self, v_offset: float = 0, h_offset: float = 0, **kwargs):
        """
        Plots the chromatogram. When plotting multiple chromatograms on the same figure, it may be beneficial to use
        offsets so that baselines/peaks do not overlap.

        Args:
            v_offset (float): Amount to shift the chromatogram vertically, in the units of the y-axis (milli-arbitrary
            units).
            h_offset (float): Amount to shift the chromatogram horizontally, in the units of the x-axis (minutes).
        """
        plt.plot(self.times + h_offset, self.saturated_signal + v_offset, **kwargs)

    @_detector_saturation
    def get_chromatogram_data(self, pandas=False):
        """
        Returns a pandas DataFrame of the chromatogram data with column `"x"` as the x-axis and column `"y"` as the
        y-axis.

        Args:
            pandas (bool): If true, returns a pandas datatrame of x and y values. If false, returns a tuple with x and
            y np.arrays.

        Returns:
            out (pd.DataFrame|tuple(np.array)): DataFrame containing the chromatogram data.

        """
        if pandas:
            df = pd.DataFrame({"x": self.times, "y": self.saturated_signal})
            return df
        else:
            return self.times, self.saturated_signal


class Baseline(Chromatogram):
    """
    Class for creating a baseline.
    """

    def __init__(
        self, times, initial_values, wavelength: float, noise_level=BASELINE_NOISE
    ) -> None:
        """
        Creates a new baseline chromatogram.

        Args:
            times (np.array): array of times in minutes at which to generate chromatogram data.
            initial_values (np.array): array of initial signal values to add peaks to.
            noise_level (float): value specifying the noise level of the baseline. Roughly 95% of baseline values will
            be in the interval [`initial_values - noise_level`, `initial_values + noise_level`].
        """
        super().__init__(times, initial_values, wavelength)
        self.noise_level = noise_level
        self.create_baseline()

    def create_baseline(self):
        """
        Creates the baseline. Uses a method for generating baseline noise using an AR(1) model (1-parameter
        autoregressive model).
        """
        # to account for fact 95% of samples from normal distribution are within 1.96 std
        sigma = self.noise_level / 1.96
        corr = BASELINE_AUTOCORRELATION_PARAMETER
        noise = create_autocorrelated_data(len(self.signal), sigma, corr)
        self.signal += noise


class ChromatogramError(ValueError):
    """
    ValueError specifically for the Chromatogram Class.
    """

    pass
