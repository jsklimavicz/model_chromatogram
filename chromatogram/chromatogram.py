import numpy.random as random
import numpy as np
from user_parameters import *
import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Chromatogram:
    def __init__(self, times, inital_values) -> None:
        self.times = times
        self.signal = inital_values
        self.mean_values = inital_values
        self.saturation_filter = False

    def add_compound_peak(self, absorbance, signal):
        self.signal += absorbance * signal

    def _detector_saturation(func):
        def _adjust_saturation(x):
            val = LINEAR_LIMIT
            diff_x = x - val
            diff = SATURATION_SCALE * (
                1 - np.exp(-(diff_x / SATURATION_SCALE)) ** np.log(2)
            )
            return val + diff

        def adjust_signal(self, *args, **kwargs):
            if not self.saturation_filter:
                self.signal = np.where(
                    self.signal < LINEAR_LIMIT,
                    self.signal,
                    _adjust_saturation(self.signal),
                )
                self.saturation_filter = True
            return func(self, *args, **kwargs)

        return adjust_signal

    @_detector_saturation
    def plot(
        self, offset: float = 0, v_offset: float = 0, h_offset: float = 0, **kwargs
    ):
        if offset != 0:
            plt.plot(self.times + offset / (60), self.signal + offset, **kwargs)
        else:
            plt.plot(self.times + h_offset, self.signal + v_offset, **kwargs)

    @_detector_saturation
    def get_chromatogram_data(self):
        df = pd.DataFrame({"x": self.times, "y": self.signal})
        return df


class Baseline(Chromatogram):
    def __init__(self, times, inital_values, noise_level=BASELINE_NOISE) -> None:
        super().__init__(times, inital_values)
        self.noise_level = noise_level
        self.create_background()

    def __create_autocorrelated_data(self, sigma):
        """
        adapted from https://stackoverflow.com/a/33904277
        """
        corr = BASELINE_AUTOCORRELATION_PARAMETER
        assert (
            0 < corr < 1
        ), f"BASELINE_AUTOCORRELATION_PARAMETER must be set between 0 and 1, exclusive, but is set to {BASELINE_AUTOCORRELATION_PARAMETER}"

        c: np.array = self.mean_values * (1 - corr)
        eps = np.sqrt((sigma**2) * (1 - corr**2))
        signal = c + random.normal(loc=0, scale=eps, size=np.shape(c))
        signal[0] += -c[0] + self.mean_values[0]
        for ind in range(1, len(signal)):
            signal[ind] += corr * signal[ind - 1]
        return signal

    def create_background(self):
        """
        For generating 1-D backgrounds
        x_points: numpy array of time values
        noise_level: non-negative value for 95% confidence interval of noise level. If noise_level == 0, then no noise is added.
        mean_value: level around which noise will vary.
        autocorrelation_parameter: parameter for autocorrelation of background data. In reality, signal data from detectors is often autocorrelated.
        """
        # to account for fact 95% of samples from normal distribution are within 2 std
        sigma = self.noise_level / 2
        self.signal = self.__create_autocorrelated_data(sigma)

        return self.signal
