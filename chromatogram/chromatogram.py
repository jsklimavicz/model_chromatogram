import numpy.random as random
import numpy as np
from user_parameters import *
import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
from chromatogram.peakcreator import PeakCreator
from compounds.compound import Compound


class Chromatogram:
    def __init__(self, times, inital_values) -> None:
        self.times = times
        self.signal = inital_values
        self.mean_values = inital_values
        self.saturation_filter = False

    def plot(
        self, offset: float = 0, v_offset: float = 0, h_offset: float = 0, **kwargs
    ):
        self.__detector_saturation()
        if offset != 0:
            plt.plot(self.times + offset / (60), self.signal + offset, **kwargs)
        else:
            plt.plot(self.times + h_offset, self.signal + v_offset, **kwargs)

    def add_compound_peak(
        self, peak_creator: PeakCreator, compound: Compound, absorbance: float
    ):
        self.signal += peak_creator.compound_peak(compound, absorbance, self.times)

    def __detector_saturation(self):
        if self.saturation_filter:
            return

        def _adjust_saturation(x):
            if x < LINEAR_LIMIT:
                return x
            else:
                val = LINEAR_LIMIT
                diff_x = x - val
                diff = SATURATION_SCALE * (
                    1 - np.exp(-(diff_x / SATURATION_SCALE)) ** np.log(2)
                )
                return val + diff

        self.signal = np.array([_adjust_saturation(x) for x in self.signal])
        self.saturation_filter = True


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
        c = self.mean_values * (1 - corr)
        eps = np.sqrt((sigma**2) * (1 - corr**2))
        signal = np.zeros_like(self.mean_values)
        signal[0] = c[0] + random.normal(0, eps)
        for ind in range(1, len(signal)):
            signal[ind] = c[ind] + corr * signal[ind - 1] + random.normal(0, eps)
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
