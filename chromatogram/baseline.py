import numpy.random as random
import numpy as np
from user_parameters import *
import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np


class Baseline:
    def __init__(self, noise_level=1, mean_value=1) -> None:
        """
        noise_level: non-negative value for 95% confidence interval of noise level. If noise_level == 0, then no noise is added.
        mean_value: level around which noise will vary.
        """
        self.noise_level = noise_level
        self.mean_value = mean_value

    def create_autocorrelated_data(self, num_samples, mean, sigma, corr):
        """
        adapted from https://stackoverflow.com/a/33904277
        """

        assert 0 < corr < 1
        c = mean * (1 - corr)
        eps = np.sqrt((sigma**2) * (1 - corr**2))
        signal = np.zeros(num_samples)
        signal[0] = c + random.normal(0, eps)
        for ind in range(1, num_samples):
            signal[ind] = c + corr * signal[ind - 1] + random.normal(0, eps)

        return signal

    def create_background(
        self,
        n_points,
        noise_level=1,
        mean=1,
        autocorrelation_parameter=BASELINE_AUTOCORRELATION_PARAMETER,
    ):
        """
        For generating 1-D backgrounds
        x_points: numpy array of time values
        noise_level: non-negative value for 95% confidence interval of noise level. If noise_level == 0, then no noise is added.
        mean_value: level around which noise will vary.
        autocorrelation_parameter: parameter for autocorrelation of background data. In reality, signal data from detectors is often autocorrelated.
        """

        sigma = (
            noise_level / 2
        )  # to account for fact 95% of samples from normal distribution are within 2 std

        self.signal = self.create_autocorrelated_data(
            n_points, mean, sigma, autocorrelation_parameter
        )

        return self.signal
