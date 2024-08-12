from baseline import Baseline
from peak_model import Peak, PeakDefinitionError
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol
from scipy import signal

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from user_parameters import *


class Chromatogram:
    def __init__(
        self,
        duration=RUN_LENGTH,
        sample_rate=SAMPLE_RATE,
        background_mean=BACKGROUND_MEAN,
        background_noise=BACKGROUND_NOISE,
        linear_level=LINEAR_LIMIT,
        smoothing=BACKGROUND_SMOOTHING,
    ) -> None:
        """
        duration: time in minutes for chromatogram (Unit: Min)
        sample_rate: number of samples per second taken (Unit: Hz)
        background_mean: average background signal for chromatogram.
        background_noise: background noise for chromatogram.
        """
        self.duration = duration
        self.linear_level = linear_level
        self.saturation_scale = 0.2 * linear_level
        self.sample_rate = sample_rate
        self.smoothing = smoothing
        self.n_points = duration * sample_rate * 60 + 1
        self.times = np.linspace(0, duration, self.n_points)
        baseline = Baseline()
        if background_noise is None:
            background_noise = sample_rate / 20
        self.signal = baseline.create_background(
            self.n_points, noise_level=background_noise, mean=background_mean
        )
        self._peaks = []

    def add_peak(self, rt, height, sigma=None, peak_width_10=None, k=None):
        if peak_width_10 is not None and sigma is None:
            sigma = peak_width_10 / (4.29194)
        if peak_width_10 is None and sigma is None:
            raise PeakDefinitionError(
                "You must specify `sigma` or `peak_width_10` to define a peak."
            )
        if k is None:
            k = 1 / (sigma * 10)

        new_peak = Peak(rt=rt, sigma=sigma, height=height, k=k)
        if k >= 0:
            self.signal += new_peak.get_signal(self.times)
        else:
            self.signal += new_peak.get_signal(-self.times)
        self._peaks.append(new_peak)

    def _adjust_saturation(self, x):
        if x < self.linear_level:
            return x
        else:
            val = self.linear_level
            diff_x = x - val
            diff = self.saturation_scale * (1 - np.exp(-diff_x / self.saturation_scale))
            return val + diff

    def get_signal(self):
        return np.array([self._adjust_saturation(x) for x in self.signal])

    def signal_derivative(self, n=1):
        # use central differences; pad with values on ends
        if n == 1:
            weights = np.array([-1, 0, 1]) / 2.0
        elif n == 2:
            weights = np.array([1, -2.0, 1])

        b, a = signal.butter(1, 0.05)

        curr_signal = self.get_signal()

        curr_signal = savgol(curr_signal, self.smoothing + 1, 3)
        forward = np.array([*curr_signal[1:], curr_signal[-1]])
        backward = np.array([curr_signal[0], *curr_signal[:-1]])

        dt = self.times[1] - self.times[0]

        vals = (
            weights[0] * backward + weights[1] * curr_signal + weights[2] * forward
        ) / (dt**n)

        vals = np.convolve(
            vals, np.ones(2 * self.smoothing) / (2 * self.smoothing), "same"
        )

        # return savgol(vals, 5 * self.sample_rate + 1, 3)
        return signal.filtfilt(b, a, savgol(vals, self.smoothing + 1, 3))
        # return vals
        # return signal.filtfilt(b, a, vals)

    def plot(self, ax):
        plt.xlabel("time (min)")
        plt.ylabel("absorbance (mAU)")
        ax.plot(self.times, self.get_signal())
