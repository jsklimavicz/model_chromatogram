from numpy.random import uniform
import math

from model_chromatogram.utils import exponnorm, exponnorm_scalar, scaled_exponnorm
from scipy.optimize import minimize_scalar
from model_chromatogram.compounds import Compound
import numpy as np
from model_chromatogram.system import Column, System

from model_chromatogram.user_parameters import *


class PeakCreator:
    """
    Class designed to create peak signals based on peak-specific information and user-defined parameters. Generated peaks have an exponentially-modified gaussian shape to mimic peak asymmetry. Only one PeakCreator should be used per injection to ensure that injection-specific variations in retention time and peak height/area remain constant within an injection.
    """

    def __init__(self, system: System) -> None:
        """
        Class designed to create peak signals based on peak-specific information and user-defined parameters. Sets the injection-specific parameters.
        """
        self.system = system
        self.column = system.get_column()
        self.column_broadening, self.column_asymmetry, self.rt_diff = (
            self.column.get_column_peak_vals()
        )
        self.injection_specific_retention_time_offset = (
            uniform(-RETENTION_TIME_RANDOM_OFFSET_MAX, RETENTION_TIME_RANDOM_OFFSET_MAX)
            + system.get_retention_time_offset()
        )
        self.injection_specific_heigh_modifier = uniform(
            1 - OVERALL_HEIGHT_RANDOM_NOISE, 1 + OVERALL_HEIGHT_RANDOM_NOISE
        )

    def compound_peak(self, compound: Compound, times: np.array) -> np.array:
        """
        Creates the signal for a compound peak.

        Args:
            compound (Compound): the compound from the compound library generate a peak signal for.
            times (np.array): The time points at which to calculate the signal.

        Returns:
            out (np.array): The signal values for the peak.
        """

        # set initial asymmetry
        peak_asymmetry = (1 + compound.asymmetry_addition) * DEFAULT_BASE_ASYMMETRY

        # get all parameters for the peak
        peak_dict = self.peak(
            retention_time=compound.retention_time,
            base_asymmetry=peak_asymmetry,
            base_width=DEFAULT_PEAK_WIDTH,
            compound=compound,
        )

        raw_signal = exponnorm(
            times,
            K=peak_dict["asymmetry"],
            loc=peak_dict["time"],
            scale=peak_dict["width"],
        )

        # Set elements less than 1e-8 to 0
        raw_signal[raw_signal < 1e-8] = 0
        return raw_signal

    def peak(
        self,
        retention_time: float,
        compound: Compound,
        height: float = 1,
        name: str = None,
        base_width: float = DEFAULT_PEAK_WIDTH,
        base_asymmetry: float = DEFAULT_BASE_ASYMMETRY,
    ) -> dict:
        """
        Calculates proper height, width, and asymmetry for a peak given initial values and a retention time.

        Args:
            retention_time (float): The re
        """

        def height_change_for_broadening(
            base_sigma, curr_sigma, base_asymmetry, curr_asymmetry
        ):

            curr_times = np.arange(0, max(base_sigma, curr_sigma), 0.0001)
            curr_signal = exponnorm(curr_times, curr_asymmetry, scale=curr_sigma)
            base_signal = exponnorm(curr_times, base_asymmetry, scale=base_sigma)
            curr_fun = np.max(curr_signal)
            base_fun = np.max(base_signal)
            mode = curr_times[np.argmax(curr_signal)]

            return curr_fun / base_fun, mode

        base_sigma = base_width / (2 * math.sqrt(2 * math.log(2)))
        curr_sigma = (
            base_sigma * math.pow(WIDENING_CONSTANT, retention_time)
            + self.column_broadening
        ) * compound.broadening_factor

        asymmetry = (
            1.0
            / (
                base_asymmetry
                * math.pow(ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME, retention_time)
            )
            + self.column_asymmetry
        )

        # determine how much the heigh of a peak changes based on its broadening
        peak_broadening_height_factor, mode = height_change_for_broadening(
            base_sigma, curr_sigma, base_asymmetry, asymmetry
        )

        # calculate a modified height based on original height, the peak broadening factor, the injection-specific height modifier, the user-defined signal multiplier, and a random number close to 1 for variation in peak height within a run.
        mod_height = (
            height
            * peak_broadening_height_factor
            * self.injection_specific_heigh_modifier
            * SIGNAL_MULTIPLIER
            * uniform(
                1 - INDIVIDUAL_HEIGHT_RANDOM_NOISE, 1 + INDIVIDUAL_HEIGHT_RANDOM_NOISE
            )
        )

        desired_mode = (
            retention_time
            + self.injection_specific_retention_time_offset
            + uniform(
                -INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE,
                INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE,
            )
        ) * self.rt_diff

        exponentially_modified_gaussian_mean = desired_mode - mode

        return dict(
            name=name,
            time=exponentially_modified_gaussian_mean,
            height=mod_height,
            width=curr_sigma,
            asymmetry=asymmetry,
        )
