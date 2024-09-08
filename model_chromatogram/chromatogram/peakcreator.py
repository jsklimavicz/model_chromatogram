from numpy.random import uniform
import math

from model_chromatogram.lib import exponnorm
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

    def __find_exponnorm_mode(
        self, sigma: float, asymmetry: float, return_fun: bool = False
    ) -> None:
        """
        Finds the mode/peak of an exponentially-modified gaussian.

        Args:
            sigma (float): the standard deviation of the base gaussian curve
            asymmetry (float): the factor provided to exponnorm to make asymmetric peaks.
            return_fun (bool): If `True`, returns both the x-value and function value of the mode; otherwise, returns only the x-value.

        Returns:
            x (float): The x-value of the distribution maximum

            fun (float): Optional when `return_fun == True`, the distribution value at `x`.
        """

        # Define a function that returns the negative PDF (to maximize)
        def neg_pdf(x):
            # return -exponnorm.pdf(x, scale=sigma, K=asymmetry)
            return -exponnorm(x, scale=sigma, K=asymmetry)

        # Use optimization to find the mode
        result = minimize_scalar(neg_pdf, tol=1e-4)
        if return_fun:
            return result.x, -result.fun
        else:
            return result.x

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

        # return signal array
        # raw_signal = exponnorm.pdf(
        #     times,
        #     K=peak_dict["asymmetry"],
        #     loc=peak_dict["time"],
        #     scale=peak_dict["width"],
        # )
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
            mode, curr_fun = self.__find_exponnorm_mode(
                curr_sigma, curr_asymmetry, return_fun=True
            )
            _, base_fun = self.__find_exponnorm_mode(
                base_sigma, base_asymmetry, return_fun=True
            )
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
