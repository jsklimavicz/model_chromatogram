from numpy.random import uniform
import math
from scipy.stats import exponnorm
from scipy.optimize import minimize_scalar
from compounds.compound import Compound

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from user_parameters import *


class PeakCreator:

    def __init__(self):
        self.retention_time_offset = uniform(
            -RETENTION_TIME_RANDOM_OFFSET_MAX, RETENTION_TIME_RANDOM_OFFSET_MAX
        )
        self.height_modifier = uniform(
            1 - OVERALL_HEIGHT_RANDOM_NOISE, 1 + OVERALL_HEIGHT_RANDOM_NOISE
        )

    def __find_exponnorm_mode(self, sigma, asymmetry, return_fun=False):
        exponnorm_dist = exponnorm(scale=sigma, K=asymmetry)

        # Find the mode
        # Define a function that returns the negative PDF (to maximize)
        def neg_pdf(x):
            return -exponnorm_dist.pdf(x)

        # Use optimization to find the mode
        result = minimize_scalar(neg_pdf, tol=1e-4)
        if return_fun:
            return result.x, -result.fun
        else:
            return result.x

    def compound_peak(self, compound: Compound, times):
        peak_asymmetry = (1 + compound.asymmetry_addition) * DEFAULT_BASE_ASYMMETRY
        peak_dict = self.peak(
            retention_time=compound.retention_time,
            base_asymmetry=peak_asymmetry,
        )
        return exponnorm.pdf(
            times,
            K=peak_dict["asymmetry"],
            loc=peak_dict["time"],
            scale=peak_dict["width"],
        )

    def peak(
        self,
        retention_time: float,
        height: float = 1,
        name: str = None,
        base_width: float = DEFAULT_PEAK_WIDTH,
        base_asymmetry: float = DEFAULT_BASE_ASYMMETRY,
    ) -> dict:

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
        curr_sigma = base_sigma * math.pow(WIDENING_CONSTANT, retention_time)

        asymmetry = 1.0 / (
            base_asymmetry
            * math.pow(ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME, retention_time)
        )
        peak_broadening_height_factor, mode = height_change_for_broadening(
            base_sigma, curr_sigma, base_asymmetry, asymmetry
        )

        mod_height = (
            height
            * peak_broadening_height_factor
            * self.height_modifier
            * SIGNAL_MULIPLIER
            * uniform(
                1 - INDIVIDUAL_HEIGHT_RANDOM_NOISE, 1 + INDIVIDUAL_HEIGHT_RANDOM_NOISE
            )
        )

        desired_mode = (
            retention_time
            + self.retention_time_offset
            + uniform(
                -INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE,
                INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE,
            )
        )

        exponentially_modified_gaussian_mean = desired_mode - mode

        return dict(
            name=name,
            time=exponentially_modified_gaussian_mean,
            height=mod_height,
            width=curr_sigma,
            asymmetry=asymmetry,
        )
