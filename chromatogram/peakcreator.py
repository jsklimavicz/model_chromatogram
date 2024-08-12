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

    def __find_exponnorm_mode(self, sigma, asymmetry):
        exponnorm_dist = exponnorm(scale=sigma, K=asymmetry)

        # Find the mode
        # Define a function that returns the negative PDF (to maximize)
        def neg_pdf(x):
            return -exponnorm_dist.pdf(x)

        # Use optimization to find the mode
        result = minimize_scalar(neg_pdf)
        return result.x

    def compound_peak(self, compound: Compound, absorbance: float, times):
        peak_dict = self.peak(
            retention_time=compound.default_retention_time, height=absorbance
        )
        return peak_dict["height"] * exponnorm.pdf(
            times,
            K=peak_dict["asymmetry"],
            loc=peak_dict["time"],
            scale=peak_dict["width"],
        )

    def peak(
        self,
        retention_time: float,
        height: float,
        name: str = None,
        base_width: float = DEFAULT_PEAK_WIDTH,
        base_asymmetry: float = DEFAULT_BASE_ASYMMETRY,
    ) -> dict:

        sigma = (
            base_width
            * math.pow(WIDENING_CONSTANT, retention_time)
            / (2 * math.sqrt(2 * math.log(2)))
        )

        asymmetry = 1.0 / (
            base_asymmetry
            * math.pow(ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME, retention_time)
        )

        mod_height = (
            height
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

        exponentially_modified_gaussian_mean = (
            desired_mode - self.__find_exponnorm_mode(sigma, asymmetry)
        )

        return dict(
            name=name,
            time=exponentially_modified_gaussian_mean,
            height=mod_height,
            width=sigma,
            asymmetry=asymmetry,
        )
