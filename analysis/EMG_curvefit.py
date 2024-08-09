from scipy.optimize import least_squares
from scipy.special import erfcx
import numpy as np
import math


def expmodgauss(x, h, mu, sigma, tau):
    sigma_over_tau = sigma / tau
    z_score = (x - mu) / sigma
    erfcx_arg = 2 ** (-1 / 2) * (sigma_over_tau - z_score)
    val = (
        h
        * np.exp(-0.5 * (z_score**2))
        * sigma_over_tau
        * np.sqrt(np.pi / 2)
        * erfcx(erfcx_arg)
    )
    return val
