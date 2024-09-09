import numpy as np
from scipy.special import erfc


def exponnorm(x, K, loc=0, scale=1):
    y = (x - loc) / scale
    vals = 1 / (2 * K**2) - y / K
    vals += np.log(erfc(((1 / K) - y) / np.sqrt(2)))
    return np.exp(vals) / (2 * K * scale)


def scaled_exponnorm(x, h, K, loc=0, scale=1):
    return h * exponnorm(x, K=K, loc=loc, scale=scale)
