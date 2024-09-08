import numpy as np
from scipy.special import erfc


def _exponnorm(x, K, loc=0, scale=1):
    y = (x - loc) / scale
    vals = 1 / (2 * K) * np.exp(1 / (2 * K**2) - y / K)
    vals *= erfc(-((y - (1 / K)) / np.sqrt(2)))
    return vals / scale
