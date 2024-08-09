from scipy.special import erfcx
from scipy.stats import exponnorm
import numpy as np
import matplotlib.pyplot as plt


def expmodgauss_pdf(x, h, mu, sigma, tau):
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

    print(val)
    plt.plot(x, val, label="Fitted Curve", color="red")
    plt.show()
    return val


def gaussian_pdf(x, height=1, scale=1, loc=0):
    return (
        height / (scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - loc) / scale) ** 2)
    )
