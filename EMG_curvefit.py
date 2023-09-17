from scipy.optimize import 
import numpy as np
import math



def expmodgauss(x, h, mu, sigma, tau):
    sigma_over_tau = sigma/tau
    z_score = (x-mu)/sigma
    val = h * exp(-0.5 * (z_score ** 2)) * sigma_over_tau * 