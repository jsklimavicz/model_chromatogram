import numpy as np
from scipy.sparse import diags

from scipy.sparse.linalg import spsolve

from scipy.linalg import solve


def als_psalsa(
    raw_time: np.array,
    raw_signal: np.array,
    sr: int = 5,
    p: float = 0.001,
    s: float = 1,
    k: float = 2,
    rel_tol: float = 1e-6,
):
    """Peaked Signal’s Asymmetric Least Squares Algorithm

    Algorithm for finding a baseline of a chromatogram based on the method outlined in:
    S. Oller-Moreno, A. Pardo, J. M. Jiménez-Soto, J. Samitier and S. Marco, "Adaptive Asymmetric Least Squares baseline estimation for analytical instruments," 2014 IEEE 11th International Multi-Conference on Systems, Signals & Devices (SSD14), Barcelona, Spain, 2014, pp. 1-5, doi: 10.1109/SSD.2014.6808837.
    https://diposit.ub.edu/dspace/bitstream/2445/188026/1/2014_IEEE_Adaptive_MarcoS_postprint.pdf

    Args:
        raw_time (np.array): chromatogram time values
        raw_signal (np.array): chromatogram signal values
        sr (int): sampling rate to downsample the raw time and raw signal. Every `sr`th raw signal and time are kept. Setting to 1 keeps everything; this is not recommended due to long compute times. Default = 10
        p (float): weight parameter for the asymmetric least square algorithm. Must be between 0 and 1; typically 0.001 <= p <= 0.1.
        s (float): multiplicative factor for cost function. Higher values force penalize "wavier" baselines.
        k (float): adaptive value for controlling the exponential decay of weight for peak regions

    Returns:
        out (tuple(np.array, np.array)): Tuple containing the time and signal values of the baseline.

    """
    # select a subset of points
    interval = sr / (60 * (raw_time[1] - raw_time[0]))
    interval = int(round(interval))
    time = [*raw_time[::interval]]
    signal: np.array = [*raw_signal[::interval]]

    size = len(time)

    # set initial z values
    z = (signal[-1] * signal[0]) * np.ones_like(signal)
    residuals = signal - z

    def get_weights(residuals):
        return np.where(residuals > 0, p * np.exp(-residuals / k), 1 - p)

    # set inital weights to ones
    weights = np.ones_like(z)
    converged = False
    iterations = 0

    a = -1 * np.ones(size - 1)  # upper and lower diagonals
    b = 2 * np.ones(size)  # diagonal
    b[0] = b[-1] = 1  # Perturb matrix for dealing with endpoints
    D = diags([a, b, a], [-1, 0, 1])
    D2_s = s * D @ D

    # define loss function
    def loss_function(weights, residuals, z):
        S = np.sum(weights * residuals**2)
        S += s * sum((D @ z) ** 2)
        return S

    prev_loss = 0  # intial loss

    # iterate z = (W + smoothness * D'D)^-1 W y until solved
    while not converged and iterations < 20:
        iterations += 1
        W = diags([weights], [0])
        z = spsolve(W + D2_s, (weights * signal))
        residuals = signal - z
        curr_loss = loss_function(weights, residuals, z)
        rel_loss = abs(curr_loss - prev_loss) / curr_loss
        if rel_loss < rel_tol:
            converged = True
            break
        prev_loss = curr_loss
        weights = get_weights(residuals)

    return time, z
