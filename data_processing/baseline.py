import numpy as np


def als_psalsa(
    raw_time: np.array,
    raw_signal: np.array,
    sr=10,
    p=0.01,
    smoothness=10,
    k=2,
    rel_tol=1e-6,
):
    """
    Based on
    S. Oller-Moreno, A. Pardo, J. M. Jiménez-Soto, J. Samitier and S. Marco, "Adaptive Asymmetric Least Squares baseline estimation for analytical instruments," 2014 IEEE 11th International Multi-Conference on Systems, Signals & Devices (SSD14), Barcelona, Spain, 2014, pp. 1-5, doi: 10.1109/SSD.2014.6808837.
    https://diposit.ub.edu/dspace/bitstream/2445/188026/1/2014_IEEE_Adaptive_MarcoS_postprint.pdf
    """
    # select a subset of points
    interval = sr / (60 * (raw_time[1] - raw_time[0]))
    interval = int(round(interval))
    time = [*raw_time[::interval]]
    signal: np.array = [*raw_signal[::interval]]

    # set initial z values
    z = np.mean(signal) * np.ones_like(signal)
    residuals = signal - z

    def get_weights(residuals):
        return np.where(residuals > 0, p * np.exp(-residuals / k), 1 - p)

    # set inital weights to ones
    weights = np.ones_like(z)
    converged = False
    iterations = 0

    # create tridiagonal second difference matrix
    def difference_matrix(size):
        a = -1 * np.ones(size - 1)  # upper and lower diagonals
        b = 2 * np.ones(size)  # diagonal
        D = np.diag(a, k=-1) + np.diag(b) + np.diag(a, k=1)
        # Perturb matrix for dealing with endpoints:
        D[0][0] = D[-1][-1] = 1
        return D

    D = difference_matrix(len(z))

    # define loss function
    def loss_function(weights, residuals):
        def second_deriv(residuals):
            return np.diff(np.diff(residuals))

        S = np.sum(weights * residuals**2)
        S += smoothness * sum(second_deriv(residuals) ** 2)
        return S

    prev_loss = 0  # intial loss
    while not converged and iterations < 20:
        iterations += 1
        W = np.diag(weights)
        z = np.linalg.inv(W + smoothness * D.T @ D).dot(weights * signal)
        residuals = signal - z
        curr_loss = loss_function(weights, residuals)
        rel_loss = abs(curr_loss - prev_loss) / curr_loss
        if rel_loss < rel_tol:
            converged = True
            break
        prev_loss = curr_loss
        weights = get_weights(residuals)

    return z, time
