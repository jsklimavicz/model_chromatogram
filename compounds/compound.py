from scipy.interpolate import CubicSpline
import numpy as np
from pydash import get as _get
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
from compounds import UVSpectrum


class Compound:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.name = _get(kwargs, "name").strip()
        self.id = _get(kwargs, "id")
        self.cas = _get(kwargs, "cas").strip()
        self.mw = float(kwargs["mw"])
        self.default_retention_CV = float(kwargs["default_CV"])
        self.log_p = float(kwargs["logp"])
        self.asymmetry_addition = self.__set_initial_float("asymmetry_addition")
        self.h_donors: float = self.__set_initial_float("H_donor")
        self.h_acceptors: float = self.__set_initial_float("H_acceptor")
        self.refractivity: float = self.__set_initial_float("refractivity")
        self.log_s: float = self.__set_initial_float("log_s")
        self.tpsa: float = self.__set_initial_float("tpsa")
        self.set_uv_spectrum()

    def __set_initial_float(self, key, default=0):
        try:
            return float(_get(self.kwargs, key))
        except:
            return default

    def set_uv_spectrum(self):
        try:
            self.spectrum = UVSpectrum(self.cas)
        except:
            # default UV spectrum
            self.spectrum = UVSpectrum("63-74-1")

    def get_absorbance(self, wavelength, concentration=None, pathlength=1):
        absorbance = self.spectrum.get_epsilon(wavelength) * pathlength
        absorbance *= (
            concentration if concentration is not None else self.m_molarity / 1000
        )
        return absorbance

    def set_concentration(self, concentration):
        self.concentration = concentration
        self.m_molarity = 1000 * self.concentration / self.mw

    def set_retention_time(self, column_volume: float, solvent_profiles: pd.DataFrame):
        """
        Calculates the retention time of a compound based on column volume, solvent properties, and compound properties.

        Args:
            column_volume (float): volume of column
            solvent_profiles (pd.DataFrame): dataframe containing corrsponding values for:
                time
                hydrogen bond acidity (hb_acidity)
                hydrogen bond basicity (hb_basicity)
                polarity
                dielectric

        Returns:
            retention_time (float): Value of retention time.

        """

        # first calculate constants a, b, c, and d:
        t_0 = self.default_retention_CV
        a = 2 * (3 - self.log_p) + self.tpsa / self.mw
        b = 600 / self.mw * (np.sqrt(self.h_donors) - 1)
        c = 600 / self.mw * (np.sqrt(self.h_acceptors) - 1)
        d = (1 - self.log_s) / 5

        time = solvent_profiles["time"].to_numpy()

        flow = solvent_profiles["flow"].to_numpy()

        objective = a * (solvent_profiles["polarity"].to_numpy())
        objective -= b * (solvent_profiles["hb_basicity"].to_numpy())
        objective -= c * (solvent_profiles["hb_acidity"].to_numpy())
        objective -= d * (solvent_profiles["dielectric"].to_numpy())
        linear_component = a * -0.15 + (b + c + d) * -0.05
        objective += linear_component
        objective /= 120

        objective *= flow

        dt = time[1] - time[0]
        integral_function = np.cumsum(objective) * dt

        spline = CubicSpline(
            time,
            integral_function,
            extrapolate=False,
        )

        def objective_function(t):
            return t - (t_0 - spline(t))

        v = fsolve(objective_function, t_0 + 0.2, xtol=1e-04, maxfev=50)
        adjusted_retention_volume = v[0]
        # Note: Total volume cannot be less than one column volume...
        # Let's modify with 1+norm.cdf to prevent this.
        adjusted_retention_volume = max(
            1 + norm.cdf(adjusted_retention_volume) / 2,
            adjusted_retention_volume,
        )
        cumulative_volume = np.cumsum(flow) * dt
        cumulative_volume /= column_volume

        self.retention_time = np.interp(
            adjusted_retention_volume, cumulative_volume, time
        )

        return self.retention_time
