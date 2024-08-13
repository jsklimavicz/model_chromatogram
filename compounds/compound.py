from scipy.interpolate import CubicSpline
import numpy as np
from pydash import get as _get
import pandas as pd
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
from scipy.stats import norm


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
        self.default_retention_time = None
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

        flow = solvent_profiles["flow"]

        objective = a * (solvent_profiles["polarity"] - 0.15)
        objective -= b * (solvent_profiles["hb_basicity"] + 0.05)
        objective -= c * (solvent_profiles["hb_acidity"] + 0.05)
        objective -= d * (solvent_profiles["dielectric"] + 0.050)
        objective /= 120

        objective *= flow

        dt = solvent_profiles["time"][1] - solvent_profiles["time"][0]
        integral_function = np.cumsum(objective) * dt

        spline = CubicSpline(
            solvent_profiles["time"],
            integral_function,
            extrapolate=False,
        )

        def objective_function(t):
            return t - (t_0 - spline(t))

        v = fsolve(objective_function, t_0 + 1, xtol=1e-05, maxfev=500)
        adjusted_retention_volume = v[0]
        # Note: Total volume cannot be less than one column volume...
        # Let's modify with 1+norm.cdf to prevent this.
        adjusted_retention_volume = max(
            1 + norm.cdf(adjusted_retention_volume), adjusted_retention_volume
        )
        cumulative_volume = np.cumsum(flow) * dt
        cumulative_volume /= column_volume
        spline = CubicSpline(
            cumulative_volume,
            solvent_profiles["time"],
            extrapolate=False,
        )

        self.retention_time = spline(adjusted_retention_volume)

        return self.retention_time


class UVSpectrum:
    def __init__(self, cas: str) -> None:
        self.cas = cas
        self.fetch_spectrum()
        self.extrapolate_lower_end()
        self.extrapolate_upper_end()
        self.create_spline()

    def fetch_spectrum(self) -> None:
        try:
            self.wavelengths = []
            self.log_epsilon = []
            start_read = False
            with open(f"./compounds/spectra/{self.cas}-UVVis.jdx") as f:
                while line := f.readline():
                    if "##END" in line:
                        break
                    elif start_read:
                        x, y = line.split(",")
                        self.wavelengths.append(float(x.strip()))
                        self.log_epsilon.append(float(y.strip()))
                    elif "##XYPOINTS" in line:
                        start_read = True
            self.wavelengths = np.array(self.wavelengths)
            self.log_epsilon = np.array(self.log_epsilon)
        except FileNotFoundError as e:
            print(f"Check if the jdx file for CAS {self.cas} is in this directory.")
            raise

    def create_spline(self) -> None:
        self.spline = CubicSpline(
            self.wavelengths,
            self.log_epsilon,
            extrapolate=True,
            bc_type=((2, 0), (2, 0)),
        )

    def extrapolate_lower_end(self) -> None:
        min_wave = int(np.floor(min(self.wavelengths) - 0.1))
        if min_wave > 185:
            addition_wavelengths = np.arange(180, min_wave)
        else:
            addition_wavelengths = np.arange(min_wave - 10, min_wave)
        addition_eps = np.ones_like(addition_wavelengths) * self.log_epsilon[0]
        addition_eps += 0.01 * np.sqrt(np.arange(len(addition_eps), 0, -1))
        self.wavelengths = [*addition_wavelengths, *self.wavelengths]
        self.log_epsilon = [*addition_eps, *self.log_epsilon]

    def extrapolate_upper_end(self) -> None:
        max_wave = int(np.ceil(max(self.wavelengths) + 0.1))
        addition_wavelengths = np.arange(max_wave, max_wave + 5)
        addition_eps = np.ones_like(addition_wavelengths) * self.log_epsilon[-1]
        addition_eps -= 0.01 * (np.arange(0, len(addition_eps)))
        self.wavelengths = [*self.wavelengths, *addition_wavelengths]
        self.log_epsilon = [*self.log_epsilon, *addition_eps]

    def get_epsilon(self, wavelength: np.array, log=False) -> None:
        if log:
            return self.spline(wavelength)
        else:
            return 10 ** self.spline(wavelength)
