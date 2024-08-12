from scipy.interpolate import CubicSpline
import numpy as np
from pydash import get as _get


class Compound:
    def __init__(self, **kwargs) -> None:
        self.name = _get(kwargs, "name").strip()
        self.id = _get(kwargs, "id")
        self.cas = _get(kwargs, "cas").strip()
        self.mw = float(kwargs["mw"])
        self.default_retention_CV = float(kwargs["default_CV"])
        self.logp = float(kwargs["logp"])
        self.asymmetry_addition = float(kwargs["asymmetry_addition"])
        self.kwargs = kwargs
        self.default_retention_time = None
        self.set_uv_spectrum()

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


# import matplotlib.pyplot as plt

# cas = "7732-18-5"

# spectrum = UVSpectrum(cas)
# min_val = min(spectrum.wavelengths)
# max_val = max(spectrum.wavelengths)
# wave = np.linspace(190, max_val + 100, 300)
# values = spectrum.get_epsilon(wave)
# plt.plot(wave, values, c="red")

# wave = np.linspace(190, max_val, 100)
# values = spectrum.get_epsilon(wave)

# plt.plot(wave, values, c="blue")

# plt.show()

# print("t")
