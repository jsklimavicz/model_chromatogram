from scipy.interpolate import CubicSpline
import numpy as np


class UVSpectrum:
    def __init__(self, cas: str) -> None:
        self.cas = cas
        self.fetch_spectrum()
        self.extrapolate_lower_end()
        self.extrapolate_upper_end()
        self.spline = None
        self.create_spline()

    def fetch_spectrum(self) -> None:
        try:
            self.wavelengths = []
            self.log_epsilon = []
            start_read = False
            with open(
                f"./model_chromatogram/compounds/spectra/{self.cas}-UVVis.jdx"
            ) as f:
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
        except FileNotFoundError:
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
        addition_wavelengths = np.arange(max_wave, max_wave + 10)
        addition_eps = np.ones_like(addition_wavelengths) * self.log_epsilon[-1]
        # TODO: replace decrease with slope if negative
        addition_eps -= 0.01 * (np.arange(0, len(addition_eps)))
        self.wavelengths = [*self.wavelengths, *addition_wavelengths]
        self.log_epsilon = [*self.log_epsilon, *addition_eps]

    def get_epsilon(self, wavelength: np.array, log=False) -> None:
        if self.spline is None:
            self.create_spline()
        if log:
            return self.spline(wavelength)
        else:
            return 10 ** self.spline(wavelength)
