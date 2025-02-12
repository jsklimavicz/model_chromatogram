import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from abc import abstractmethod
from model_chromatogram.user_parameters import JULIA_PARAMTERS
import os

if JULIA_PARAMTERS["julia"] is not None:
    os.environ["PYTHON_JULIAPKG_EXE"] = JULIA_PARAMTERS["julia"]

    from juliacall import Main as jl

    jl.include("./model_chromatogram/methods/viscosity.jl")
    use_julia_fit = True
else:
    use_julia_fit = False


METHANOL = {
    "density": 0.792,  # g/cm^3
    "mw": 32.04,  # g/mol
}
WATER = {
    "density": 0.997,  # g/cm^3
    "mw": 18.01528,  # g/mol
}
ACETONITRILE = {
    "density": 0.786,  # g/cm^3
    "mw": 41.05,  # g/mol
}
THF = {
    "density": 0.886,  # g/cm^3
    "mw": 72.11,  # g/mol
}


def viscosity(solvent, temp, x, pressure):
    return jl.ViscosityModel.viscosity(solvent, float(temp), float(x), float(pressure))


class ViscosityInterpolator(RegularGridInterpolator):
    def __init__(self, interp_tuple: tuple, values: np.array):
        super().__init__(
            interp_tuple,
            values,
            method="pchip",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, x):
        return super().__call__(x)

    def interpolate(self, x):
        return self(x)


class ViscosityCalculator:

    path = "./model_chromatogram/methods/viscosity_data/"

    def __init__(self) -> None:
        self._make_interpolator()
        pass

    def compute_molar_fraction_from_percent(self, percent, solvent):
        f = percent / 100
        a = f * solvent["density"] * solvent["mw"]
        b = (1 - f) * WATER["density"] * WATER["mw"]
        return a / (a + b)

    def temperature_correction(
        self, T: float, T0: float = 25, alpha: float = 0.03
    ) -> float:
        """
        Returns the ratio mu(T)/mu(T0) = exp[alpha*(T0 - T)] as a simple model.

        Args:
            T (float): The temperature in *CELSIUS*.
            T0 (float, optional): The reference temperature in *CELSIUS*. Defaults to 25.
            alpha (float, optional): The temperature coefficient. Defaults to 0.03.
        """
        T += 273.15
        T0 += 273.15
        return np.exp(alpha * (T0 - T))

    def pressure_correction(self, P: float, beta: float = 5e-5) -> float:
        """
        Returns the ratio mu(P) / mu(0) = exp(beta*P).

        Args:
            P (float): The pressure in *BAR*.
            beta (float, optional): The pressure coefficient. Defaults to 1e-5.
        """
        return np.exp(beta * P)

    @abstractmethod
    def _make_interpolator(self):
        pass

    @abstractmethod
    def interpolate_viscosity(self, pressure, temp, x):
        pass


class MethanolViscosity(ViscosityCalculator):
    file = "meoh.csv"

    def __init__(self) -> None:
        super().__init__()

    def _make_interpolator(self):
        if not use_julia_fit:
            df = pd.read_csv(self.path + self.file)
            unique_temps = np.sort(df["temp"].unique())
            unique_x = np.sort(df["x"].unique())
            unique_pressures = np.sort(df["pressure"].unique())
            df_sorted = df.sort_values(by=["temp", "x", "pressure"])
            viscosity_grid = df_sorted.pivot_table(
                index="temp", columns=["x", "pressure"], values="viscosity"
            )
            viscosity_values = viscosity_grid.values.reshape(
                len(unique_temps), len(unique_x), len(unique_pressures)
            )
            self.interpolator = ViscosityInterpolator(
                (unique_temps, unique_x, unique_pressures), viscosity_values
            )

    def interpolate_viscosity(self, pressure, temp, x):
        """
        Interpolates the viscosity at a given temperature, composition, and pressure.

        Args:
            pressure (float): The pressure in *BAR*.
            temp (float): The temperature in *CELSIUS*.
            x (float): The composition of the mobile phase.

        Returns:
            float: The interpolated viscosity.
        """
        if use_julia_fit:
            return viscosity("meoh", temp, x, pressure)
        else:
            return self.interpolator((temp, x, pressure))

    def compute_molar_fraction_from_percent(self, percent):
        return super().compute_molar_fraction_from_percent(percent, METHANOL)


class AcetonitrileViscosity(ViscosityCalculator):
    file = "acn.csv"

    def __init__(self) -> None:
        super().__init__()

    def _make_interpolator(self):
        if not use_julia_fit:
            df = pd.read_csv(self.path + self.file)
            unique_x = np.sort(df["x"].unique())
            unique_pressures = np.sort(df["pressure"].unique())
            df_sorted = df.sort_values(by=["x", "pressure"])
            viscosity_grid = df_sorted.pivot_table(
                index="x", columns=["pressure"], values="viscosity"
            )
            viscosity_values = viscosity_grid.values.reshape(
                len(unique_x), len(unique_pressures)
            )
            self.interpolator = ViscosityInterpolator(
                (unique_x, unique_pressures), viscosity_values
            )

    def temperature_correction(
        self, T: float, T_ref: float = 25, alpha: float = 0.03
    ) -> float:
        """
        Returns the ratio mu(T)/mu(T0) = exp[alpha*(T0 - T)] as a simple model.

        Args:
            T (float): The temperature in *CELSIUS*.
            T0 (float, optional): The reference temperature in *CELSIUS*. Defaults to 25.
            alpha (float, optional): The temperature coefficient. Defaults to 0.03.
        """

        a = 1.856e-11
        b = 4209
        c = 0.04527
        d = -3.376e-5

        def andrade(t):
            t += 273.15
            b1 = b / t
            c1 = c * t
            d1 = d * t * t
            return a * np.exp(b1 + c1 + d1)

        v = andrade(T.copy())
        v_ref = andrade(T_ref)

        return v / v_ref

    def interpolate_viscosity(self, pressure, temp, x):
        """
        Interpolates the viscosity at a given temperature, composition, and pressure.

        Args:
            pressure (float): The pressure in *BAR*.
            temp (float): The temperature in *CELSIUS*.
            x (float): The composition of the mobile phase.

        Returns:
            float: The interpolated viscosity.
        """
        if use_julia_fit:
            return viscosity("acn", temp, x, pressure)
        else:
            visc = self.interpolator((x, pressure))
            temp_corr = self.temperature_correction(temp)
            return visc * temp_corr

    def calculate_pressure(self):
        pass

    def compute_molar_fraction_from_percent(self, percent):
        return super().compute_molar_fraction_from_percent(percent, ACETONITRILE)


class TetrahydrofuranViscosity(ViscosityCalculator):
    file = "thf.csv"

    def __init__(self) -> None:
        super().__init__()

    def _make_interpolator(self):
        if not use_julia_fit:
            df = pd.read_csv(self.path + self.file)
            unique_temps = np.sort(df["temp"].unique())
            unique_x = np.sort(df["x"].unique())
            df_sorted = df.sort_values(by=["temp", "x"])
            viscosity_grid = df_sorted.pivot_table(
                index="temp", columns=["x"], values="viscosity"
            )
            viscosity_values = viscosity_grid.values.reshape(
                len(unique_temps), len(unique_x)
            )
            self.interpolator = ViscosityInterpolator(
                (unique_temps, unique_x), viscosity_values
            )

    def interpolate_viscosity(self, pressure, temp, x):
        """
        Interpolates the viscosity at a given temperature, composition, and pressure.

        Args:
            pressure (float): The pressure in *BAR*.
            temp (float): The temperature in *CELSIUS*.
            x (float): The composition of the mobile phase.

        Returns:
            float: The interpolated viscosity.
        """
        if use_julia_fit:
            return viscosity("thf", temp, x, pressure)
        else:
            visc = self.interpolator((temp, x))
            pressure_corr = self.pressure_correction(pressure)
            return visc * pressure_corr

    def compute_molar_fraction_from_percent(self, percent):
        return super().compute_molar_fraction_from_percent(percent, THF)
