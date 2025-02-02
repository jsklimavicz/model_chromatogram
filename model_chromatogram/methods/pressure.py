from model_chromatogram.system import System
import pandas as pd
import numpy as np
from model_chromatogram.methods.pressure_calculator import (
    MethanolViscosity,
    AcetonitrileViscosity,
    TetrahydrofuranViscosity,
)
from typing import Union


class PressureDriver:
    initial_pressure_guess = 200  # in bar

    meoh_viscosity = MethanolViscosity()
    acn_viscosity = AcetonitrileViscosity()
    thf_viscosity = TetrahydrofuranViscosity()

    def __init__(self, system: System, solvent_profile: pd.DataFrame) -> None:
        self.column = system.column
        self.column_length = self.column.length * 1e-3  # in m
        self.particle_diameter = 5 * 1e-6  # in m
        self.particle_sphericity = 1
        self.porosity = self.column.porosity

        self.solvent_profile = solvent_profile

        # add incremental volume to the profile table
        time_delta = self.solvent_profile["time"][1] - self.solvent_profile["time"][0]
        self.solvent_profile["incremental_CV"] = (
            self.solvent_profile["flow"] * time_delta / self.column.volume
        )
        self.solvent_profile["superficial_velocity"] = self.solvent_profile["flow"] / (
            self.column.inner_diameter**2 * np.pi / 4
        )
        self.solvent_profile["meoh_x"] = (
            self.meoh_viscosity.compute_molar_fraction_from_percent(
                self.solvent_profile["percent_meoh"]
            )
        )
        self.solvent_profile["acn_x"] = (
            self.thf_viscosity.compute_molar_fraction_from_percent(
                self.solvent_profile["percent_acn"]
            )
        )
        self.solvent_profile["thf_x"] = (
            self.thf_viscosity.compute_molar_fraction_from_percent(
                self.solvent_profile["percent_thf"]
            )
        )

        # print(self.solvent_profile.head(20))

    def kozney_carman_pressure(
        self,
        viscosity: Union[float, np.array],
        velocity: float,
        length: Union[float, np.array],
    ) -> float:
        """Calculate the Kozney-Carman factor.

        Args:
            viscosity (float): The viscosity of the mobile phase (in cP).
            velocity (float): The velocity of the mobile phase (in cm/min).
            length (float): The length of the section (in mm).
        Returns:
            float: The pressure drop across the section (in bar).
        """
        viscosity *= 1e-3  # Pa s
        velocity *= 1e-2 / 60  # m/s

        # Kozeny–Carman equation for pressure drop
        pressure = (150 * viscosity) / (
            (self.particle_diameter * self.particle_sphericity) ** 2
        )
        pressure *= (1 - self.porosity) ** 2 * velocity / (length * self.porosity**3)

        return pressure * 1e-5  # convert to bar

    def total_viscosity(self, meoh_x, acn_x, thf_x, temp, pressure) -> float:
        """Calculate the total viscosity of the mobile phase.

        Args:
            meoh_x (float): The molar fraction of methanol.
            acn_x (float): The molar fraction of acetonitrile.
            thf_x (float): The molar fraction of tetrahydrofuran.
            temp (float): The temperature of the mobile phase (in Celsius).
            pressure (float): The pressure of the mobile phase (in bar).

        Returns:
            float: The viscosity of the mobile phase (in cP).
        """
        meoh_nu = self.meoh_viscosity.interpolate_viscosity(pressure, temp, meoh_x)
        acn_nu = self.acn_viscosity.interpolate_viscosity(pressure, temp, acn_x)
        thf_nu = self.thf_viscosity.interpolate_viscosity(pressure, temp, thf_x)
        water_meoh_x = 1 - (acn_x + thf_x)
        return np.exp(
            np.log(meoh_nu) * water_meoh_x
            + np.log(acn_nu) * acn_x
            + np.log(thf_nu) * thf_x
        )

    def calculate_pressure_simple(self, tol=1e-6) -> None:
        """Calculate the pressure of a single timepoint segment."""
        temp = self.solvent_profile["temperature"].to_numpy()
        v = self.solvent_profile["superficial_velocity"].to_numpy()
        meoh_x = self.solvent_profile["meoh_x"].to_numpy()
        acn_x = self.solvent_profile["acn_x"].to_numpy()
        thf_x = self.solvent_profile["thf_x"].to_numpy()

        pressure_guess = self.initial_pressure_guess
        pressures = np.zeros(len(temp))
        for i in range(len(temp)):
            delta = np.inf
            while delta > tol:
                viscosity_guess = self.total_viscosity(
                    meoh_x[i], acn_x[i], thf_x[i], temp[i], pressure_guess
                )
                pressure = self.kozney_carman_pressure(
                    viscosity_guess, v[i], self.column_length
                )
                delta = np.abs(pressure_guess - pressure)
                pressure_guess = pressure

            pressures[i] = pressure_guess

        return pressures


def calculate_pressure(
    mobile_phase: dict, profile_table: pd.DataFrame, system: System
) -> float:
    """Calculate the pressure of the system.

    Args:
        mobile_phase (dict): The mobile phase dictionary.
        profile_table (dict): The profile table dictionary.
        system (System): The system object.

    Returns:
        np.array(float): The pressure of the system at each time point.
    """

    # first calculate which solvents are used in the mobile phase.
    # first calculate which solvents are used in the mobile phase.

    profile_table_copy = profile_table.copy(deep=True)

    for solvent in mobile_phase:
        old_name = f"percent_{solvent['id'].lower()}"
        name = solvent["name"].lower()
        if name in ["water", "h2o", "buffer"]:
            new_name = "percent_water"
        elif name in ["acetonitrile", "acn"]:
            new_name = "percent_acn"
        elif name in ["methanol", "meoh"]:
            new_name = "percent_meoh"
        elif name in ["thf", "tetrahydrofuran"]:
            new_name = "percent_thf"
        else:
            raise ValueError(f"Unknown solvent: {solvent["name"]}")
        profile_table_copy.rename(columns={old_name: new_name}, inplace=True)

    pressure_driver = PressureDriver(system, profile_table_copy)

    pressure = pressure_driver.calculate_pressure_simple()

    return pressure
