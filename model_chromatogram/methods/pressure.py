from model_chromatogram.system import System
import pandas as pd
import numpy as np
from model_chromatogram.methods.viscosity import (
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
        self.column_length = system.column.length_mm * 1e-3  # in m
        self.particle_diameter = system.column.particle_diameter_um * 1e-6  # in m
        self.particle_sphericity = 0.95
        self.porosity = system.column.porosity

        self.solvent_profile = solvent_profile

        # add incremental volume to the profile table
        time_delta = self.solvent_profile["time"][1] - self.solvent_profile["time"][0]
        self.solvent_profile["incremental_CV"] = (
            self.solvent_profile["flow"] * time_delta / system.column.volume
        )
        # flow in cm**3/min, column inner_diameter in mm. Convert to m/s
        self.solvent_profile["superficial_velocity"] = (
            self.solvent_profile["flow"]
            / (system.column.id_mm**2 * np.pi / 400)
            / (100 * 60)
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

    def kozeny_carman_pressure(
        self,
        viscosity: Union[float, np.array],
        velocity: float,
        length: Union[float, np.array],
        permeability_factor=20,
    ) -> float:
        """Calculate the Kozeny-Carman factor.

        Args:
            viscosity (float): The viscosity of the mobile phase (in cP).
            velocity (float): The velocity of the mobile phase (in cm/min).
            length (float): The length of the section (in mm).
        Returns:
            float: The pressure drop across the section (in bar).
        """
        viscosity *= 1e-3  # Pa s (kg m^-1 s^-1)

        # Kozeny–Carman equation for pressure drop
        pressure = (150 * viscosity) / (
            (self.particle_diameter * self.particle_sphericity) ** 2
        )  # (kg m^-3 s^-1)
        pressure *= (1 - self.porosity) ** 2 * velocity * length / (self.porosity**3)
        # (kg m^-3 s^-1) * (m s^-1) * m  = kg m^-1 s^-2 = Pa

        pressure *= permeability_factor

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

    def calculate_pressure_simple(self, tol=1e-5) -> None:
        """Calculate the pressure at each time point, treating the solvent composition as uniform across the column;
        this is an approximate method.

        Args:
            tol (float, optional): The tolerance for the pressure calculation. Defaults to 1e-6.

        Returns:
            np.array(float): The pressure of the system at each time point.
        """
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
                pressure = self.kozeny_carman_pressure(
                    viscosity_guess, v[i], self.column_length
                )
                delta = np.abs(pressure_guess - pressure)
                pressure_guess = pressure

            pressures[i] = pressure_guess

        return pressures

    def calculate_pressure_finite_difference(self, tol=1e-5) -> None:
        """Calculate the pressure at each time point using a finite difference method.

        Args:
            tol (float, optional): The tolerance for the pressure calculation. Defaults to 1e-6.

        Returns:
            np.array(float): The pressure of the system at each time point.
        """
        temp = self.solvent_profile["temperature"].to_numpy()
        v = self.solvent_profile["superficial_velocity"].to_numpy()
        meoh_x = self.solvent_profile["meoh_x"].to_numpy()
        acn_x = self.solvent_profile["acn_x"].to_numpy()
        thf_x = self.solvent_profile["thf_x"].to_numpy()
        incremental_CV = self.solvent_profile["incremental_CV"].to_numpy()

        pressure_guess = self.initial_pressure_guess
        pressures = np.zeros(len(temp))
        viscosities = np.zeros(len(temp))
        for i in range(len(temp)):
            delta = np.inf
            while delta > tol:
                column_fraction_accounted_for = 0
                lower_index = i
                while column_fraction_accounted_for < 1:
                    if lower_index == 0:
                        break
                    lower_index -= 1
                    column_fraction_accounted_for += incremental_CV[lower_index]

                upper_index = i + 1

                vals = self.total_viscosity(
                    meoh_x[lower_index:upper_index],
                    acn_x[lower_index:upper_index],
                    thf_x[lower_index:upper_index],
                    temp[lower_index:upper_index],
                    pressure_guess,
                )

                viscosities[lower_index:upper_index] = vals

                column_fraction_accounted_for = 0
                pressure = 0
                j = i
                while column_fraction_accounted_for < 1:
                    length = self.column_length * incremental_CV[j]
                    incremental_pressure = self.kozeny_carman_pressure(
                        viscosities[j], v[j], length
                    )
                    if j == 0 or column_fraction_accounted_for + incremental_CV[j] > 1:
                        pressure += (
                            incremental_pressure
                            * (1 - column_fraction_accounted_for)
                            / incremental_CV[j]
                        )
                        break
                    else:
                        pressure += incremental_pressure
                    j = max(j - 1, 0)
                    column_fraction_accounted_for += incremental_CV[j]
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

    pressure = pressure_driver.calculate_pressure_finite_difference()

    return pressure
