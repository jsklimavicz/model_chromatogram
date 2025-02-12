import os
from model_chromatogram.user_parameters import JULIA_PARAMTERS

if JULIA_PARAMTERS["julia"] is not None:
    os.environ["PYTHON_JULIAPKG_EXE"] = JULIA_PARAMTERS["julia"]

    from juliacall import Main as jl

    jl.include("./model_chromatogram/methods/pressure.jl")
    use_julia_fit = True
else:
    use_julia_fit = False


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

    def __init__(
        self,
        system: System,
        solvent_profile: pd.DataFrame,
        column_permeability_factor=20,
    ) -> None:
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
        self.solvent_profile["incremental_length"] = (
            self.solvent_profile["incremental_CV"] * self.column_length
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

        self.kozeny_carman_multiplier = (
            column_permeability_factor
            * 150
            / (self.particle_diameter * self.particle_sphericity) ** 2
            * (1 - self.porosity) ** 2
            / (self.porosity**3)
        )

    def kozeny_carman_pressure(
        self,
        viscosity: Union[float, np.array],
        velocity: float,
        length: Union[float, np.array],
    ) -> float:
        """Calculate the Kozeny-Carman factor.

        Args:
            viscosity (float): The viscosity of the mobile phase (in cP).
            velocity (float): The velocity of the mobile phase (in cm/min).
            length (float): The length of the section (in mm).
        Returns:
            float: The pressure drop across the section (in bar).
        """
        # Kozeny–Carman equation for pressure drop
        pressure = self.kozeny_carman_multiplier * viscosity * velocity * length

        return pressure * 1e-8  # convert to bar

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

        if acn_x == 0:
            acn_nu = 1
        else:
            acn_nu = self.acn_viscosity.interpolate_viscosity(pressure, temp, acn_x)
        if thf_x == 0:
            thf_nu = 1
        else:
            thf_nu = self.thf_viscosity.interpolate_viscosity(pressure, temp, thf_x)

        if meoh_nu == thf_nu == acn_nu == 1:
            return self.meoh_viscosity.interpolate_viscosity(pressure, temp, meoh_x)

        else:
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

    def calculate_pressure_finite_difference(
        self, tol=1e-5, recalculate_viscosities=False
    ) -> None:
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
        incremental_lengths = self.solvent_profile["incremental_length"].to_numpy()

        pressure_guess = self.initial_pressure_guess
        pressures = np.zeros(len(temp))
        viscosities = np.zeros(len(temp))
        for i in range(len(temp)):
            delta = np.inf

            column_fraction_accounted_for = 0
            lower_index = i
            column_unaccounted_for = 0
            column_overaccounted_for = 0
            while column_fraction_accounted_for < 1:
                column_fraction_accounted_for += incremental_CV[lower_index]
                if column_fraction_accounted_for >= 1:
                    column_overaccounted_for = column_fraction_accounted_for - 1
                    break
                if lower_index == 0:
                    column_unaccounted_for = 1 - column_fraction_accounted_for
                    break
                lower_index -= 1

            upper_index = i + 1

            while delta > tol:
                if recalculate_viscosities:
                    # viscosities for each segment
                    curr_viscosities = self.total_viscosity(
                        meoh_x[lower_index:upper_index],
                        acn_x[lower_index:upper_index],
                        thf_x[lower_index:upper_index],
                        temp[lower_index:upper_index],
                        pressure_guess,
                    )
                else:
                    viscosities[i] = self.total_viscosity(
                        meoh_x[i],
                        acn_x[i],
                        thf_x[i],
                        temp[i],
                        pressure_guess,
                    )
                    curr_viscosities = viscosities[lower_index:upper_index].copy()

                # incremental pressures for each segment
                incremental_pressures = self.kozeny_carman_pressure(
                    curr_viscosities,
                    v[lower_index:upper_index],
                    incremental_lengths[lower_index:upper_index],
                )

                pressure = np.sum(incremental_pressures)
                if column_unaccounted_for > 0:
                    pressure += (
                        column_unaccounted_for
                        / (incremental_CV[lower_index])
                        * incremental_pressures[0]
                    )
                elif column_overaccounted_for > 0:
                    pressure -= (
                        column_overaccounted_for
                        / (incremental_CV[lower_index])
                        * incremental_pressures[-1]
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

    col_struct = [
        system.column.length_mm * 1e-3,
        system.column.particle_diameter_um * 1e-6,
        0.95,
        system.column.porosity,
        system.column.volume,
        system.column.id_mm,
    ]
    if use_julia_fit:
        pressure = jl.Pressure.pressure_driver(profile_table_copy, col_struct)

    else:
        pressure_driver = PressureDriver(system, profile_table_copy)
        pressure = pressure_driver.calculate_pressure_finite_difference()
    return pressure
