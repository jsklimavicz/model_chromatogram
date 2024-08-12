import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from methods.solvent_library import solvent_library, Solvent
from pydash import get as _get
from user_parameters import *
import numpy as np
from compounds.compound import Compound
from scipy import signal
import pandas as pd


class Method:
    __solvent_ids = ["a", "b", "c", "d"]
    __solvent_percents = [f"percent_{lett}" for lett in __solvent_ids]

    def __init__(
        self,
        name,
        mobile_phases,
        mobile_phase_gradient_steps,
        detection,
        sample_rate=SAMPLE_RATE,
        run_time=RUN_LENGTH,
        **kwargs,
    ) -> None:
        """
        Creates a method class describing the instrument method.

        Args:
            name (str): The name of the instrument method.
            mobile_phases (list): A list of dictionaries for the mobile phases. Each dictionary must have a `name` (str) and an `id` (str); permitted values for `id` are the case-insensitve `a`, `b`, `c`, and `d`. The value for `name` must match a name, ID, or CAS number in the `solvent_library.csv` file.
            mobile_phase_gradient_steps (list): A list of dictionaries outlining the steps in gradient table. Each dictionary can have:
                "time": float
                "flow": float
                "percent_a": float
                "percent_b": float
                "percent_c": float
                "percent_d": float
                "curve": int (1-9)
            where `time`, `flow`, and `percent_b` are mandatory; `percent_a` is determined by subtracting the other percentages from 100, and the `curve` parameter (default: 5) describes the interpolation method between two time points.
            detection: A dictionary containing detection parameters. Current supported is the `uv_vis_parameters` key, where the value is a `list` of `dict`s, where each dict has a `channel_name` (str) and a `wavelength` (float).
            sample_rate (float): Number of samples per second. Default global SAMPLE_RATE.
            run_time (float): Length of the run (in minutes). Default global RUN_LENGTH.
        """
        self.name: str = name
        self.run_time: float = run_time
        self.sample_rate: float = sample_rate
        self.detection: dict = detection
        self.mobile_phases: list = mobile_phases
        self.__update_mobile_phase_dictionary()
        self.gradient_steps: pd.DataFrame = pd.DataFrame.from_dict(
            mobile_phase_gradient_steps
        )
        self.__create_gradient_profile()
        self.__create_composite_profiles()
        self.__dict__ = {**self.__dict__, **kwargs}

    def __update_mobile_phase_dictionary(self):
        """
        Helper function to add a solvent library Compound for each item in the solvent library.
        """
        for ind, mobile_phase in enumerate(self.mobile_phases):
            solvent: Compound = solvent_library.lookup(mobile_phase["name"])
            self.mobile_phases[ind]["solvent"] = solvent

    def __convolve_profile(
        self,
        y: np.array,
        convolution_width: float = SOLVENT_PROFILE_CONVOLUTION_WIDTH,
    ):
        """
        Convolves a signal with a Tukey window to produce a more realistic solvent profile curve, instead of discontinuities in the derivative. Input array is padded with constant values on the right end only to produce an outut with the same length as the input.

        Args:
            y (np.array): Array of samples to convolve
            sample_rate (float): Number of samples per second. Default global SAMPLE_RATE.
            convolution_width (float): Width of the convolution window in minutes. Default global SOLVENT_PROFILE_CONVOLUTION_WIDTH

        Returns:
            filtered (np.array): The convolved signal values
        """
        window_size = int(round(self.sample_rate * 60 * convolution_width))
        window = signal.windows.tukey(window_size)
        y = np.pad(
            y,
            (window_size - 1, 0),
            "constant",
            constant_values=(y[0], y[-1]),
        )
        filtered = signal.convolve(y, window, mode="valid") / sum(window)
        return filtered

    def __create_profile(
        self,
        interp_times,
        grad_x,
        grad_y,
        convolve=True,
        set_zero_time=False,
    ):
        """
        Creates an array of time points and corresponding profile of a solvent or parameter.

        Args:
            interp_times (np.array): array of times to interpolate on.
            grad_x (np.array): array of grandient table time values
            grad_y (np.array): array of grandient table y values
            convolve (bool): Specifies whether to convolve the profile (True) or not (False)
            set_zero_time (bool): Specifies whether to set the start point of the profile to the initial value (True) or not (False)

        Returns:
            y (np.array): The array of signal values

        TODO: Curve parameter input
        TODO: Handling for repeated x-values.

        """
        y = np.interp(interp_times, grad_x, grad_y)
        if convolve:
            y = self.__convolve_profile(y)
        if set_zero_time:
            y = y - y[0]
        return y

    def __create_gradient_profile(self):
        """
        Helper function to make gradient profiles from the method input in dict format from the method json.
        """

        n_points = round(self.run_time * 60 * self.sample_rate)
        # account for max time not being a multiple of frequency:
        max_time = n_points / (self.sample_rate * 60)
        times = np.linspace(0, max_time, n_points)

        self.profile_table = pd.DataFrame({"time": times})

        keys = ["flow", *self.__solvent_percents]
        for name in keys:
            self.profile_table[name] = self.__create_profile(
                times, self.gradient_steps["time"], self.gradient_steps[name]
            )

    def __create_composite_profiles(self):
        """
        Helper function to create profiles for the solvent parameters "hb_acidity", "hb_basicity", "dipolarity", "polarity", and "dielelectric", which affect how quickly compounds elute from the column. This function assumes that there is a linear relation between these parameters and the percentage of solvent, which is unlikely to be true in the real world, but is still a useful approximation here.
        """

        solvents = [_get(solvent, "solvent") for solvent in self.mobile_phases]

        for comp_property in [
            "hb_acidity",
            "hb_basicity",
            "dipolarity",
            "polarity",
            "dielelectric",
        ]:
            comp_values = [s.__dict__[comp_property] for s in solvents]
            self.profile_table[comp_property] = 0
            for mult, name in zip(comp_values, self.__solvent_percents):
                self.profile_table[comp_property] += mult * self.profile_table[name]

    def get_uv_background(self, wavelength, set_zero_time=True):
        """
        Calculates the UV background spectrum based on solvent composition.

        Args:
            wavelength (float): Wavelength at which background should be calculated.
            set_zero_time (bool): Determines whether the background is zeroed to t=0 (default: True),

        Returns:
            time (np.array): Array of time values
            background (np.array): Array of background values at the requested wavelength.
        """

        solvents: list[Solvent] = [
            _get(solvent, "solvent") for solvent in self.mobile_phases
        ]
        comp_values = [s.get_absorbance(wavelength, concentration=1) for s in solvents]
        background = np.zeros_like(self.profile_table["time"])
        for mult, name in zip(comp_values, self.__solvent_percents):
            background += mult * self.profile_table[name]

        if set_zero_time:
            background -= background[0]

        return self.profile_table["time"], background

    def get_times(self):
        """
        Aliased function specifically to return a list of times in the profile table.

        Returns:
            np.array of time values.
        """
        return self.get_profile("time")

    def get_profile(self, profile_name, order=0):
        """
        Returns values of a profile from the profile_table.

        Args:
            profile_name (str): Name of profile to retrieve.
            order (int): Profile curve type to return. Permitted values are:
                1  -- Calculates the profile curve integral
                0  -- Calculates the base profile curve
                -1 -- Calculates the profile curve derivative

        Returns:
            np.array containing the requested profile.
        """
        try:
            if order == 0:
                return self.profile_table[profile_name]
            elif order == 1:  # integral
                vals = np.cumsum(self.profile_table[profile_name])
                dt = 1.0 / (60 * self.sample_rate)
                return vals * dt
            elif order == -1:  # derivative
                vals = np.diff(self.profile_table[profile_name])
                vals = (np.pad(vals, (0, 1), "constant"),)  # pad to same size
                dt = 1.0 / (60 * self.sample_rate)
                return vals / dt
        except Exception:
            columns = "\n".join(self.profile_table.columns)
            print(f"Column not found. Available columnes are {columns}")
            raise
