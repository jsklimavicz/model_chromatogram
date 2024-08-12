import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from methods.solvent_library import solvent_library
from pydash import get as _get
from user_parameters import *
import numpy as np
from compounds.compound import Compound
from scipy import signal


class Method:
    def __init__(
        self,
        name,
        mobile_phases,
        mobile_phase_gradient_steps,
        detection,
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
            run_time: Length of the run (in minutes)
        """
        self.name: str = name
        self.run_time: float = run_time
        self.detection: dict = detection
        self.mobile_phases: list = mobile_phases
        self.__update_mobile_phase_dictionary()
        self.gradient_steps: list = mobile_phase_gradient_steps
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

    def __create_gradient_profile(self):
        """
        Helper function to make gradient profiles from the method input in dict format from the method json.
        """

        def get_solvent_percents(step, solvent_letter):
            """
            Internal function to extract solvent percentages from a step.

            Args:
                step (dict): mobile phase gradient steps item from method dictionary
                solvent_letter (str): sovent to retrieve percentage for. Should be 'a', 'b', 'c', or 'd'.

            Returns:
                percentage (float): The retrieved percentage. Set to 0 if percentage could not be retrieved.
            """
            percentage = _get(step, f"percent_{solvent_letter}")
            return percentage if percentage is not None else 0

        for ind, step in enumerate(self.gradient_steps):
            # retrieve percentages for each solvent in the step.
            a: float = get_solvent_percents(step, "a")
            b: float = get_solvent_percents(step, "b")
            c: float = get_solvent_percents(step, "c")
            d: float = get_solvent_percents(step, "d")
            assert 0.2 > abs(
                100 - (a + b + c + d)
            ), f"Please make sure that the percentages in step {ind} add up to 100%."  # some slack (0.2) allowed for rounding.

        # create the times for the profiles. Times for a-d are identicle so just use a.
        self.profile_times = np.array([step["time"] for step in self.gradient_steps])

        # use submitted values for b-d to set these arrays, and create a dict to hold them
        raw_profiles: dict = {}
        for solv_letter in ["b", "c", "d"]:
            raw_profiles[solv_letter] = np.array(
                [step[f"percent_{solv_letter}"] for step in self.gradient_steps]
            )
        # calculate solvent 'a' by subtraction to ensure the sum is 100 and avoid potential rounding error.
        raw_profiles["a"] = np.array(
            [
                100 - (b + c + d)
                for b, c, d in zip(
                    raw_profiles["b"], raw_profiles["c"], raw_profiles["d"]
                )
            ]
        )

        # set actual profiles for solvent percentages a-d
        self.profiles = {}
        for solv_letter in ["a", "b", "c", "d"]:
            self.profiles[solv_letter] = Profile(
                self.profile_times, raw_profiles[solv_letter], self.run_time
            )

        # create profile for flow
        self.profiles["flow"] = Profile(
            self.profile_times,
            np.array([step["flow"] for step in self.gradient_steps]),
            self.run_time,
        )

    def __create_composite_profiles(self):
        """
        Helper function to create profiles for the solvent parameters "hb_acidity", "hb_basicity", "dipolarity", "polarity", and "dielelectric", which affect how quickly compounds elute from the column. This function assumes that there is a linear relation between these parameters and the percentage of solvent, which is unlikely to be true in the real world, but is still a useful approximation here.
        """
        for comp_property in [
            "hb_acidity",
            "hb_basicity",
            "dipolarity",
            "polarity",
            "dielelectric",
        ]:
            composite = np.zeros_like(self.profile_times)
            for solvent in zip(self.mobile_phases):
                solvent = _get(solvent, "0")
                profile = self.profiles[_get(solvent, "id").lower()]
                composite += solvent["solvent"].__dict__[comp_property] * profile.y
            self.profiles[comp_property] = Profile(
                self.profile_times, composite, self.run_time
            )

    def get_uv_background(
        self, wavelength, sample_rate=SAMPLE_RATE, set_zero_time=True
    ):
        composite = np.zeros_like(self.profile_times)
        for solvent in self.mobile_phases:
            solvent = _get(solvent, "0")
            profile = self.profiles[_get(solvent, "id").lower()]
            composite += (
                solvent["solvent"].get_absorbance(wavelength, concentration=1)
                * profile.y
            )
        profile = Profile(self.profile_times, composite, self.run_time)
        return profile.create_profile(sample_rate, set_zero_time=set_zero_time)


class Profile:
    """
    Handles creation of profiles for solvent gradient composition, flow, and composite pramteers with respect to time.
    """

    def __init__(self, x: np.array, y: np.array, max_time: float) -> None:
        """
        Creates a profile object with time values and paired y values describing the profile.

        Args:
            x (np.array): Array of non-decreasing x values (time in minutes)
            y (np.array): Array of corresponding profile values for each x
            max_time (float): Length of the run (in minutes)
        """
        assert len(x) == len(y), "x and y must be the same length."
        self.x = x
        self.y = y
        self.max_time = max_time

    def __convolve(
        self,
        y: np.array,
        sample_rate: float = SAMPLE_RATE,
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
        window_size = int(round(sample_rate * 60 * convolution_width))
        window = signal.windows.tukey(window_size)
        y = np.pad(
            y,
            (window_size - 1, 0),
            "constant",
            constant_values=(y[0], y[-1]),
        )
        filtered = signal.convolve(y, window, mode="valid") / sum(window)
        return filtered

    def create_profile(
        self, sample_rate=SAMPLE_RATE, order: int = 0, convolve=True, set_zero_time=True
    ):
        """
        Creates an array of time points and corresponding profile of a solvent or parameter.

        Args:
            sample_rate (float): Number of samples per second. Default global SAMPLE_RATE.
            order (int): Profile curve type to return. Permitted values are:
                1  -- Calculates the profile curve integral
                0  -- Calculates the base profile curve
                -1 -- Calculates the profile curve derivative

        Returns:
            times (np.array): The array of times
            y (np.array): The array of signal values

        TODO: Implement integral and derivative methods
        TODO: Curve parameter input
        TODO: Handling for repeated x-values.

        """
        n_points = round(self.max_time * 60 * sample_rate)
        # account for max time not being a multiple of frequency:
        max_time = n_points / (sample_rate * 60)
        times = np.linspace(0, max_time, n_points)
        y = np.interp(times, self.x, self.y)
        if convolve:
            y = self.__convolve(y, sample_rate)
        if set_zero_time:
            y = y - y[0]
        return times, y


# import json
# import matplotlib.pyplot as plt

# with open("./methods/instrument_methods.json") as f:
#     method_list = json.load(f)
# method_json = _get(method_list, "0")
# method = Method(**method_json)
# times, values = method.get_uv_background(230)
# plt.plot(times, values, c="red")
# plt.show()
# print(method)
