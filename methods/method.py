import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from solvent_library import solvent_library
from pydash import get as _get
from user_parameters import *
import numpy as np
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
        self.name: str = name
        self.run_time: float = run_time
        self.detection_dict: dict = detection
        self.mobile_phases: list = mobile_phases
        self.update_mobile_phase_dictionary()
        self.mobile_phase_gradient_steps: list = mobile_phase_gradient_steps
        self.create_gradient_profile()
        self.create_composite_profiles()
        self.__dict__ = {**self.__dict__, **kwargs}

    def update_mobile_phase_dictionary(self):
        for ind, mobile_phase in enumerate(self.mobile_phases):
            solvent = solvent_library.lookup(mobile_phase["name"])
            self.mobile_phases[ind]["solvent"] = solvent

    def create_gradient_profile(self):
        raw_profiles = {}
        for ind, step in enumerate(self.mobile_phase_gradient_steps):
            a = _get(step, "percent_a")
            b = _get(step, "percent_b")
            c = _get(step, "percent_c")
            d = _get(step, "percent_d")
            # first check that the step is valid
            if a is None:
                assert (
                    100 >= b + c + d
                ), f"Please make sure that the percentages in step {ind} add up to less than 100%."
            else:
                assert 0.2 > abs(
                    100 - (a + b + c + d)
                ), f"Please make sure that the percentages in step {ind} add up to 100%."
        self.profile_times = np.array(
            [a["time"] for a in self.mobile_phase_gradient_steps]
        )
        for solv_letter in ["b", "c", "d"]:
            raw_profiles[solv_letter] = np.array(
                [a[f"percent_{solv_letter}"] for a in self.mobile_phase_gradient_steps]
            )
        raw_profiles["a"] = np.array(
            [
                100 - (b + c + d)
                for b, c, d in zip(
                    raw_profiles["b"], raw_profiles["c"], raw_profiles["d"]
                )
            ]
        )

        self.profiles = {}
        for solv_letter in ["a", "b", "c", "d"]:
            self.profiles[solv_letter] = Profile(
                self.profile_times, raw_profiles[solv_letter], self.run_time
            )

    def create_composite_profiles(self):
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

    def get_uv_background(self, wavelength, sample_rate=SAMPLE_RATE, zero_time=0):
        composite = np.zeros_like(self.profile_times)
        for solvent in zip(self.mobile_phases):
            solvent = _get(solvent, "0")
            profile = self.profiles[_get(solvent, "id").lower()]
            composite += (
                solvent["solvent"].get_absorbance(wavelength, concentration=1)
                * profile.y
            )
        profile = Profile(self.profile_times, composite, self.run_time)
        return profile.create_profile(sample_rate)


class Profile:
    def __init__(self, x: np.array, y: np.array, length: float) -> None:
        self.x = x
        self.y = y
        self.length = length

    def __convolve(
        self, y, sample_rate, convolution_width=SOLVENT_PROFILE_CONVOLUTION_WIDTH
    ):
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

    def create_profile(self, sample_rate, convolve=True, set_zero_time=True):
        n_points = round(self.length * 60 * sample_rate)
        # account for max time not being a multiple of frequency:
        max_time = n_points / (sample_rate * 60)
        times = np.linspace(0, max_time, n_points)
        y = np.interp(times, self.x, self.y)
        if convolve:
            y = self.__convolve(y, sample_rate)
        if set_zero_time:
            y = y - y[0]
        return times, y


import json
import matplotlib.pyplot as plt

with open("./methods/instrument_methods.json") as f:
    method_list = json.load(f)
method_json = _get(method_list, "0")
method = Method(**method_json)
times, values = method.get_uv_background(230)
plt.plot(times, values, c="red")
plt.show()
print(method)
