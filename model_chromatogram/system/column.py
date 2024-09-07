import numpy as np
import random
from model_chromatogram.user_parameters import (
    DEFAULT_BASE_ASYMMETRY,
    DEFAULT_PEAK_WIDTH,
)
from pydash import get


class Parameters:
    def __init__(self, parameter_dict):
        self.id = get(parameter_dict, "id", 91)
        self.h = get(parameter_dict, "h", 1.03)
        self.s_star = get(parameter_dict, "s_star", 0.01)
        self.a = get(parameter_dict, "a", -0.14)
        self.b = get(parameter_dict, "b", -0.02)
        self.c28 = get(parameter_dict, "c28", 0.08)
        self.c7 = get(parameter_dict, "c7", 0.0)
        self.eb = get(parameter_dict, "eb", 10.1)


class Column:
    def __init__(
        self,
        inner_diameter,
        length,
        type="C18",
        serial_number="1995032",
        injection_count=0,
        failure_risk_count=1000,
        **kwargs
    ) -> None:
        self.inner_diameter = get(inner_diameter, "value")
        self.inner_diameter_unit = get(inner_diameter, "unit")
        self.length = get(length, "value")
        self.length_unit = get(length, "unit")
        self.find_volume()
        self.type = type
        self.serial_number = serial_number
        self.injection_count = injection_count
        self.inherent_asymmetry = random.uniform(-0.005, 0.015)
        self.inherent_broadening = random.uniform(0, DEFAULT_PEAK_WIDTH / 50)
        self.inherent_rt_diff = random.uniform(-0.002, 0.002)
        self.failure_risk_count = failure_risk_count
        self.failed = False
        self.failure_number = 0
        self.failure_asymmetry = 0
        self.failure_broadening = 0
        self.failure_rt_shift_mult = 1
        self.set_parameters(**kwargs)

    def set_parameters(self, **kwargs):
        param_dict = get(kwargs, "parameters")
        self.parameters = Parameters(param_dict)

    def todict(self):
        return {
            "serial_number": self.serial_number,
            "injection_count": self.injection_count,
            "inner_diameter": {
                "value": self.inner_diameter,
                "unit": self.inner_diameter_unit,
            },
            "length": {"value": self.length, "unit": self.length_unit},
            "volume": self.volume,
            "type": self.type,
        }

    def find_volume(self):
        self.volume = (self.inner_diameter / 2) ** 2 * np.pi * self.length
        self.volume /= 1000  # mm^3 to mL conversion

    def inject(self, count=1):
        for i in range(count):
            self.injection_count += 1
            self.__check_for_failure()
        if self.failed:
            self.__set_failure_values()

    def get_column_peak_vals(self):
        broadening = self.inherent_broadening + self.failure_broadening
        asymmetry = self.inherent_asymmetry + self.failure_asymmetry
        rt_shift = self.inherent_rt_diff + self.failure_rt_shift_mult
        return broadening, asymmetry, rt_shift

    def __check_for_failure(self):
        if self.injection_count > 500 and not self.failed:
            x = random.uniform(0, 1)
            if x < 1.5e-4 * (self.injection_count - self.failure_risk_count):
                self.failed = True
                self.failure_number = self.injection_count

    def __set_failure_values(self):
        inconsitency_factor = random.uniform(0.95, 1.05)
        x = self.injection_count - self.failure_number + 1
        mult_value = 0.5 * inconsitency_factor * np.log(1 + ((x / 2 - 1) ** 3) / 1e7)
        self.failure_asymmetry = (DEFAULT_BASE_ASYMMETRY - 1) * 20 * mult_value
        self.failure_broadening = (DEFAULT_PEAK_WIDTH) / 20 * mult_value
        self.failure_rt_shift_mult = 1 + mult_value / 100
