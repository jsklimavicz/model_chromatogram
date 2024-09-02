import numpy as np
import random
from user_parameters import DEFAULT_BASE_ASYMMETRY, DEFAULT_PEAK_WIDTH
from pydash import get as _get


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
        self.inner_diameter = _get(inner_diameter, "value")
        self.inner_diameter_unit = _get(inner_diameter, "unit")
        self.length = _get(length, "value")
        self.length_unit = _get(length, "unit")
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
