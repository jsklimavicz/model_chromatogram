import numpy as np
import random
from user_parameters import DEFAULT_BASE_ASYMMETRY, DEFAULT_PEAK_WIDTH


class Column:
    def __init__(
        self,
        inner_diameter,
        length,
        type="C18",
        serial_number="1995032",
        injection_count=0,
        **kwargs
    ) -> None:
        self.inner_diameter = inner_diameter
        self.length = length
        self.find_volume()
        self.type = type
        self.serial_number = serial_number
        self.injection_count = injection_count
        self.failed = False
        self.failure_number = 0
        self.failure_asymmetry = 0
        self.failure_broadening = 0

    def find_volume(self):
        self.volume = (self.inner_diameter / 2) ** 2 * np.pi * self.length
        self.volume /= 1000  # mm^3 to mL conversion

    def inject(self, count=1):
        for i in range(count):
            self.injection_count += 1
            self.__check_for_failure()
        if self.failed:
            self.__set_failure_values()

    def get_column_broadening_and_asymmetry(self):
        return self.failure_broadening, self.failure_asymmetry

    def __check_for_failure(self):
        if self.injection_count > 500 and not self.failed:
            x = random.uniform(0, 1)
            if x < 2e-4 * (self.injection_count - 500):
                self.failed = True
                self.failure_number = self.injection_count

    def __set_failure_values(self):
        inconsitency_factor = random.uniform(0.95, 1.05)
        x = self.injection_count - self.failure_number + 1
        mult_value = np.log(1 + 2 * ((x - 1) / 20) ** 2)
        self.failure_asymmetry = (
            (DEFAULT_BASE_ASYMMETRY - 1) / 3 * mult_value * inconsitency_factor
        )
        self.failure_broadening = (
            (DEFAULT_PEAK_WIDTH) / 10 * mult_value * inconsitency_factor
        )
