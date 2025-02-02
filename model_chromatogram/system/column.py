import numpy as np
import random
import csv
import string
from model_chromatogram.user_parameters import (
    DEFAULT_BASE_ASYMMETRY,
    DEFAULT_PEAK_WIDTH,
)
from pydash import get


def random_column_serial_number():
    digits = "".join(random.choices(string.digits, k=6))
    capital_letter = random.choice(string.ascii_uppercase)
    # Combine them with a hyphen
    random_string = f"00{digits}-{capital_letter}"
    return random_string


def convert_to_mm(value, unit):
    if unit == "cm":
        return value * 10
    elif unit == "in":
        return value * 25.4
    elif unit == "m":
        return value * 1000
    elif unit == "um":
        return value / 1000
    elif unit == "mm":
        return value
    else:
        raise ValueError(f"Unit {unit} for the column is not recognized.")


class Parameters:
    def __init__(self, parameter_dict={}):
        self.parameter_dict = parameter_dict
        self.id = self.__get_float_value("id", 1)
        self.h = self.__get_float_value("h", 1.08)
        self.s_star = self.__get_float_value("s_star", 0.05)
        self.a = self.__get_float_value("a", 0.47)
        self.b = self.__get_float_value("b", 0.06)
        self.c28 = self.__get_float_value("c28", 1.48)
        self.c7 = self.__get_float_value("c7", 1.56)
        self.eb = self.__get_float_value("eb", 10.7)
        self.name = get(parameter_dict, "name", "Zorbax C18")
        self.manufacturer = get(parameter_dict, "manufacturer", "Agilent Technologies")
        self.silica_type = get(parameter_dict, "silica_type", "A")
        self.usp_type = get(parameter_dict, "usp_type", "L1")
        self.phase_type = get(parameter_dict, "phase_type", "C18")

    def __get_float_value(self, field_name, default):
        try:
            return float(get(self.parameter_dict, field_name, default))
        except ValueError:
            return default

    def todict(self):
        return self.parameter_dict


class Column:

    def __init__(
        self,
        inner_diameter,
        length,
        particle_size,
        serial_number=None,
        injection_count=0,
        failure_risk_count=1000,
        porosity=0.65,
        **kwargs,
    ) -> None:
        self.inner_diameter = get(inner_diameter, "value")
        self.inner_diameter_unit = get(inner_diameter, "unit")
        self.id_mm = convert_to_mm(self.inner_diameter, self.inner_diameter_unit)
        self.length = get(length, "value")
        self.length_unit = get(length, "unit")
        self.length_mm = convert_to_mm(self.length, self.length_unit)
        self.particle_size = get(particle_size, "value")
        self.particle_size_unit = get(particle_size, "unit")
        self.particle_diameter_um = 1000 * convert_to_mm(
            self.particle_size, self.particle_size_unit
        )
        self.porosity = porosity
        self.find_volume()
        if serial_number is None:
            self.serial_number = random_column_serial_number()
        else:
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
        param_dict = get(kwargs, "parameters", {})
        parameters_id = get(kwargs, "parameters_id")
        if parameters_id is not None:
            self.parameters = fetch_column_parameters(parameters_id)
        else:
            self.parameters = Parameters(param_dict)

        self.type_ = self.parameters.phase_type
        self.name = self.parameters.name
        self.manufacturer = self.parameters.manufacturer

    def todict(self):
        return {
            "serial_number": self.serial_number,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "injection_count": self.injection_count,
            "inner_diameter": {
                "value": self.inner_diameter,
                "unit": self.inner_diameter_unit,
            },
            "length": {"value": self.length, "unit": self.length_unit},
            "volume": self.nominal_volume,
            "type": self.type_,
        }

    def find_volume(self):
        self.nominal_volume = (self.inner_diameter / 2) ** 2 * np.pi * self.length
        self.nominal_volume /= 1000  # mm^3 to mL conversion
        self.volume = self.nominal_volume * self.porosity  # porosity adjustment

    def inject(self, count=1):
        for _ in range(count):
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


def fetch_column_parameters(id=1) -> Parameters:
    with open(
        "./model_chromatogram/system/columns.csv", "r", encoding="utf-8-sig"
    ) as f:
        reader = csv.reader(f)

        headers = next(reader)  # Extract the first row as column headers
        for row in reader:
            if str(row[0]) == str(id):
                return Parameters(dict(zip(headers, row)))

    # If the ID is not found, return None
    print("Specified column id was not found. Returning a default column.")
    return Parameters()
