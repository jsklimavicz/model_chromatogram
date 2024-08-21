import math


class Column:
    def __init__(
        self,
        inner_diameter,
        length,
        type="C18",
        serial_number="1995032",
        injection_count=0,
    ) -> None:
        self.inner_diameter = inner_diameter
        self.length = length
        self.find_volume()
        self.type = type
        self.serial_number = serial_number
        self.injection_count = injection_count

    def find_volume(self):
        self.volume = (self.inner_diameter / 2) ** 2 * math.pi * self.length
        self.volume /= 1000  # mm^3 to mL conversion

    def inject(self):
        self.injection_count += 1
