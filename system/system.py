import numpy as np
from system import Column


class System:
    def __init__(
        self, name, column=None, system_retention_time_offset=0, **kwargs
    ) -> None:
        self.name = name
        self.column = Column(**column)
        self.retention_time_offset = system_retention_time_offset
        self.kwargs = {"name": name, "column": self.column.todict(), **kwargs}

    def get_column(self):
        return self.column

    def replace_column(self, column: Column | None = None, serial_number=""):
        if column is None:
            column_dict = self.column.todict()
            column_dict["serial_number"] = serial_number
            self.column = Column(**column_dict)
        else:
            self.column = column

    def get_column_volume(self):
        return self.column.volume

    def get_retention_time_offset(self):
        return self.retention_time_offset

    def inject(self, count=1):
        self.column.inject(count)

    def todict(self):
        d = self.kwargs.copy()
        if "retention_time_offset" in d.keys():
            d.pop("retention_time_offset")
        return d
