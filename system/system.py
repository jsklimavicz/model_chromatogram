import numpy as np
from system import Column


class System:
    def __init__(self, name, column=None) -> None:
        self.name = name
        self.column = column

    def set_column(self, column: Column):
        self.column = column

    def get_column(self):
        return self.column

    def get_column_volume(self):
        return self.column.volume

    def inject(self):
        self.column.inject()
