import csv
from random import randrange, shuffle
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from compounds.compound import Compound
from user_parameters import RANDOM_PEAK_ID_DIGITS, IMP_PEAK_PREFIX


class CompoundLibrary:
    def __init__(self) -> None:
        self.compounds: list[Compound] = []
        with open("./compounds/compounds.csv", mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            self.compounds.append(Compound(**row))

    def __lookup(self, id: str, field: str):
        for compound in self.compounds:
            if compound.kwargs[field] == id:
                return compound
        return None  # no matching ID found

    def lookup(self, id: str, field: str = "id", enforce_field: bool = False):
        methods = ["id", "name", "cas"]
        try:
            methods.remove(field)
        except ValueError as e:
            print(f"field argument for library lookup must be one of {methods}")
            raise
        compound = self.__lookup(id, field)
        if compound is not None:
            return compound
        elif enforce_field:
            return None  # no compound found in specified field.
        else:
            for method in methods:
                compound = self.__lookup(id, method)
                if compound is not None:
                    return compound

    def __create_id(self, prefix: str = None):
        if prefix is None:
            prefix = IMP_PEAK_PREFIX
        id_number = f"{randrange(10**RANDOM_PEAK_ID_DIGITS)}".zfill(
            RANDOM_PEAK_ID_DIGITS
        )
        return f"{prefix}-{id_number}"

    def fetch_random_compounds(
        self,
        count: int,
        cas_exclude_list: list[str],
        replace_names: bool = False,
        prefix: str = None,
    ) -> list[Compound]:
        shuffle(self.compounds)
        compound_return = []
        for compound in self.compounds:
            if compound.cas not in cas_exclude_list:
                if replace_names:
                    compound.id = self.__create_id(prefix)
                compound_return.append(compound)
            if len(compound_return) == count:
                return compound_return


compound_library = CompoundLibrary()
# print(compound_library)
