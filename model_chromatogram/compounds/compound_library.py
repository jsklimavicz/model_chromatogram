import csv
from random import randrange, shuffle
from model_chromatogram.user_parameters import RANDOM_PEAK_ID_DIGITS, IMP_PEAK_PREFIX
from model_chromatogram.compounds import Compound
import warnings
import pickle
from copy import copy
from pathlib import Path


class CompoundLibrary:
    """
    Acts as a library of compounds in the compounds.csv file, allowing lookups and generation of random samples.
    """

    def __init__(self) -> None:
        """
        Creates new instance of library by reading from the csv file. Each line in the csv file must be a valid
        compound for the Compound class.
        """
        self.compounds: list[Compound] = []
        with open(
            "./model_chromatogram/compounds/compounds.csv",
            mode="r",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            self.compounds.append(Compound(**row))

    def __lookup(self, id: str, field: str):
        """
        Internal lookup method with specified field.

        Args:
            id (str): the ID of the compound to look up in the library.
            field (str): the field the look for the ID in.

        Returns:
            out (None|Compound): a `Compound` object if the id is found in the specified field for a compound in the
            compound library; else, None.
        """
        for compound in self.compounds:
            if isinstance(compound.kwargs[field], list):
                if id in compound.kwargs[field]:
                    return compound
            elif compound.kwargs[field] == id:
                return compound
        return None  # no matching ID found

    def lookup(self, id: str, field: str = "id", enforce_field: bool = False):
        """
        Looks up compound in a field.

        Args:
            id (str): the ID of the compound to look up in the library. Must be one of `"id"`, `"name"`, or `"cas"`
            field (str): the field the look for the ID in. Default = `"id"`.
            enforce_field (bool): Determines whether we look *only* in the specified `field` (`True`) or in al fields
            (`False`, default).

        Returns:
            out (None|Compound): a `Compound` object if the id is found in the specified field for a compound in the
            compound library; else,`None`.
        """
        methods = ["id", "name", "cas"]
        try:
            methods.remove(field)
        except ValueError as e:
            print(f"field argument for library lookup must be one of {methods}")
            raise e
        compound = self.__lookup(id, field)
        if compound is not None:
            return compound
        elif enforce_field:
            return None  # no compound found in specified field.
        else:
            for method in methods:
                compound = self.__lookup(id, method)
                if compound is not None:
                    return copy(compound)

    def __create_id(self, prefix: str | None):
        """
        Internal method for creating IDs.

        Args:
            prefix (str): initial string component of a compound ID code.

        Returns:
            id (str): string ID
        """
        id_number = f"{randrange(10**RANDOM_PEAK_ID_DIGITS)}".zfill(
            RANDOM_PEAK_ID_DIGITS
        )
        if prefix is None or prefix == "":
            return f"{id_number}"
        else:
            return f"{prefix}-{id_number}"

    def fetch_random_compounds(
        self,
        count: int,
        cas_exclude_list: list[str],
        set_unknown: bool = False,
        replace_names: bool = False,
        prefix: str = IMP_PEAK_PREFIX,
    ) -> list[Compound]:
        """
        Retrieves a list of compounds from the compound library, with the option to exclude compounds based on CAS
        number.

        Args:
            count (int): The number of compounds to retrieve.
            cas_exclude_list (list[str]): List of CAS numbers to exclude from the random sample.
            set_unknown (bool): If True, returns a list of compounds with "unknown" as the name. This overrides the
                args `replace_names` and `prefix`.
            replace_names (bool): Whether to replace the names in the library with a code, e.g. "ABC-123". The prefrix
                of the code (here, "ABC") is set by `prefix`, while the length of the numeric code is set by
                `RANDOM_PEAK_ID_DIGITS`.
            prefix (str): The initial string component of a compound ID code. The default is set by the user parameter
                `IMP_PEAK_PREFIX`

        Returns:
            compounds (list[Compound]): a list of compounds as requested.
        """
        # TODO implement unknown peaks (no compound IDs)
        shuffle(self.compounds)
        compound_return = []
        for compound in self.compounds:
            if compound.cas not in cas_exclude_list:
                new_compound = copy(compound)
                if set_unknown:
                    new_compound.id = "unknown"
                elif replace_names:
                    new_compound.id = self.__create_id(prefix)
                compound_return.append(new_compound)
            if len(compound_return) == count:
                return compound_return

        warnings.warn(
            "More compounds requested than are available in the library with the given exclusion list. "
            "Returning the compounds available."
        )
        return compound_return


class CompoundLibraryWarning(UserWarning):
    pass


Path("./cache/").mkdir(exist_ok=True)

try:
    with open("./cache/compound_library.obj", "rb") as f:
        COMPOUND_LIBRARY = pickle.load(f)
        if not isinstance(COMPOUND_LIBRARY, CompoundLibrary):
            raise TypeError("Loaded library is not a CompoundLibrary.")
except (TypeError, FileNotFoundError, EOFError):
    # create compound library to call
    COMPOUND_LIBRARY = CompoundLibrary()
    with open("./cache/compound_library.obj", "wb") as f:
        pickle.dump(COMPOUND_LIBRARY, f, pickle.HIGHEST_PROTOCOL)
