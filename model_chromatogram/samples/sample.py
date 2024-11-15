from model_chromatogram.compounds import Compound, COMPOUND_LIBRARY
from random import uniform
import datetime
from copy import copy
from typing import Literal

ALLOWED_TYPES = [
    "Unknown",
    "Standard",
    "Blank",
    "Check Standard",
    "QC Sample",
    "Calibration Standard",
    "Matrix",
    "Spiked",
    "Unspiked",
    "RF Internal Standard",
]
LITERAL_ALLOWED_TYPES = Literal[
    "Unknown",
    "Standard",
    "Blank",
    "Check Standard",
    "QC Sample",
    "Calibration Standard",
    "Matrix",
    "Spiked",
    "Unspiked",
    "RF Internal Standard",
]


class Sample:

    def __init__(
        self,
        name: str,
        compound_id_list: list[str] | None = None,
        compound_concentration_list: list[float] | None = None,
        creation_date: datetime.datetime | None = None,
        compound_alias: list[str] | None = None,
        concentration_unit=4,
        n_random_named_peaks: int = 0,
        random_named_concentration_range: list[float] = [0, 1],
        n_unknown_peaks: int = 0,
        unknown_concentration_range: list[float] = [0, 1],
        location=None,
        type_: LITERAL_ALLOWED_TYPES = "Unknown",
        **kwargs,
    ) -> None:
        """
        Creates a sample with a list of compounds.

        Args:
            name (str): The name of the sample.
            compound_id_list (list[str]): A list of compound identifiers for finding compounds in the COMPOUND_LIBRARY
            compound_concentration_list (list[float]): A list of concentrations in the same order as the `compound_id_list`.
            creation_date (datetime): Date that the sample was created.
            compound_alias (list[str]): A list of names to label the compounds in the sample as. These are the names that will be used in processing methods to label peaks.
            concentration_unit (int): An int to signify which units are used for the concentration of the input concentrations.
                1: mg/ml (part per thousand w/v)
                2: ug/ml (ppm w/v)
                3: ng/ml (ppb w/v)
                4: umol/ml (or mM)
                5: nmol/ml (or uM)
            n_random_named_peaks (int): Number of random compounds to add to the sample, where each peak is assigned a code name.
            random_named_concentration_range (list[float]): 2-element list containing the min and max concentration values for a randomly added named compound.
            n_unknown_peaks (int): Number of unknown compounds to add to the sample, where each peak is assigned the name "unknown".
            random_unknown_range (list[float]): 2-element list containing the min and max concentration values for unknown compounds.
        """

        self.name: str = name

        self.type = kwargs.get("type") or type_
        if self.type not in ALLOWED_TYPES:
            raise ValueError(
                f"Invalid sample type: {self.type}. Must be one of {LITERAL_ALLOWED_TYPES}"
            )

        if isinstance(creation_date, datetime.datetime):
            self.creation_date = creation_date.isoformat()
        else:
            self.creation_date = creation_date
        if compound_id_list is None:
            compound_id_list = []
        if compound_concentration_list is None:
            compound_concentration_list = []
        self.compounds: list[Compound] = []

        if compound_alias is None:
            compound_alias = [False] * len(compound_id_list)
        for id, conc, alias in zip(
            compound_id_list, compound_concentration_list, compound_alias
        ):
            compound = COMPOUND_LIBRARY.lookup(id)
            compound.set_concentration(conc, unit=concentration_unit)
            if alias:
                compound.id = alias
            self.compounds.append(compound)

        exclude_cas = [a.cas for a in self.compounds]
        random_peaks = []
        unknown_peaks = []
        if n_random_named_peaks > 0:
            random_peaks = COMPOUND_LIBRARY.fetch_random_compounds(
                n_random_named_peaks, exclude_cas, replace_names=True
            )
            for compound in random_peaks:
                compound.set_concentration(
                    uniform(*random_named_concentration_range), unit=concentration_unit
                )
        if n_unknown_peaks > 0:
            unknown_peaks = COMPOUND_LIBRARY.fetch_random_compounds(
                n_unknown_peaks, exclude_cas, set_unknown=True
            )
            for compound in unknown_peaks:
                compound.set_concentration(
                    uniform(*unknown_concentration_range), unit=concentration_unit
                )

        self.compounds = [*self.compounds, *random_peaks, *unknown_peaks]
        self.compounds.sort(key=lambda x: x.intrinsic_log_p)
        self.location = location

    def output_sample_dict(self):
        self_dict = {
            "sample_name": self.name,
            "location": self.location,
            "compounds": ",".join([compound.cas for compound in self.compounds]),
            "concentrations": ",".join(
                [compound.concentration for compound in self.compounds]
            ),
            "random_peaks": 0,
            "max_random_amount": 0,
            "type": self.type,
        }
        return self_dict

    def print_compound_list(self):
        print(
            f'{"compound name": <20}{" "}{"CAS ":<12}\t{"logp "}\t{"MW ":<8}\t{"Conc. "}\t{"mM"}'
        )
        for compound in self.compounds:
            print(
                f'{compound.id: <20}{" "}{compound.cas:<12}\t{round(compound.intrinsic_log_p,3):0.2f}\t{round(compound.mw,3):0.3f}  \t{round(compound.concentration,3):0.3f}\t{round(compound.m_molarity,3):0.3f}'
            )

    def __iter__(self):
        return iter(self.compounds)

    def __copy__(self):
        new_cmpd = Sample(name=self.name)
        for compound in self:
            new_cmpd.compounds.append(compound)
        new_cmpd.name = self.name
        new_cmpd.creation_date = self.creation_date
        new_cmpd.location = self.location
        return new_cmpd

    def __deepcopy__(self):
        new_cmpd = Sample(name=self.name)
        for compound in self:
            new_cmpd.compounds.append(copy(compound))
        new_cmpd.name = self.name
        new_cmpd.creation_date = self.creation_date
        new_cmpd.location = self.location
        return new_cmpd
