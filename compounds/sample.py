from compound import Compound
from compound_library import compound_library
from random import uniform


class Sample:
    def __init__(
        self,
        sample_name: str,
        location: str,
        compound_list: list[str],
        concentration_list: list[float],
        num_random_peaks: int,
        max_random_concentration: float,
    ) -> None:
        self.name: str = sample_name
        self.location: str = location
        self.compounds: list[Compound] = [
            compound_library.lookup(id) for id in compound_list
        ]
        for conc, compound in zip(concentration_list, self.compounds):
            compound.set_concentration(conc)
        exclude_cas = [a.cas for a in self.compounds]
        random_peaks = compound_library.fetch_random_compounds(
            num_random_peaks, exclude_cas, replace_names=True
        )
        for compound in random_peaks:
            compound.set_concentration(
                uniform(max_random_concentration / 10, max_random_concentration)
            )

        self.compounds = [*self.compounds, *random_peaks]

    def print_compound_list(self):
        for compound in self.compounds:
            print(
                f"{compound.id} ({compound.cas}) -- MW: {compound.mw}; concentration: {compound.concentration}, molarity: {compound.m_molarity}"
            )


sample_dict = {
    "sample_name": "test-1",
    "location": "R:A1",
    "compound_list": "guanosine, chloroquine, DPU, coumarin",
    "concentration_list": "2.1, 3.2, 1.3, 2.4",
    "num_random_peaks": 5,
    "max_random_concentration": 0.5,
}

sample_dict["compound_list"] = [
    a.strip() for a in sample_dict["compound_list"].split(",")
]
sample_dict["concentration_list"] = [
    float(a) for a in sample_dict["concentration_list"].split(",")
]

my_sample = Sample(**sample_dict)
my_sample.print_compound_list()
