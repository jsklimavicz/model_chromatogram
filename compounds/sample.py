from compounds.compound import Compound
from compounds.compound_library import compound_library
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
        self.compounds.sort(key=lambda x: x.default_retention_CV)

    def print_compound_list(self):
        print(
            f'{"compound name": <20}{" "}{"CAS ":<12}\t{"RetV "}\t{"MW ":<8}\t{"Conc. "}\t{"mM"}'
        )
        for compound in self.compounds:
            print(
                f'{compound.id: <20}{" "}{compound.cas:<12}\t{round(compound.default_retention_CV,3):0.2f}\t{round(compound.mw,3):0.3f}  \t{round(compound.concentration,3):0.3f}\t{round(compound.m_molarity,3):0.3f}'
            )
