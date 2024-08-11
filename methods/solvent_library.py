import os, sys, csv
from pydash import get as _get

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from compounds.compound_library import CompoundLibrary
from compounds.compound import Compound


class SolventLibrary(CompoundLibrary):
    def __init__(self) -> None:
        self.solvents: list[Solvent] = []
        with open("./compounds/solvents.csv", mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            self.solvents.append(Solvent(**row))


class Solvent(Compound):
    def __init__(self, **kwargs) -> None:
        id = [a.strip() for a in kwargs["id_list"].split(",")]
        super().__init__(**kwargs, id=id, logp=0, default_CV=0, asymmetry_addition=0)
        self.hb_acidity = float(_get(kwargs, "hb_acidity"))
        self.hb_basicity = float(_get(kwargs, "hb_basicity"))
        self.dipolarity = float(_get(kwargs, "dipolarity"))
        self.polarity = float(_get(kwargs, "polarity"))
        self.dielelectric = float(_get(kwargs, "dielelectric"))


solvent_library = SolventLibrary()
