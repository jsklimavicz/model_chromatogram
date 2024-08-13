import os, sys, csv
from pydash import get as _get

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from compounds.compound_library import CompoundLibrary
from compounds.compound import Compound


class SolventLibrary(CompoundLibrary):
    def __init__(self) -> None:
        self.compounds: list[Solvent] = []
        with open("./compounds/solvents.csv", mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            self.compounds.append(Solvent(**row))


class Solvent(Compound):
    def __init__(self, **kwargs) -> None:
        id = [a.strip() for a in kwargs["id_list"].split(",")]
        super().__init__(**kwargs, id=id, logp=0, default_CV=0, asymmetry_addition=0)
        self.hb_acidity = float(kwargs["hb_acidity"])
        self.hb_basicity = float(kwargs["hb_basicity"])
        self.dipolarity = float(kwargs["dipolarity"])
        self.polarity = float(kwargs["polarity"])
        self.dielectric = float(kwargs["dielectric"])
        self.set_concentration(float(_get(kwargs, "density")))


solvent_library = SolventLibrary()
