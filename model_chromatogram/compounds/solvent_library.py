import csv
from model_chromatogram.compounds import CompoundLibrary, Solvent


class SolventLibrary(CompoundLibrary):
    def __init__(self) -> None:
        self.compounds: list[Solvent] = []
        with open(
            "./model_chromatogram/compounds/solvents.csv",
            mode="r",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            self.compounds.append(Solvent(**row))


SOLVENT_LIBRARY = SolventLibrary()
