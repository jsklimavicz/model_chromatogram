from compound import Compound
import csv


class CompoundLibrary:
    def __init__(self) -> None:
        self.compounds = []
        with open("./compounds/compounds.csv", mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            self.compounds.append(Compound(**row))


compound_library = CompoundLibrary()
print(compound_library)
