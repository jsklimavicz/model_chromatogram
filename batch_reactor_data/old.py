from model_chromatogram import (
    Sample,
    SampleCreator,
    Injection,
    InstrumentMethod,
    ProcessingMethod,
    Sequence,
    System,
    BatchReaction,
)
import numpy as np
import json, random
from pydash import get
from datetime import datetime, timedelta


# Define the reaction network
compound_mapping = [
    {"reactants": ["A", "B"], "products": ["C"], "k": 32},
    {"reactants": ["A", "B"], "products": ["C2"], "k": 0.91},
    {"reactants": ["B"], "products": ["D"], "k": 0.02},
    {"reactants": ["D", "D"], "products": ["D2"], "k": 0.1},
    {"reactants": ["B", "B"], "products": ["M"], "k": 0.48},
    {"reactants": ["B"], "products": ["E1"], "k": 0.014},
    {"reactants": ["E1"], "products": ["B"], "k": 0.01},
    {"reactants": ["B"], "products": ["E2"], "k": 0.013},
    {"reactants": ["E2"], "products": ["B"], "k": 0.015},
    {"reactants": ["A", "D"], "products": ["F"], "k": 0.7},
    {"reactants": ["F"], "products": ["C", "G"], "k": 0.07},
    {"reactants": ["H", "C"], "products": ["I"], "k": 52},
    {"reactants": ["H", "C2"], "products": ["I2"], "k": 67},
    {"reactants": ["I", "H"], "products": ["J"], "k": 0.3},
    {"reactants": ["I", "H"], "products": ["K"], "k": 0.4},
    {"reactants": ["M", "H"], "products": ["N"], "k": 0.71},  # Toxic impurity
    {"reactants": ["B", "H"], "products": ["O"], "k": 0.54},
]

# Compound name mapping
cas_mapping = {
    "A": "4556-23-4",
    "B": "67766-78-3",
    "C": "60683-54-7",
    "C2": "60683-66-1",
    "D": "1120-90-7",
    "D2": "1603-79-8",
    "E1": "56843-54-0",
    "E2": "17015-99-5",
    "F": "7348-39-2",
    "G": "17606-70-1",
    "H": "637-07-0",
    "I": "20264-61-3",
    "I2": "19725-49-6",
    "J": "77-10-1",
    "K": "53451-83-5",
    "M": "54-05-7",
    "N": "72-43-5",
    "O": "117-89-5",
}

# Compound name mapping
compound_name_mapping = {
    "A": "Amine A-391",
    "B": "R-005 isocyanate precursor",
    "C": "TS-302",
    "C2": "epi-TS-302",
    "D": "R-005 urea",
    "D2": "R-005 biuret",
    "E1": "2,3-dehydro-R-005",
    "E2": "3,4-dehydro-R-005",
    "F": "TS-302 urea",
    "G": "oxazol-2-ylamino-TS-304",
    "H": "R-007",
    "I": "TS-304",
    "I2": "epi-TS-304",
    "J": "TS-304 ring closure b",
    "K": "TS-304 ring closure c",
    "M": "R-005 dimer",
    "N": "N-nitroso TS-304",
    "O": "TS-304 N-oxide",
}

cas_no = [value for key, value in cas_mapping.items()]
conc = np.ones(len(cas_no)) * 0.21


with open("./batch_reactor_data/input_json/instrument_methods.json") as f:
    method_list = json.load(f)
method = method_list[0]

with open("./batch_reactor_data/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)
processing_method = processing_method_list[0]

with open("./batch_reactor_data/input_json/systems.json") as f:
    systems_json = json.load(f)
system = System(**systems_json[0])


standard = Sample(
    name=f"TS-304 Std",
    compound_id_list=cas_no,
    compound_concentration_list=conc,
    concentration_unit=4,
)

initial_date = datetime(2022, 4, 15, 8, 54, 38)

sequence = Sequence(
    "TS-304 Std",
    f"{system.name.upper()}",
    start_time=initial_date,
    url="",
)

injection_list = []

curr_injection = Injection(
    sample=standard,
    method=InstrumentMethod(**method),
    processing_method=ProcessingMethod(**processing_method),
    sequence=sequence,
    system=system,
    user="Robert Boyle",
    injection_time=initial_date,
)

peak_finder = curr_injection.find_peaks("UV_VIS_1")
injection_list.append(curr_injection)
curr_injection.plot_chromatogram("UV_VIS_1")
peak_finder.save_peaks()

import matplotlib.pyplot as plt

plt.show()
