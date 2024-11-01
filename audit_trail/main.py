from model_chromatogram import (
    Sample,
    Injection,
    InstrumentMethod,
    ProcessingMethod,
    Sequence,
    System,
)
import numpy as np
import json, random
from pydash import get as get_
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path


initial_date = datetime(2022, 4, 14, 0, 0, 0)
curr_date = initial_date

batch_duration = 40  # days


with open("./audit_trail/input_json/instrument_methods.json") as f:
    method_list = json.load(f)
instrument_method = InstrumentMethod(**method_list[0])

with open("./audit_trail/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)
processing_method = ProcessingMethod(**processing_method_list[0])

with open("./audit_trail/input_json/systems.json") as f:
    systems_json = json.load(f)

system_list = [System(**system) for system in systems_json]


blank = Sample(
    name=f"blank",
    compound_id_list=[],
    compound_concentration_list=[],
    concentration_unit=2,
)

users = [
    "Charles Darwin",
    "Galileo Galilei",
    "Albert Einstein",
    "Alice Augusta Ball",
    "Jane Goodall",
    "George Washington Carver",
]

# Define the reaction network
cas_list = [
    "53226-33-8",
    "65-46-3",
    "7348-39-2",
    "36797-84-9",
    "66127-87-5",
    "1762-57-8",
    "98-00-0",
    "121-34-6",
    "56075-45-7",
    "69-72-7",
    "65300-91-6",
    "13672-36-1",
    "35990-93-3",
    "6123-77-9",
    "58144-64-2",
    "531-59-9",
]

conc_list = np.ones(len(cas_list)) * 19.95

sample = Sample("test", cas_list, conc_list, initial_date, concentration_unit=2)

sequence = Sequence("test_sequence", "EMPOWER/test/", initial_date, url="EMPOWER/test/")


system = system_list[-1]
print(system.name)
print(system.column.parameters.id)

curr_injection = Injection(
    sample,
    instrument_method,
    processing_method,
    sequence,
    system_list[0],
    user="James Klimavicz",
    injection_time=initial_date,
)
peak_finder = curr_injection.find_peaks("UV_VIS_1")
curr_injection.plot_chromatogram("UV_VIS_1")
peak_finder.save_peaks()

plt.show()
