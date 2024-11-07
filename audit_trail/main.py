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
from pydash import get as get_, set_
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path


with open("./audit_trail/input_json/instrument_methods.json") as f:
    method_list = json.load(f)
instrument_method = method_list[0]

with open("./audit_trail/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)
processing_method = processing_method_list[0]

with open("./audit_trail/input_json/systems.json") as f:
    systems_json = json.load(f)

system_list = [System(**system) for system in systems_json]

method_creation_by_system = {
    "Wu": {
        "method": {
            "computer": "Wu",
            "comment": "Method for TetraScience metabolite study for TS-613",
            "time": "2023-02-06T09:11:53.04Z",
            "user": "Johann Heinrich Pott",
            "data_vault_name": "TetraScience_PRD/TetraScience/TS-613",
        },
        "users": [
            "Johann Heinrich Pott",
            "Jöns Jacob Berzelius",
        ],
        "sample_name": "TS-613 - 2 ug/ml standard",
        "sequence_name": "calibration",
        "method_name": "TS-613 metabolite analysis",
    },
    "Johnson": {
        "method": {
            "computer": "Johnson",
            "comment": "TetraScience: TS-613 metabolite study",
            "time": "2023-02-09T07:45:12.11Z",
            "user": "Hans Christian Ørsted",
            "data_vault_name": "TS_EMPOWER/TETRASCIENCE/TS-613",
        },
        "users": [
            "Hans Christian Ørsted",
            "Peter Jacob Hjelm",
        ],
        "sample_name": "TS-613_standard",
        "sequence_name": "standards",
        "method_name": "TS-613 metabolites",
    },
    "Mangold": {
        "method": {
            "computer": "Mangold",
            "comment": "TetraScience metabolite study -- TS-613",
            "time": "2023-02-06T15:05:51.40Z",
            "user": "Martin Heinrich Klaproth",
            "data_vault_name": "Enscitra/T/TetraScience/TS-613",
        },
        "users": [
            "Carl Wilhelm Scheele",
            "Johan Gottlieb Gahn",
        ],
        "sample_name": "TS-613 standard",
        "sequence_name": "calibration",
        "method_name": "TS-613 metabolite study",
    },
    "Merian": {
        "method": {
            "computer": "Merian",
            "comment": None,
            "time": "2023-02-11T07:45:36.87Z",
            "user": "William Cruickshank",
            "data_vault_name": "Scitetra_prd/tetrascience/TS-613",
        },
        "users": [
            "William Cruickshank",
            "Eugène-Melchior Péligot",
        ],
        "sample_name": "TS-613 standard",
        "sequence_name": "calibration",
        "method_name": "TS-613 metabolites",
    },
}

blank = Sample(
    name=f"blank",
    compound_id_list=[],
    compound_concentration_list=[],
    concentration_unit=2,
)


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


def create_injection(
    sample, instrument_method, processing_method, sequence, system, user, injection_time
):
    curr_injection = Injection(
        sample,
        instrument_method,
        processing_method,
        sequence,
        system,
        user=user,
        injection_time=injection_time,
    )
    peak_finder = curr_injection.find_peaks("UV_VIS_1")
    named_peak_count = len(
        [peak.name for peak in peak_finder.peaks.peaks if peak.name is not None]
    )
    return (curr_injection, named_peak_count)


# for ind, system in enumerate(system_list):
#     initial_date = datetime(2023, 2, 13, 8, 0, 0)
#     if system_name == "Wu":
#         initial_date += timedelta(days=1)

#     system_name = system.name
#     system_lib = method_creation_by_system[system_name]
#     sequence_name = "calibration_standards"
#     sample = Sample(
#         system_lib["sample_name"],
#         cas_list,
#         conc_list,
#         initial_date,
#         concentration_unit=2,
#     )
#     set_(instrument_method, "name", system_lib["method_name"])
#     set_(instrument_method, "creation", system_lib["method"])
#     set_(instrument_method, "last_update", system_lib["method"])
#     sequence = Sequence(
#         system_lib["sample_name"],
#         system_lib["method"]["data_vault_name"],
#         initial_date,
#         url=system_lib["method"]["data_vault_name"],
#     )

#     injection_date = initial_date + timedelta(minutes=np.random.uniform(0, 480))

#     while initial_date < datetime(2024, 10, 21, 0, 0, 0):
#         pass


system = system_list[0]
system.retention_time_offset = 0.085
a = []

sample = Sample(
    "test",
    cas_list,
    conc_list,
    datetime.now(),
    concentration_unit=2,
)

sequence = Sequence("Test", "test", datetime.now(), "test")

for i in range(1000):

    curr_injection = Injection(
        sample,
        InstrumentMethod(**instrument_method),
        ProcessingMethod(**processing_method),
        sequence,
        system,
        user="James Klimavicz",
        injection_time=datetime.now(),
    )
    peak_finder = curr_injection.find_peaks("UV_VIS_1")
    a.append(len(peak_finder.peaks.peaks))

    if i % 100 == 0:
        print(i)
        system.replace_column()

plt.hist(a, len(set(a)))
plt.show()
