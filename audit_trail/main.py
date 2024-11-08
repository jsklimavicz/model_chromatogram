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


def rmdir(directory):
    directory = Path(directory)
    try:
        for item in directory.iterdir():
            if item.is_dir():
                rmdir(item)
            else:
                item.unlink()
        directory.rmdir()
    except:
        pass


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
        "sample_name_high": "TS-613 - 2 ug/ml standard",
        "sample_name_low": "TS-613 - 0.2 ug/ml standard",
        "sequence_name": "TS-613 SST",
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
        "sample_name_high": "TS-613_standard_high",
        "sample_name_low": "TS-613_standard_low",
        "sequence_name": "TS-613 standards",
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
        "sample_name_high": "TS-613 standard 2 ug",
        "sample_name_low": "TS-613 standard 0.2 ug",
        "sequence_name": "TS-613 calibration",
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
        "sample_name_high": "TS-613 2 ug/ml",
        "sample_name_low": "TS-613 0.2 ug/ml",
        "sequence_name": "TS-613 system suitability",
        "method_name": "TS-613 metabolites",
    },
}

true_blank = Sample(
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


def generate_random_time(date):
    """Generate a random time between 8 AM and 4 PM on a given date."""
    hour = random.randint(8, 15)  # 8 AM to 3 PM for starting hour
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    ms = random.randint(0, 999_999)
    return datetime(date.year, date.month, date.day, hour, minute, second, ms)


folder = "output"
rmdir(Path(folder))
system_list = [system_list[-1]]
for ind, system in enumerate(system_list):
    initial_date = datetime(2023, 2, 13, 8, 0, 0)

    system_name = system.name
    system_lib = method_creation_by_system[system_name]
    sequence_name = "calibration_standards"
    sample_high = Sample(
        system_lib["sample_name_high"],
        cas_list,
        conc_list,
        initial_date,
        concentration_unit=2,
    )
    sample_low = Sample(
        system_lib["sample_name_low"],
        cas_list,
        conc_list,
        initial_date,
        concentration_unit=2,
    )
    set_(instrument_method, "name", system_lib["method_name"])
    set_(instrument_method, "creation", system_lib["method"])
    set_(instrument_method, "last_update", system_lib["method"])
    sequence = Sequence(
        system_lib["sequence_name"],
        system_lib["method"]["data_vault_name"],
        initial_date,
        url=system_lib["method"]["data_vault_name"],
    )

    injection_date = initial_date + timedelta(minutes=np.random.uniform(0, 480))
    default_sys_offset = system.get_retention_time_offset()
    injections = []

    im = InstrumentMethod(**instrument_method)
    pm = ProcessingMethod(**processing_method)

    while initial_date < datetime(2024, 10, 21, 0, 0, 0):
        first_day = random.choice([0, 1])  # Monday = 0, Tuesday = 1
        datetime1 = generate_random_time(initial_date + timedelta(days=first_day))
        second_day = random.randint(first_day + 2, 4)  # Weekday (Wednesday to Friday)
        datetime2 = generate_random_time(initial_date + timedelta(days=second_day))

        for dt in [datetime1, datetime2]:
            user = random.choice(system_lib["users"])
            blank_repeat = False
            random_val = np.random.uniform()
            if (system.name == "Merian" and random_val < 0.1) or random_val < 0.02:
                blank = Sample(
                    "blank",
                    None,
                    None,
                    concentration_unit=2,
                    n_unknown_peaks=np.random.randint(2, 5),
                    unknown_concentration_range=[0.5, 5],
                )
                blank_repeat = True
            else:
                blank = true_blank
            for sample in [blank, sample_low, sample_high]:
                reject_injection = True
                reinject_count = 0
                while reject_injection:
                    inj, peak_count = create_injection(
                        sample,
                        im,
                        pm,
                        sequence,
                        system,
                        user,
                        dt,
                    )
                    inj.find_peaks("UV_VIS_1")
                    injections.append(inj)
                    dt += timedelta(minutes=im.run_time + np.random.uniform(0.05, 0.2))
                    if (peak_count == 0 and sample.name == "blank") or peak_count == 16:
                        test = np.random.uniform()
                        if test < 0.99:
                            reject_injection = False
                        if system.name == "Merian":
                            system.retention_time_offset = 0.07
                    elif system.name == "Merian":
                        system.retention_time_offset = 0.08
                    else:
                        pass
                    if blank_repeat:
                        sample = true_blank
                    if system.column.failed:
                        system.replace_column()

        initial_date += timedelta(days=7)

    print(f"saving injections for {system.name}...")

    for ind, injection in enumerate(injections):
        if ind % 100 == 0:
            print(f"Saving file {ind} out of {len(injections)}")
        inj_dict = injection.to_dict()
        path = f'./{folder}/{get_(inj_dict, "runs.0.sequence.url")}'
        file_name = f'./{folder}/{get_(inj_dict, "runs.0.injection_url")}'
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(inj_dict, f)


# system = system_list[0]
# system.retention_time_offset = 0.085
# a = []

# sample = Sample(
#     "test",
#     cas_list,
#     conc_list,
#     datetime.now(),
#     concentration_unit=2,
# )

# sequence = Sequence("Test", "test", datetime.now(), "test")

# inj = InstrumentMethod(**instrument_method)
# p = ProcessingMethod(**processing_method)

# for i in range(10000):

#     curr_injection = Injection(
#         sample,
#         inj,
#         p,
#         sequence,
#         system,
#         user="James Klimavicz",
#         injection_time=datetime.now(),
#     )
#     peak_finder = curr_injection.find_peaks("UV_VIS_1")
#     a.append(len(peak_finder.peaks.peaks))

#     if i % 100 == 0:
#         print(i)
#         system.replace_column()

# from collections import Counter

# counts = Counter(a)
# count_list = list(counts.items())
# print(count_list)


# plt.hist(a, len(set(a)))
# plt.show()
