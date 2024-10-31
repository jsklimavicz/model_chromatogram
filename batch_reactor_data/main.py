from model_chromatogram import (
    Sample,
    Injection,
    InstrumentMethod,
    ProcessingMethod,
    Sequence,
    System,
    BatchReaction,
)
import numpy as np
import json, random
from pydash import get as get_
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path


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

initial_date = datetime(2022, 4, 14, 0, 0, 0)
curr_date = initial_date

batch_duration = 40  # days


with open("./batch_reactor_data/input_json/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./batch_reactor_data/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)
processing_method = processing_method_list[0]

with open("./batch_reactor_data/input_json/systems.json") as f:
    systems_json = json.load(f)

system_list = [System(**system) for system in systems_json]


blank = Sample(
    name=f"blank",
    compound_id_list=[],
    compound_concentration_list=[],
    concentration_unit=4,
)

users = [
    "Charles Darwin",
    "Galileo Galilei",
    "Albert Einstein",
    "Alice Augusta Ball",
    "Jane Goodall",
    "George Washington Carver",
]

while curr_date < datetime(2024, 5, 2, 15, 24, 10):

    # Define the reaction network
    compound_mapping = [
        {"reactants": ["A", "B"], "products": ["C"], "k": 32},
        {"reactants": ["A", "B"], "products": ["C2"], "k": 0.91},
        {"reactants": ["B"], "products": ["D"], "k": 0.02},
        {"reactants": ["D", "D"], "products": ["D2"], "k": 0.1},
        {"reactants": ["B", "B"], "products": ["M"], "k": 0.38},
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
        {"reactants": ["M", "H"], "products": ["N"], "k": 0.41},  # Toxic impurity
        {"reactants": ["B", "H"], "products": ["O"], "k": 0.54},
    ]

    batch_id = f"TS-304_{curr_date.year}{curr_date.month:02d}{curr_date.day:02d}"
    batch_start_date = datetime(curr_date.year, curr_date.month, curr_date.day, 0, 0, 0)

    # batch setup
    initial_concentrations = {
        "A": 0.050 + np.random.normal(0, 0.0002),
    }
    initial_volume = 100 + np.random.normal(0, 0.05)  # Initial volume in liters

    modifier = np.clip(np.random.normal(1, 0.01), 0.95, 1.05)
    N_mod = np.clip(np.random.normal(1.02, 0.02), 0.98, 1.08)
    l = np.random.uniform(0, 1)
    if curr_date > datetime(2024, 4, 1):
        l = 1.7
    else:
        l = 1
    for rxn in compound_mapping:
        rxn["k"] *= modifier * np.clip(np.random.normal(1, 0.02), 0.93, 1.07)
        if (rxn["reactants"] == ["M", "H"]) or rxn["reactants"] == ["B", "B"]:
            rxn["k"] *= N_mod * l

    additions = [
        # Add B over time with volume increase
        {
            "type": "continuous",
            "start_time": 0,
            "end_time": 5,
            "rate": {
                "B": 1.1 * np.clip(np.random.normal(1, 0.02), 0.97, 1.05)
            },  # mol/day added
            "volume_rate": 5,  # L/day added
        },
        {
            "type": "continuous",
            "start_time": 20,
            "end_time": 22,
            "rate": {
                "H": 2.55 * np.clip(np.random.normal(1, 0.02), 0.97, 1.05)
            },  # mol/day added
            "volume_rate": 5,  # L/day added
        },
    ]

    batch_reaction = BatchReaction(
        reaction_map=compound_mapping,
        initial_concentrations=initial_concentrations,
        initial_volume=initial_volume,
        additions=additions,
        start_day=0,
        end_day=41,
        n_points=41 * 24 + 1,
    )

    sequence = Sequence(
        f"{batch_id}",
        f"Empower/TS-304/batch_reaction/",
        start_time=batch_start_date,
        url=f"Empower/TS-304/batch_reaction/{batch_id}",
    )

    injection_list = []
    for day in range(41):
        t1 = day + (8.75 + np.clip(np.random.normal(0, 0.4), -2, 2)) / 24
        t2 = day + (14.5 + np.clip(np.random.normal(0, 0.5), -2, 2)) / 24
        t3 = day + (22.3 + np.clip(np.random.normal(0, 0.3), -2, 2)) / 24
        times = [t1, t2, t3]
        conc_at_times = batch_reaction.get_concentrations_at_times(times)

        for conc_dict, time in zip(conc_at_times, times):
            cas_list = []
            conc_list = []
            for key, value in conc_dict.items():
                value /= 1e3  # injection in ul, not ml
                if value < 1e-9:
                    continue
                cas_list.append(cas_mapping[key])
                conc_list.append(value)

            curr_date = batch_start_date + timedelta(days=time)
            if curr_date > datetime(2023, 5, 7, 10, 0, 0):
                instrument = system_list[np.random.randint(0, len(systems_json))]
            else:
                instrument = system_list[np.random.randint(0, len(systems_json) - 1)]

            if instrument.column.failed:
                instrument.replace_column()

            instrument_name = instrument.name

            instrument_method = None
            for method in method_list:
                if get_(
                    method, "creation.computer"
                ) == instrument_name and "minor" in get_(method, "name"):
                    instrument_method_impurities = method
                elif get_(method, "creation.computer") == instrument_name:
                    instrument_method_main = method

            user = random.choice(users)
            curr_date += timedelta(minutes=np.random.uniform(10, 20))

            blank_injection = Injection(
                sample=blank,
                method=InstrumentMethod(**instrument_method_main),
                processing_method=ProcessingMethod(**processing_method),
                sequence=sequence,
                system=instrument,
                user=user,
                injection_time=curr_date,
            )
            injection_list.append(blank_injection)

            for instrument_method in [
                instrument_method_main,
                instrument_method_impurities,
            ]:
                curr_date += timedelta(
                    minutes=instrument_method_main["run_time"] + 0.21
                )
                samp = Sample(
                    name=f"{batch_id}_day{day}_hour_{t1//1}",
                    compound_id_list=cas_list,
                    compound_concentration_list=conc_list,
                    concentration_unit=4,
                )

                curr_injection = Injection(
                    sample=samp,
                    method=InstrumentMethod(**instrument_method),
                    processing_method=ProcessingMethod(**processing_method),
                    sequence=sequence,
                    system=instrument,
                    user=user,
                    injection_time=curr_date,
                )

                peak_finder = curr_injection.find_peaks("UV_VIS_1")
                curr_injection.plot_chromatogram("UV_VIS_1")
                for peak in peak_finder.peaks.peaks:
                    if peak.name and "nitroso" in peak.name:
                        if peak.relative_amount > 0.1:
                            curr_injection = Injection(
                                sample=samp,
                                method=InstrumentMethod(**instrument_method),
                                processing_method=ProcessingMethod(**processing_method),
                                sequence=sequence,
                                system=instrument,
                                user=user,
                                injection_time=curr_date,
                            )
                            peak_finder = curr_injection.find_peaks("UV_VIS_1")
                            curr_injection.plot_chromatogram("UV_VIS_1")
                        print(f"{time:.3f}:\t {peak.relative_amount}")

                injection_list.append(curr_injection)

    curr_date += timedelta(days=2)
    folder = "batch_data"
    print("saving injections...")

    for ind, injection in enumerate(injection_list):
        if ind % 100 == 0:
            print(f"Saving file {ind} out of {len(injection_list)}")
        inj_dict = injection.to_dict()
        path = f'./{folder}/{get_(inj_dict, "runs.0.sequence.url")}'
        file_name = f'./{folder}/{get_(inj_dict, "runs.0.injection_url")}'
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(inj_dict, f)
