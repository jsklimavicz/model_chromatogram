from model_chromatogram import (
    Sample,
    SampleCreator,
    Injection,
    InstrumentMethod,
    ProcessingMethod,
    Sequence,
    System,
)
import numpy as np
import json, random
from pydash import get
from datetime import datetime, timedelta
from sample_kinetics_testing.temperature import simulate_room_temperature as get_temp

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from copy import copy, deepcopy

sample_creator = SampleCreator()

with open("./sample_kinetics_testing/kinetics_samples.json", "r") as f:
    kinetics = json.load(f)


conditions = get(kinetics, "conditions")
compound_name_mapping = get(kinetics, "compound_name_mapping")
sample_name_base = get(kinetics, "sample_base")

full_sample_list: list[Sample] = []
initial_date = datetime(2022, 2, 14, 8, 30, 0)

time_points = [*np.arange(371, 460, 7).tolist()]

for condition in conditions:
    sample_name = f"{sample_name_base}_{get(condition, 'conditions')}_day"
    compound_mapping = get(condition, "compound_mapping")
    initial_date += timedelta(minutes=20)
    curr_samples = sample_creator.product_stability_samples(
        time_points=time_points,
        compound_mapping=compound_mapping,
        compound_name_mapping=compound_name_mapping,
        base_name=sample_name,
        start_date=initial_date,
    )
    full_sample_list = [*full_sample_list, *curr_samples]

full_sample_list = sorted(full_sample_list, key=lambda sample: sample.creation_date)

with open("./sample_kinetics_testing/input_json/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./sample_kinetics_testing/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)


with open("./sample_kinetics_testing/input_json/systems.json") as f:
    systems_json = json.load(f)
system_stable = System(**systems_json[0])
system_temps = System(**systems_json[1])
systems = [system_stable, system_temps]

stable_sequence = Sequence(
    name="batch_39930_extended_testing",
    datavault="TETRASCIENCE",
    start_time=initial_date + timedelta(seconds=43),
    url=f"TETRASCIENCE/{system_stable.name.upper()}/tetracinib_stability/batch_39930_extended_testing",
)

temps_sequence = Sequence(
    name="batch_39931_extended_testing",
    datavault=system_temps.name.upper(),
    start_time=initial_date + timedelta(seconds=3679),
    url=f"TETRASCIENCE/{system_temps.name.upper()}/tetracinib_stability/batch_39931_extended_testing",
)
sequences = [stable_sequence, temps_sequence]
users = ["Niels Bohr", "James Maxwell"]

tetrascinib_method = None
for method in method_list:
    if get(method, "name") == "tetracinib_gradient":
        tetrascinib_method = InstrumentMethod(**method)
        break

tetrascinib_processing = None
for method in processing_method_list:
    if get(method, "name") == "tetracinib_stability_quant":
        tetrascinib_processing = ProcessingMethod(**method)
        break


cmpds = [
    "79-11-8",  # 5.444         1
    "304-21-2",  # 9.241      2
    "94-59-7",  # 12.162       6
    "93-76-5",  # 11.325        4
    "773-76-2",  # 9.965       3
    "491-78-1",  # 11,650      5
    "58144-64-2",  # 16.120    8
    "58144-68-6",  # 16.843    9
    "90094-11-4",  # 15.121    7
    "117-89-5",  # 21.111      11
    "3075-84-1",  # 17.898     10
]
conc = np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 10
standard = Sample(
    name=f"tetracinib_degradation_standard",
    compound_id_list=cmpds,
    compound_concentration_list=conc,
    concentration_unit=4,
)

standard_low = Sample(
    name=f"tetracinib_degradation_standard_low",
    compound_id_list=cmpds,
    compound_concentration_list=conc / 10,
    concentration_unit=4,
)

blank_sample = Sample("blank")

injection_list = []

for day in time_points:
    print(f"On day {day}; injection list has {len(injection_list)} injections.")
    start_range = datetime(
        initial_date.year, initial_date.month, initial_date.day
    ) + timedelta(days=day)
    time_1 = (
        start_range + timedelta(hours=9) + timedelta(seconds=random.randint(1, 7200))
    )
    time_2 = (
        start_range + timedelta(hours=9.2) + timedelta(seconds=random.randint(1, 7200))
    )
    times = [time_1, time_2]
    end_range = start_range + timedelta(days=1)
    # run blank injection

    # day's injections
    curr_samples = [blank_sample, standard_low, standard, blank_sample]
    for sample in full_sample_list:
        if start_range <= datetime.fromisoformat(sample.creation_date) < end_range:
            rep2 = copy(sample)
            large_inject_rep1 = copy(sample)
            large_inject_rep2 = copy(sample)
            sample.name += "_rep1"
            rep2.name += "_rep2"
            large_inject_rep1.name += "_high_vol_rep1"
            large_inject_rep2.name += "_high_vol_rep2"
            curr_samples.append(sample)
            curr_samples.append(rep2)
            curr_samples.append(large_inject_rep1)
            curr_samples.append(large_inject_rep2)
            curr_samples.append(blank_sample)

    for sample in curr_samples:
        for system, sequence, user, time in zip(systems, sequences, users, times):
            if system.name == "Hayden":
                # temp = get_temp(time) + 273.15
                temp = 299 + random.uniform(-1, 1)
            else:
                temp = 298 + random.uniform(-1, 1)
            tetrascinib_method.temperature = temp

            if "high_vol" in sample.name:
                inject_vol = 10
            else:
                inject_vol = 1
            tetrascinib_method.set_injection_volume(inject_vol)

            curr_injection = Injection(
                sample=sample,
                method=tetrascinib_method,
                processing_method=tetrascinib_processing,
                sequence=sequence,
                system=system,
                user=user,
                injection_time=time,
            )
            curr_injection.find_peaks("UV_VIS_1")
            # curr_injection.find_peaks("UV_VIS_2")
            injection_list.append(curr_injection)
        new_times = []
        for time in times:
            new_times.append(
                time
                + timedelta(minutes=tetrascinib_method.run_time)
                + timedelta(seconds=random.randint(5, 30))
            )
        times = new_times

        for system in systems:
            if system.column.failed:
                print(f"Replacing column on {system.name}")
                column_var = system.column.inherent_rt_diff
                system.replace_column()
                system.column.inherent_rt_diff = column_var

folder = "output_test_reset"
print("saving injections...")


for ind, injection in enumerate(injection_list):
    if ind % 100 == 0:
        print(f"Saving file {ind} out of {len(injection_list)}")
    inj_dict = injection.to_dict()
    path = f'./{folder}/{get(inj_dict, "runs.0.sequence.url")}'
    file_name = f'./{folder}/{get(inj_dict, "runs.0.injection_url")}'
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(file_name, "w") as f:
        json.dump(inj_dict, f)
