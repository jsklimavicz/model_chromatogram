import json
from pathlib import Path
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import holidays
import concurrent.futures
from pydash import get as get

from model_chromatogram import (
    InstrumentMethod,
    ProcessingMethod,
    Sample,
    Injection,
    Sequence,
    System,
    SampleCreator,
)
from copy import copy
import matplotlib.pyplot as plt


sample_creator = SampleCreator()

with open("./sample_kinetics_testing/kinetics_scitetracine.json", "r") as f:
    kinetics = json.load(f)


conditions = get(kinetics, "conditions")
compound_name_mapping = get(kinetics, "compound_name_mapping")
sample_name_base = get(kinetics, "sample_base")

full_sample_list: list[Sample] = []
initial_date = datetime(2023, 1, 8, 8, 50, 0)

# time_points = [0, 4, *np.arange(7, 365, 7).tolist()]
time_points = [200]

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


folder = "output11"

cmpds = [
    "773-76-2",  # "scitetrazine"
    "61530-11-8",  # "TS-575"
    "67766-78-3",  # "scitetrazine hydrate"
    "1617-90-9",  # "scitetrazine condensation pdt"
    "137-58-6",  # "scitetrazine free acid"
    "42079-78-7",  # "dehydroscitetrazine"
    "95-57-8",  # "TS-576A"
    "492-38-6",  # "TS-576B"
    "13244-35-4",  # "TS-551 amide"
    "4556-23-4",  # "TS-551"
    "51-17-2",  # "des-t-butyl TS-576A"
]
conc = np.ones(len(cmpds)) * 1000


with open("./test/input_json/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./test/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)

with open("./test/input_json/systems.json") as f:
    systems_json = json.load(f)

system = None
for sys in systems_json:
    if get(sys, "name") == "Lonsdale-2":
        system = System(**sys)
        break

scitetrazine_method = None
for method in method_list:
    if get(method, "name") == "scitetrazine_gradient":
        scitetrazine_method = InstrumentMethod(**method)
        break

scitetrazine_processing = None
for method in processing_method_list:
    if get(method, "name") == "scitetracine_stability_quant":
        scitetrazine_processing = ProcessingMethod(**method)
        break

standard = Sample(
    name=f"scitetracine_degradation_standard",
    compound_id_list=cmpds,
    compound_concentration_list=conc,
    concentration_unit=2,
)

standard_low = Sample(
    name=f"scitetracine_degradation_standard_low",
    compound_id_list=cmpds,
    compound_concentration_list=conc / 10,
    concentration_unit=2,
)

blank = Sample("blank")

sequence = Sequence(
    "scitetracine_stability",
    f"{system.name.upper()}",
    start_time=initial_date,
    url="",
)


injection_list = []

for day in time_points:
    print(f"On day {day}; injection list has {len(injection_list)} injections.")
    start_range = datetime(
        initial_date.year, initial_date.month, initial_date.day
    ) + timedelta(days=day)
    time = (
        start_range + timedelta(hours=9) + timedelta(seconds=random.randint(-900, 7200))
    )
    end_range = start_range + timedelta(days=1)
    # run blank injection

    # day's injections
    curr_samples = [blank, standard_low, standard, blank]
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
            curr_samples.append(blank)

    if (
        initial_date + timedelta(days=100)
        <= start_range
        < initial_date + timedelta(days=130)
    ):
        scitetrazine_method.ph = 5.1
    elif (
        initial_date + timedelta(days=130)
        <= start_range
        < initial_date + timedelta(days=160)
    ):
        scitetrazine_method.ph = 5.22
    else:
        scitetrazine_method.ph = 5.3

    for sample in curr_samples:
        if "high_vol" in sample.name:
            inject_vol = 10
        else:
            inject_vol = 1
        scitetrazine_method.set_injection_volume(inject_vol)
        if "standard" in sample.name:
            scitetrazine_method.name = "scitetrazine_standard"
        else:
            scitetrazine_method.name = "scitetrazine_gradient"

        curr_injection = Injection(
            sample=sample,
            method=scitetrazine_method,
            processing_method=scitetrazine_processing,
            sequence=sequence,
            system=system,
            user="Robert Boyle",
            injection_time=time,
        )
        peak_finder = curr_injection.find_peaks("UV_VIS_1")
        injection_list.append(curr_injection)

        time += timedelta(minutes=scitetrazine_method.run_time) + timedelta(
            seconds=random.randint(5, 30)
        )

        # if sample.name == "scitetracine_degradation_standard":
        #     peak_finder.plot_peaks(show_spline=True, highlight_peaks=True)
        #     plt.title(f"{start_range}")
        #     plt.show()

    if system.column.injection_count > 900:
        print(f"Resetting column count on {system.name}")
        system.column.injection_count = 0


folder = "ph_test"
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
