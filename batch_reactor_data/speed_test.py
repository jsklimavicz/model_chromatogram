import json
from pathlib import Path
import numpy as np
import pandas as pd
import random
import datetime
import holidays
import concurrent.futures
from pydash import get as get_

from model_chromatogram import (
    InstrumentMethod,
    ProcessingMethod,
    Sample,
    Injection,
    Sequence,
    System,
)

folder = "output11"

cmpds = [
    "79-11-8",  # 3.77         1
    "304-21-2",  # 6.859       2
    "94-59-7",  # 9.837        6
    "93-76-5",  # 8.436        4
    "773-76-2",  # 8.114       3
    "491-78-1",  # 9.235       5
    "58144-64-2",  # 13.514    8
    "58144-68-6",  # 14.195    9
    "90094-11-4",  # 11.989    7
    "117-89-5",  # 17.853      11
    "3075-84-1",  # 16.386     10
]
conc = np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


with open("./sample_kinetics_testing/input_json/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./sample_kinetics_testing/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)


with open("./sample_kinetics_testing/input_json/systems.json") as f:
    systems_json = json.load(f)
system = System(**systems_json[0])

validation_method = None
for method in method_list:
    if get_(method, "name") == "tetracinib_gradient":
        validation_method = InstrumentMethod(**method)
        break

validation_processing = None
for method in processing_method_list:
    if get_(method, "name") == "tetracinib_stability_quant":
        validation_processing = ProcessingMethod(**method)
        break

sample = Sample(
    name=f"tetracinib_degradation_standard",
    compound_id_list=cmpds,
    compound_concentration_list=conc,
    concentration_unit=1,
)

sequence = Sequence(
    "tetracinib_stability",
    f"{system.name.upper()}",
    start_time=datetime.datetime.now(),
    url="",
)


from cProfile import Profile
from pstats import Stats

with Profile() as profile:
    for _ in range(100):
        curr_injection = Injection(
            sample=sample,
            method=validation_method,
            processing_method=validation_processing,
            sequence=sequence,
            system=system,
            user="James",
            injection_time=datetime.datetime.now(),
        )

        peak_finder = curr_injection.find_peaks("UV_VIS_1")

    with open("stats.prof", "w") as f:
        Stats(profile, stream=f).strip_dirs().sort_stats("cumtime").print_stats()

# df: pd.DataFrame = curr_injection.get_chromatogram_data("UV_VIS_2", pandas=True)
# df.to_csv("./sample_kinetics_testing/chromatogram.csv", index=False)
peak_finder.save_peaks("./sample_kinetics_testing/peaks3.csv")
# # peak_finder.plot_peaks(second_derivative=True, first_derivative=True, smoothed=True)

import matplotlib.pyplot as plt

peak_finder.plot_peaks(show_spline=True, highlight_peaks=True)
plt.show()


# plt.savefig("./image.png", transparent=True)
