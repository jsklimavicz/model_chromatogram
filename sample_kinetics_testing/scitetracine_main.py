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
    "773-76-2",  # "scitetrazine"
    "7245-05-2",  # "TS-575"
    "67766-78-3",  # "scitetrazine hydrate"
    "1617-90-9",  # "scitetrazine condensation pdt"
    "137-58-6",  # "scitetrazine free acid"
    "42079-78-7",  # "dehydrotetracinib"
    "95-57-8",  # "TS-576A"
    "492-38-6",  # "TS-576B"
    "13244-35-4",  # "TS-551 amide"
    "4556-23-4",  # "TS-551"
    "51-17-2",  # "des-t-butyl TS-576A"
]
conc = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


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
    concentration_unit=4,
)

sequence = Sequence(
    "tetracinib_stability",
    f"{system.name.upper()}",
    start_time=datetime.datetime.now(),
    url="",
)


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


# df: pd.DataFrame = curr_injection.get_chromatogram_data("UV_VIS_2", pandas=True)
# df.to_csv("./sample_kinetics_testing/chromatogram.csv", index=False)
peak_finder.save_peaks("./sample_kinetics_testing/peaks3.csv")
# # peak_finder.plot_peaks(second_derivative=True, first_derivative=True, smoothed=True)

import matplotlib.pyplot as plt

peak_finder.plot_peaks(show_spline=True, highlight_peaks=True)
plt.show()


# plt.savefig("./image.png", transparent=True)
