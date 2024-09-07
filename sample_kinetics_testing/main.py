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
    "79-11-8",  # 5.144
    "304-21-2",  # 11.307
    "94-59-7",  # 11.761
    "93-76-5",  # 12.802
    "773-76-2",  # 13.211
    "491-78-1",  # 13.608
    "58144-64-2",  # 14.927
    "58144-68-6",  # 15.842
    "90094-11-4",  # 16.813
    "117-89-5",  # 17.076
    "3075-84-1",  # 17.393
]
conc = np.array([1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 5


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


sample_dict = {
    "name": f"tetracinib_degradation_standard",
    "compound_id_list": cmpds,
    "compound_concentration_list": conc,
}
sample = Sample(**sample_dict)

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
df: pd.DataFrame = curr_injection.get_chromatogram_data("UV_VIS_2", pandas=True)
df.to_csv("./sample_kinetics_testing/chromatogram.csv", index=False)
peak_finder.save_peaks("./sample_kinetics_testing/peaks.csv")
# peak_finder.plot_peaks(second_derivative=True, first_derivative=True, smoothed=True)
import matplotlib.pyplot as plt

peak_finder.plot_peaks()
plt.show()


# plt.savefig("./image.png", transparent=True)
