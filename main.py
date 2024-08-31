from samples import Sample
import json
import matplotlib.pyplot as plt
from methods import InstrumentMethod, ProcessingMethod

from pydash import get as _get
from injection import Injection
from system import *
import pandas as pd
from data_processing import PeakFinder
from sequence import Sequence
import datetime

import random

random.seed(903)

from cProfile import Profile
from pstats import Stats
import json

import numpy as np

np.seterr(all="warn")


sample_dict = {
    "name": "Calibration Standard",
    "compound_id_list": ["58-55-9", "83-07-8", "1617-90-9"],
    "compound_concentration_list": [0.05, 0.07, 0.1],
}


with open("./system/systems.json") as f:
    system_list = json.load(f)

system = System(**system_list[0])

my_sample = Sample(**sample_dict)
# my_sample.print_compound_list()

with open("./methods/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./methods/processing_methods.json") as f:
    processing_method_list = json.load(f)

validation_method = None
for method in method_list:
    if _get(method, "name") == "column_quality_check":
        validation_method = InstrumentMethod(**method)
        break

validation_processing = None
for method in processing_method_list:
    if _get(method, "name") == "column performance test":
        validation_processing = ProcessingMethod(**method)
        break

curr_sequence = Sequence(
    name="Test",
    datavault="Test",
    start_time=datetime.datetime.now(),
    url=f"Test",
)


# with Profile() as profile:
# for i in range(50):
injection1 = Injection(
    sample=my_sample,
    method=validation_method,
    system=system,
    processing_method=validation_processing,
    sequence=curr_sequence,
)

peak_finder = PeakFinder(
    *injection1.get_chromatogram_data("UV_VIS_1", pandas=False),
    processing_method=validation_processing,
)

# peak = peak_finder[1].get_properties()
peak_finder.plot_peaks(smoothed=True, second_derivative=True, noise=True)
injection1.find_peaks("UV_VIS_1")
injection_json = injection1.to_dict()
# peak_finder.plot_peaks()
# peak_vals.append(peak)
system.inject(count=30)

with open("./injection.json", "w") as f:
    json.dump(injection_json, f)
    # peak_finder.save_peaks("./output.csv")

    # f = open("stats.prof", "a")
    # Stats(profile, stream=f).strip_dirs().sort_stats("cumtime").print_stats()

exit()
peak_list = pd.DataFrame.from_dict(peak_vals)
print(peak_list)
peak_list.to_csv("./peak_list_test.csv", index=False)


# data: pd.DataFrame = injection1.get_chromatogram_data("UV_VIS_1", pandas=True)
# data.to_csv("./chromatogram.csv")


exit()

# method_1 = Method(**_get(method_list, "0"))
method_1 = InstrumentMethod(**_get(method_list, "5"))
# method_2 = Method(**_get(method_list, "1"))
# method_3 = Method(**_get(method_list, "2"))
# method_4 = Method(**_get(method_list, "3"))

injection1 = Injection(sample=my_sample, method=validation_method, system=system)
# injection2 = Injection(sample=my_sample, method=method_2, system=system)
# injection3 = Injection(sample=my_sample, method=method_3, system=system)
# injection4 = Injection(sample=my_sample, method=method_4, system=system)

# injection1.plot_chromatogram("UV_VIS_1", c="red")
# data: pd.DataFrame = injection1.get_chromatogram_data("UV_VIS_1", pandas=True)
# data.to_parquet("./data_processing/test.parquet")
# injection2.plot_chromatogram("UV_VIS_1", h_offset=0, c="blue")
# injection3.plot_chromatogram("UV_VIS_1", h_offset=0, c="black")
# injection4.plot_chromatogram("UV_VIS_1", h_offset=0, c="green")

# injection.plot_chromatogram("UV_VIS_1", c="red")
# injection.plot_chromatogram("UV_VIS_2", h_offset=0.3, v_offset=30, c="blue")
# injection.plot_chromatogram("UV_VIS_3", h_offset=0.6, v_offset=60, c="black")
# injection.plot_chromatogram("UV_VIS_4", h_offset=0.9, v_offset=90, c="green")
# plt.show()

# df = pd.read_parquet("./data_processing/test.parquet")


times, raw_signal = injection1.get_chromatogram_data("UV_VIS_1", pandas=False)

# with Profile() as profile:
peak_finder = PeakFinder(times, raw_signal)

# f = open("stats.prof", "a")
# Stats(profile, stream=f).strip_dirs().sort_stats("tottime").print_stats()

peak_finder.print_peaks()
peak_finder.save_peaks("./output.csv")
peak_finder.plot_peaks()

# with Profile() as profile:
#     for i in range(1):
#         sample_dict = {
#             "sample_name": "test-1",
#             "location": "R:A1",
#             "compound_list": "cystine, guanosine, chloroquine, DPU, coumarin",
#             "concentration_list": "0.5, 2.1, 3.2, 1.3, 2.4",
#             "num_random_peaks": 10,
#             "max_random_concentration": 1,
#         }
#         sample_dict["compound_list"] = [
#             a.strip() for a in sample_dict["compound_list"].split(",")
#         ]
#         sample_dict["concentration_list"] = [
#             float(a) for a in sample_dict["concentration_list"].split(",")
#         ]
#         column = Column(
#             inner_diameter=6,
#             length=100,
#             type="C18",
#             serial_number="1995032",
#             injection_count=0,
#         )
#         system = System(name="James Test", column=column)
#         method = Method(**_get(method_list, i % len(method_list)))
#         my_sample = Sample(**sample_dict)
#         injection = Injection(sample=my_sample, method=method_1, system=system)

#     f = open("stats.prof", "a")
#     Stats(profile, stream=f).strip_dirs().sort_stats("cumtime").print_stats()
