from compounds import Sample
import json
import matplotlib.pyplot as plt
from methods import Method

from pydash import get as _get
from injection import Injection
from system import *
import pandas as pd
from data_processing import PeakFinder

import random

random.seed(903)

from cProfile import Profile
from pstats import Stats


sample_dict = {
    "sample_name": "test-1",
    "location": "R:A1",
    # "compound_list": "61530-11-8, 614-47-1, 7364-19-4, 10541-56-7, coumarin",
    # "concentration_list": "2.1, 2.2, 14.12, 2.13, 1.6",
    "compound_list": None,
    "concentration_list": None,
    "num_random_peaks": 10,
    "max_random_concentration": 0.5,
}

# sample_dict["compound_list"] = [
#     a.strip() for a in sample_dict["compound_list"].split(",")
# ]
# sample_dict["concentration_list"] = [
#     float(a) for a in sample_dict["concentration_list"].split(",")
# ]

column = Column(
    inner_diameter=10,
    length=150,
    type="C18",
    serial_number="1995032",
    injection_count=0,
)
system = System(name="James Test", column=column)

my_sample = Sample(**sample_dict)
# my_sample.print_compound_list()

with open("./methods/instrument_methods.json") as f:
    method_list = json.load(f)

# method_1 = Method(**_get(method_list, "0"))
method_1 = Method(**_get(method_list, "5"))
# method_2 = Method(**_get(method_list, "1"))
# method_3 = Method(**_get(method_list, "2"))
# method_4 = Method(**_get(method_list, "3"))

injection1 = Injection(sample=my_sample, method=method_1, system=system)
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
peak_finder = PeakFinder(times, raw_signal)

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
