from compounds.sample import Sample
import json
import matplotlib.pyplot as plt
from methods.method import Method
from chromatogram.chromatogram import Baseline

from pydash import get as _get
from injection import Injection
from system import *

sample_dict = {
    "sample_name": "test-1",
    "location": "R:A1",
    "compound_list": "guanosine, chloroquine, DPU, coumarin",
    "concentration_list": "2.1, 3.2, 1.3, 2.4",
    "num_random_peaks": 5,
    "max_random_concentration": 0.5,
}

sample_dict["compound_list"] = [
    a.strip() for a in sample_dict["compound_list"].split(",")
]
sample_dict["concentration_list"] = [
    float(a) for a in sample_dict["concentration_list"].split(",")
]

column = Column(
    inner_diameter=6,
    length=100,
    type="C18",
    serial_number="1995032",
    injection_count=0,
)
system = System(name="James Test", column=column)

my_sample = Sample(**sample_dict)
my_sample.print_compound_list()


with open("./methods/instrument_methods.json") as f:
    method_list = json.load(f)
method_json = _get(method_list, "0")
method = Method(**method_json)
# times, values = method.get_uv_background(230)
# plt.plot(times, values, c="red")
# times, values = method.get_uv_background(250)
# plt.plot(times, values, c="blue")
# plt.show()
# print(method)

injection = Injection(sample=my_sample, method=method, system=system)
injection.plot_chromatogram("UV_VIS_1", c="red")
injection.plot_chromatogram("UV_VIS_2", h_offset=0.3, v_offset=30, c="blue")
injection.plot_chromatogram("UV_VIS_3", h_offset=0.6, v_offset=60, c="black")
injection.plot_chromatogram("UV_VIS_4", h_offset=0.9, v_offset=90, c="green")

plt.show()
