from compounds.sample import Sample
import json
import matplotlib.pyplot as plt
from methods.method import Method
from chromatogram.chromatogram import Baseline

from pydash import get as _get
from injection import Injection

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

injection = Injection(sample=my_sample, method=method)
