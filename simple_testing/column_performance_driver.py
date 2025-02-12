import json
from pathlib import Path
import random
import datetime
from pydash import get as get_

from model_chromatogram.methods import InstrumentMethod, ProcessingMethod
from model_chromatogram.samples import Sample
from model_chromatogram.injection import Injection
from model_chromatogram.sequence import Sequence
from model_chromatogram.system import System

import matplotlib.pyplot as plt


import cProfile
import pstats
import io
from pstats import SortKey


folder = "./test_output"


with open("./simple_testing/input_json/systems.json") as f:
    systems_json = json.load(f)
systems = [System(**system) for system in systems_json]
system = systems[0]

with open("./simple_testing/input_json/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./simple_testing/input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)

validation_method = None
for method in method_list:
    if get_(method, "name") == "column_quality_check":
        validation_method = InstrumentMethod(**method, system=system)
        break

validation_processing = None
for method in processing_method_list:
    if get_(method, "name") == "column performance test":
        validation_processing = ProcessingMethod(**method)
        break


def save_injections(quarterly_injection_list):
    for injection in quarterly_injection_list:
        inj_dict = injection.to_dict()
        path = f'./{folder}/{get_(inj_dict, "runs.0.sequence.url")}'
        file_name = f'./{folder}/{get_(inj_dict, "runs.0.injection_url")}'
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(inj_dict, f)


# Generate datetimes ensuring they are at least 20 minutes apart
current_date = datetime.datetime(2023, 1, 1, 10, 0, 0)
user = "John Dalton"

sample_dict = {
    "name": "Calibration Standard",
    "compound_id_list": ["58-55-9", "83-07-8", "1617-90-9"],
    "compound_concentration_list": [
        0.2 * random.uniform(0.997, 1.003),
        0.28 * random.uniform(0.997, 1.003),
        0.4 * random.uniform(0.997, 1.003),
    ],
}
sample = Sample(**sample_dict)
sequence = Sequence(
    name="test_sequence", datavault="JSK", start_time=current_date, url="JSK_test"
)

ob = cProfile.Profile()
ob.enable()
for i in range(1):
    for method in method_list:
        if get_(method, "name") == "column_quality_check":
            validation_method = InstrumentMethod(**method, system=system)
            break

    curr_injection = Injection(
        sample=sample,
        method=validation_method,
        processing_method=validation_processing,
        sequence=sequence,
        system=system,
        user=user,
        injection_time=current_date,
    )

    curr_injection.find_peaks("UV_VIS_1")

ob.disable()
sec = io.StringIO()
sortby = SortKey.TIME
ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
ps.print_stats()

with open(f"./{folder}/profile.txt", "w") as f:
    f.write(sec.getvalue())

curr_injection.plot_chromatogram("Pressure")
plt.xlabel("Time (min)")
plt.ylabel("Pressure (bar)")
plt.show()

inj_dict = curr_injection.to_dict()
path = f'./{folder}/{get_(inj_dict, "runs.0.sequence.url")}'
file_name = f'./{folder}/{get_(inj_dict, "runs.0.injection_url")}'
Path(path).mkdir(parents=True, exist_ok=True)
with open(file_name, "w") as f:
    json.dump(inj_dict, f)
