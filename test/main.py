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

import concurrent.futures
from pathlib import Path
import matplotlib.pyplot as plt

sample_creator = SampleCreator()

with open("./sample_kinetics_testing/kinetics_scitetracine.json", "r") as f:
    kinetics = json.load(f)


conditions = get(kinetics, "conditions")
compound_name_mapping = get(kinetics, "compound_name_mapping")
sample_name_base = get(kinetics, "sample_base")

for condition in conditions:
    title = f"{sample_name_base}_{get(condition, 'conditions')}"
    compound_mapping = get(condition, "compound_mapping")
    sample_creator.plot_kinetics(
        compound_mapping,
        start_day=0,
        end_day=365,
        title=title,
        compound_name_mapping=compound_name_mapping,
    )
    plt.show()
