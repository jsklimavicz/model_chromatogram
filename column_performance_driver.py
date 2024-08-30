from samples import Sample
import json

# import matplotlib.pyplot as plt
from methods import InstrumentMethod, ProcessingMethod

from pydash import get as _get
from injection import Injection
from system import *
from sequence import Sequence
from pathlib import Path
import os
import numpy as np

# import pandas as pd
# from data_processing import PeakFinder

import random

random.seed(903)

import datetime
import holidays
import concurrent.futures


# Start and end dates
start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime(2024, 7, 1)

# Define the time range (9 AM to 12 PM)
time_range_start = datetime.time(9, 0)
time_range_end = datetime.time(16, 0)

# Define the delta for weekly intervals
delta = datetime.timedelta(weeks=1)
us_holidays = holidays.UnitedStates()


# Function to generate a random time between 9 AM and 4 PM
def random_time():
    random_hour = random.randint(9, 15)  # Between 9 AM and 3 PM
    random_minute = random.randint(0, 59)  # Any minute
    random_second = random.randint(0, 59)  # Any second
    return datetime.time(random_hour, random_minute, random_second)


# Function to generate a datetime with a random time on Tuesday or Wednesday
def generate_datetime_for_week(current_date):
    while True:
        val = random.random()
        if val < 0.623:
            offset = 0
        elif val < 0.841:
            offset = 1
        elif val < 0.91:
            offset = 2
        elif val < 0.964:
            offset = 3
        else:
            offset = 4

        day_offset = (offset - current_date.weekday()) % 7  # Wednesday

        # Calculate the date for Tuesday or Wednesday
        event_date = current_date + datetime.timedelta(days=day_offset)

        if event_date not in us_holidays:
            return datetime.datetime.combine(event_date, random_time())


def generate_datetime_set(current_date, count):
    weekly_datetimes = []
    while len(weekly_datetimes) < count:
        new_datetime = generate_datetime_for_week(current_date)

        # Check if the new datetime is at least 20 minutes apart from all others
        if all(
            abs((new_datetime - dt).total_seconds())
            >= minimum_time_difference.total_seconds()
            for dt in weekly_datetimes
        ):
            weekly_datetimes.append(new_datetime)

    return sorted(weekly_datetimes)


with open("./system/systems.json") as f:
    systems_json = json.load(f)
systems = [System(**system) for system in systems_json]
# systems = [systems[0]]

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


def process_injection(
    time, system, sequence, validation_method, validation_processing, sample, user
):
    sequence.run_blank(
        injection_time=time
        - datetime.timedelta(minutes=validation_method.run_time - 0.2)
    )

    curr_injection = Injection(
        sample=sample,
        method=validation_method,
        processing_method=validation_processing,
        sequence=sequence,
        system=system,
        user=user,
        injection_time=time,
    )
    curr_injection.find_peaks("UV_VIS_1")
    system.inject(random.randrange(24, 38))

    return curr_injection


def save_injections(quarterly_injection_list):
    for injection in quarterly_injection_list:
        inj_dict = injection.to_dict()
        path = f'./output/{_get(inj_dict, "runs.0.sequence.url")}'
        file_name = f'./output/{_get(inj_dict, "runs.0.injection_url")}'
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(inj_dict, f)


# Generate datetimes ensuring they are at least 20 minutes apart
current_date = start_date
minimum_time_difference = datetime.timedelta(minutes=20)
previous_quarter = None
users = ["John Dalton", "Amedeo Avogadro", "Antoine Lavoisier"]
quarterly_injection_list = []
failed_ago = np.zeros(len(systems))

while current_date <= end_date:
    print(current_date)
    weekly_datetimes = generate_datetime_set(current_date, len(systems))

    # Determine the current quarter
    current_quarter = (current_date.month - 1) // 3 + 1
    # Check if the quarter has changed
    if current_quarter != previous_quarter:
        previous_quarter = current_quarter
        year = current_date.year
        # make a new calibration standard
        sample_dict = {
            "name": f"Calibration Standard {year}Q{current_quarter}",
            "compound_id_list": ["58-55-9", "83-07-8", "1617-90-9"],
            "compound_concentration_list": [
                5 * random.uniform(0.99, 1.01),
                7 * random.uniform(0.99, 1.01),
                10 * random.uniform(0.99, 1.01),
            ],
        }
        sample = Sample(**sample_dict)
        # make new sequences
        sequence_name_prefix = f"calibration/{current_date.year}Q{current_quarter}"
        sequences: list[Sequence] = []
        for time, system in zip(weekly_datetimes, systems):
            name = f"{system.name}_calibration/{current_date.year}Q{current_quarter}"
            datavault = system.name.upper()
            curr_sequence = Sequence(
                name=name,
                datavault=system.name.upper(),
                start_time=time,
                url=f"{datavault}/{name}",
            )
            sequences.append(curr_sequence)

        save_injections(quarterly_injection_list)

        quarterly_injection_list = []

    user = random.choice(users)

    # Use ThreadPoolExecutor to parallelize
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_injection,
                time,
                system,
                sequence,
                validation_method,
                validation_processing,
                sample,
                user,
            )
            for time, system, sequence in zip(weekly_datetimes, systems, sequences)
        ]
        for future in concurrent.futures.as_completed(futures):
            quarterly_injection_list.append(future.result())

    for ind, system in enumerate(systems):
        if system.column.failed:
            failed_ago[ind] += 1
        if failed_ago[ind] > 3 and (current_date.year != 2023 or system.name != "Cori"):
            p = random.uniform(0, 0.5) + (failed_ago[ind] - 3) / 10
            if p > 1:
                print(
                    f"Column replaced on {system.name} on {current_date} after {system.column.injection_count} injections."
                )
                system.replace_column()
                failed_ago[ind] = 0

    # Move to the next week
    current_date += delta

save_injections(quarterly_injection_list)
