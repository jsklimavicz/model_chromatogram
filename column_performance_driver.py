from samples import Sample
import json
import matplotlib.pyplot as plt
from methods import InstrumentMethod, ProcessingMethod

from pydash import get as _get
from injection import Injection
from system import *
import pandas as pd
from data_processing import PeakFinder

import random

random.seed(903)

import datetime
import holidays


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

# Generate datetimes ensuring they are at least 20 minutes apart
current_date = start_date
minimum_time_difference = datetime.timedelta(minutes=20)
previous_quarter = None

while current_date <= end_date:
    weekly_datetimes = generate_datetime_set(current_date, len(systems))

    # Determine the current quarter
    current_quarter = (current_date.month - 1) // 3 + 1
    # Check if the quarter has changed
    if current_quarter != previous_quarter:
        sample_dict = {
            "name": "Calibration Standard",
            "compound_id_list": ["58-55-9", "83-07-8", "1617-90-9"],
            "compound_concentration_list": [
                5 + random.uniform(-0.025, 0.025),
                7 + random.uniform(-0.035, 0.035),
                10 + random.uniform(-0.1, 0.1),
            ],
        }
        sequence_name_prefix = f"calibration/{current_date.year}Q{current_quarter}"

    for time, system in zip(weekly_datetimes, systems):
        print(
            f"{system.name.upper()}/calibration/{system.name}_{sequence_name_prefix}_{time}"
        )

    # Move to the next week
    current_date += delta
