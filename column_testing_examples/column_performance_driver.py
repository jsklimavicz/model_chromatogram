import json
from pathlib import Path
import numpy as np
import random
import datetime
import holidays
import concurrent.futures
from pydash import get as get_

from model_chromatogram.methods import InstrumentMethod, ProcessingMethod
from model_chromatogram.samples import Sample
from model_chromatogram.injection import Injection
from model_chromatogram.sequence import Sequence
from model_chromatogram.system import System, Column

folder = "output11"

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
    val = random.random()
    if val < 0.217:
        random_hour = 8
    elif val < 0.392:
        random_hour = 9
    elif val < 0.511:
        random_hour = 10
    elif val < 0.564:
        random_hour = 11
    elif val < 0.613:
        random_hour = 12
    elif val < 0.764:
        random_hour = 13
    elif val < 0.897:
        random_hour = 14
    elif val < 0.951:
        random_hour = 15
    else:
        random_hour = 16
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


with open("./input_json/systems.json") as f:
    systems_json = json.load(f)
systems = [System(**system) for system in systems_json]
# systems = [systems[0]]

with open("./input_json/instrument_methods.json") as f:
    method_list = json.load(f)

with open("./input_json/processing_methods.json") as f:
    processing_method_list = json.load(f)

validation_method = None
for method in method_list:
    if get_(method, "name") == "column_quality_check":
        validation_method = InstrumentMethod(**method)
        break

validation_processing = None
for method in processing_method_list:
    if get_(method, "name") == "column performance test":
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
        path = f'./{folder}/{get_(inj_dict, "runs.0.sequence.url")}'
        file_name = f'./{folder}/{get_(inj_dict, "runs.0.injection_url")}'
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
                0.2 * random.uniform(0.997, 1.003),
                0.28 * random.uniform(0.997, 1.003),
                0.4 * random.uniform(0.997, 1.003),
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
        if failed_ago[ind] > 5 and (
            (
                system.column.injection_count - system.column.failure_risk_count > 1600
                or current_date.year != 2023
            )
            or system.name != "Cori"
        ):
            p = random.uniform(0, 0.5) + (failed_ago[ind] - 5) / 5
            if p > 1:
                print(
                    f"Column replaced on {system.name} on {current_date} after {system.column.injection_count} injections."
                )
                system.replace_column()
                failed_ago[ind] = 0

    # Move to the next week
    current_date += delta

save_injections(quarterly_injection_list)
