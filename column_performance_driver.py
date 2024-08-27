from samples import Sample
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

import datetime
import random

# Start and end dates
start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime(2024, 7, 1)

# Define the time range (9 AM to 12 PM)
time_range_start = datetime.time(9, 0)
time_range_end = datetime.time(16, 0)

# Define the delta for weekly intervals
delta = datetime.timedelta(weeks=1)


# Function to generate a random time between 9 AM and 4 PM
def random_time():
    random_hour = random.randint(9, 15)  # Between 9 AM and 3 PM
    random_minute = random.randint(0, 59)  # Any minute
    random_second = random.randint(0, 59)  # Any second
    return datetime.time(random_hour, random_minute, random_second)


# Function to generate a datetime with a random time on Tuesday or Wednesday
def generate_datetime_for_week(current_date):
    # 80% chance for Tuesday, 20% chance for Wednesday
    if random.random() < 0.72:
        day_offset = (1 - current_date.weekday()) % 7  # Tuesday
    else:
        day_offset = (2 - current_date.weekday()) % 7  # Wednesday

    # Calculate the date for Tuesday or Wednesday
    event_date = current_date + datetime.timedelta(days=day_offset)

    # Combine the date with a random time
    return datetime.datetime.combine(event_date, random_time())


# Generate datetimes ensuring they are at least 20 minutes apart
current_date = start_date
minimum_time_difference = datetime.timedelta(minutes=20)

while current_date <= end_date:
    weekly_datetimes = []

    while len(weekly_datetimes) < 7:
        new_datetime = generate_datetime_for_week(current_date)

        # Check if the new datetime is at least 20 minutes apart from all others
        if all(
            abs((new_datetime - dt).total_seconds())
            >= minimum_time_difference.total_seconds()
            for dt in weekly_datetimes
        ):
            weekly_datetimes.append(new_datetime)

    # Print or store the weekly datetimes
    for dt in weekly_datetimes:
        print(dt)

    # Move to the next week
    current_date += delta
