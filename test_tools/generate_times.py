import datetime
import holidays
import random

# Start and end dates
start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime(2024, 7, 1)

# Define the time range (9 AM to 12 PM)
time_range_start = datetime.time(9, 0)
time_range_end = datetime.time(16, 0)

# Define the delta for weekly intervals
delta = datetime.timedelta(weeks=1)
us_holidays = holidays.UnitedStates()

MINIMUM_TIME_DIFFERENCE = datetime.timedelta(minutes=20)


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
            >= MINIMUM_TIME_DIFFERENCE.total_seconds()
            for dt in weekly_datetimes
        ):
            weekly_datetimes.append(new_datetime)

    return sorted(weekly_datetimes)
