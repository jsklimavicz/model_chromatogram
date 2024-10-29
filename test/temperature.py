import numpy as np
from datetime import datetime, timedelta

# Constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_YEAR = 365 * SECONDS_PER_DAY

# Parameters
average_winter_temp = 16  # °C
average_summer_temp = 21  # °C
daily_fluctuation = 2  # °C


# Exponential decay constant (rate of heating/cooling)
def exp_decay(time, tau):
    return np.exp(-time / tau)


def yearly_temp_variation_with_harmonics(date: datetime):
    """Models the average yearly temperature fluctuation with added harmonics"""
    start_of_year = datetime(date.year, 1, 1, 0, 0, 0)
    time_difference = date - start_of_year
    seconds_since_start_of_year = time_difference.total_seconds()
    year_fraction = (seconds_since_start_of_year % SECONDS_PER_YEAR) / SECONDS_PER_YEAR
    temp_amplitude = (average_summer_temp - average_winter_temp) / 2
    seasonal_temp = (
        temp_amplitude * np.sin(2 * np.pi * (year_fraction + 0.75))
        + (average_summer_temp + average_winter_temp) / 2
    )

    # Add some harmonics to make it less regular
    harmonics = 0.5 * np.cos(4 * np.pi * year_fraction + 2542 * 0.1)
    harmonics += 0.3 * np.cos(6 * np.pi * year_fraction + 34582126 * 0.1)
    harmonics -= 0.2 * np.cos(10 * np.pi * year_fraction + 2346457 * 0.1)
    harmonics += 0.2 * np.cos(14 * np.pi * year_fraction + 5682346485 * 0.1)
    harmonics -= 0.15 * np.cos(20 * np.pi * year_fraction + 382348 * 0.1)
    harmonics += 0.15 * np.cos(30 * np.pi * year_fraction + 2843276 * 0.1)
    harmonics -= 0.1 * np.cos(60 * np.pi * year_fraction + 13456823 * 0.1)
    harmonics -= 0.2 * np.cos(36.50 * np.pi * year_fraction + 54723546 * 0.1)
    harmonics += 0.35 * np.cos(18.250 * np.pi * year_fraction + 547432167 * 0.1)

    return seasonal_temp + harmonics


def daily_temp_variation(date):
    """Models daily temperature fluctuation of 5°C with added randomness"""
    start_of_day = datetime(date.year, date.month, date.day, 0, 0, 0)
    time_difference = date - start_of_day
    day_fraction = (time_difference.seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY
    daily_temp = daily_fluctuation * np.sin(2 * np.pi * day_fraction)

    # Adding some higher harmonics (randomness)
    harmonics = 0.5 * np.sin(4 * np.pi * day_fraction + np.random.randn() * 0.1)
    harmonics += 0.3 * np.sin(3 * np.pi * day_fraction + np.random.randn() * 0.1)
    harmonics += 0.3 * np.sin(6 * np.pi * day_fraction + np.random.randn() * 0.1)
    harmonics += 0.05 * np.sin(10 * np.pi * day_fraction + np.random.randn() * 0.1)
    # harmonics -= 0.1 * np.sin(12 * np.pi * day_fraction + np.random.randn() * 0.1)
    harmonics += 2.2 * np.sin(0.25 * np.pi * day_fraction + np.random.randn() * 0.1)
    harmonics -= 1.2 * np.sin(0.125 * np.pi * day_fraction + np.random.randn() * 0.1)
    harmonics += 0.7 * np.sin(np.pi * day_fraction + np.random.randn() * 0.1)

    return daily_temp + harmonics * daily_fluctuation


def simulate_room_temperature(date: datetime):
    """Simulates the temperature fluctuations and AC/Heating system behavior with yearly harmonics"""
    # Seasonal and daily fluctuations
    seasonal_temp = yearly_temp_variation_with_harmonics(date - timedelta(hours=6))
    daily_temp = daily_temp_variation(date - timedelta(hours=6))

    # Combined temperature without AC/heating
    ambient_temp = seasonal_temp + daily_temp

    return ambient_temp
