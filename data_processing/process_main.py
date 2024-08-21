import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.peak_finder import PeakFinder

df = pd.read_parquet("./data_processing/test.parquet")
raw_signal = df["y"].to_numpy()
times = df["x"].to_numpy()
peak_finder = PeakFinder(times, raw_signal)

peak_finder.print_peaks()
# peak_finder.save_peaks()
peak_finder.plot_peaks()
