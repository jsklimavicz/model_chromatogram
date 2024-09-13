import os, json
import pandas as pd
from pydash import get

# data = []
# root_dir = "./ph_test"


# def extract_data_from_json(data):
#     curr_data = []
#     # Extract specific fields using pydash.get
#     extracted_data = {
#         "system_name": get(data, "systems.0.name"),
#         "user": get(data, "users.0.name"),
#         "injection_name": get(data, "methods.0.injection.name"),
#         "processing_name": get(data, "methods.0.processing.name"),
#         "sample_name": get(data, "samples.0.name"),
#         "injection_time": get(data, "runs.0.injection_time"),
#         "sequence_name": get(data, "runs.0.sequence.name"),
#     }

#     # Extract all peaks under "results.0.peaks.*" using pydash.get
#     peaks = get(data, "results.0.peaks", [])
#     for peak in peaks:
#         peak_data = extracted_data.copy()
#         for peak_name, peak_value in peak.items():
#             peak_data[f"{peak_name}"] = peak_value
#         curr_data.append(peak_data)

#     return curr_data


# all_data = []
# for subdir, dirs, files in os.walk(root_dir):
#     for file in files:
#         if file.endswith(".json"):
#             file_path = os.path.join(subdir, file)
#             with open(file_path, "r") as f:
#                 json_data = json.load(f)
#             extracted_data = extract_data_from_json(json_data)
#             all_data = [*all_data, *extracted_data]

# df = pd.DataFrame.from_dict(all_data)
# df.to_csv("./ph_testing.csv", index=False)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame
df = pd.read_csv("ph_testing.csv")

# Convert 'injection_time' to datetime
df["injection_time"] = pd.to_datetime(df["injection_time"])

# Get the list of unique system names
system_names = df["system_name"].unique()

# Set up the subplots in a 4x2 grid
fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Determine the global y-axis limits
y_min = df["resolution_usp_next_main"].min()
y_max = df["resolution_usp_next_main"].max()

# Iterate over system names and create a subplot for each
for i, system_name in enumerate(system_names):
    ax = axes[i]
    sns.scatterplot(
        data=df[df["system_name"] == system_name],
        x="injection_time",
        y="resolution_usp_next_main",
        hue="name",
        ax=ax,
    )
    ax.set_title(f"Resolution over Time - {system_name}")
    ax.set_xlabel("Injection Time")
    ax.set_ylabel("Resolution USP")
    ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits
    ax.legend(title="Column Serial Number", loc="lower left", bbox_to_anchor=(0, 0))
    ax.tick_params(axis="x", rotation=45)

# Hide any unused subplots (if there are fewer than 8 system names)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()
plt.savefig("Resolution_over_Time_by_System.png")


# Creating the second figure for "Cori" system with a single legend
columns_to_plot = [
    "area",
    "height",
    "retention_time",
    "width_50_full",
    "width_10_full",
    "width_5_full",
    "moment_1",
    "moment_2",
    "moment_3_standardized",
    "asymmetry_USP",
    "resolution_usp_next_main",
    "plates_USP",
]

# Filter the data for system "Cori"
df_cori = df[
    (df["name"] == "TS-576B") & (df["injection_name"] == "scitetrazine_standard")
]

# Set up the subplots in a 4x3 grid
fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
axes = axes.flatten()

# Iterate over each column and create a subplot
for i, column in enumerate(columns_to_plot):
    ax = axes[i]
    sns.scatterplot(
        data=df_cori,
        x="injection_time",
        y=column,
        hue="name",
        ax=ax,
        legend=(i == 0),  # Only display legend for the first subplot
    )
    ax.set_title(f"{column.replace('_', ' ').title()}")
    ax.set_xlabel("Injection Time")
    ax.set_ylabel(column.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)


# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("Lonsdale-2_Subplots_Single_Legend.png")
plt.show()
