import os, json
import pandas as pd
from pydash import get

# data = []
# root_dir = "./output"

# for subdir, dirs, files in os.walk(root_dir):
#     for file in files:
#         if file.endswith(".json"):
#             file_path = os.path.join(subdir, file)
#             with open(file_path, "r") as f:
#                 json_data = json.load(f)
#             system_dict = {
#                 "system_name": get(json_data, "systems.0.name"),
#                 "column_serial_number": get(
#                     json_data, "systems.0.column.serial_number"
#                 ),
#                 "column_injection_count": get(
#                     json_data, "systems.0.column.injection_count"
#                 ),
#                 "user": get(json_data, "users.0.name"),
#                 "instrument_method": get(json_data, "methods.0.injection.name"),
#                 "processing_method": get(json_data, "methods.0.processing.name"),
#                 "injection_name": get(json_data, "samples.0.name"),
#                 "injection_time": get(json_data, "runs.0.injection_time"),
#                 "sequence_name": get(json_data, "runs.0.sequence.name"),
#             }

#             peaks = get(json_data, "results.0.peaks")
#             found_peak = {}
#             for peak in peaks:
#                 if get(peak, "name") == "ibuprofen":
#                     found_peak = peak
#                     break

#             data.append({**system_dict, **found_peak})

# df = pd.DataFrame.from_dict(data)
# df.to_csv("./column_testing.csv", index=False)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame
df = pd.read_csv("column_testing.csv")

# Convert 'injection_time' to datetime
df["injection_time"] = pd.to_datetime(df["injection_time"])

# Get the list of unique system names
system_names = df["system_name"].unique()

# Set up the subplots in a 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

# Determine the global y-axis limits
y_min = df["resolution_EP"].min()
y_max = df["resolution_EP"].max()

# Iterate over system names and create a subplot for each
for i, system_name in enumerate(system_names):
    ax = axes[i]
    sns.scatterplot(
        data=df[df["system_name"] == system_name],
        x="injection_time",
        y="resolution_EP",
        hue="column_serial_number",
        ax=ax,
    )
    ax.set_title(f"Resolution over Time - {system_name}")
    ax.set_xlabel("Injection Time")
    ax.set_ylabel("Resolution EP")
    ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits
    ax.legend(title="Column Serial Number", loc="lower left", bbox_to_anchor=(0, 0))
    ax.tick_params(axis="x", rotation=45)

# Hide any unused subplots (if there are fewer than 8 system names)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
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
    "resolution_USP",
    "plates_USP",
]

# Filter the data for system "Cori"
df_cori = df[df["system_name"] == "Cori"]

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
        hue="column_serial_number",
        ax=ax,
        legend=(i == 0),  # Only display legend for the first subplot
    )
    ax.set_title(f"{column.replace('_', ' ').title()}")
    ax.set_xlabel("Injection Time")
    ax.set_ylabel(column.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)

# Place the legend outside the subplots, only once
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Column Serial Number",
    loc="lower left",
    bbox_to_anchor=(0, 0),
)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("Cori_System_Subplots_Single_Legend.png")
plt.show()
