import os, json
import pandas as pd
from pydash import get

data = []
root_dir = "./output"

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(subdir, file)
            with open(file_path, "r") as f:
                json_data = json.load(f)
            system_dict = {
                "system_name": get(json_data, "systems.0.name"),
                "column_serial_number": get(
                    json_data, "systems.0.column.serial_number"
                ),
                "column_injection_count": get(
                    json_data, "systems.0.column.injection_count"
                ),
                "user": get(json_data, "users.0.name"),
                "instrument_method": get(json_data, "methods.0.injection.name"),
                "processing_method": get(json_data, "methods.0.processing.name"),
                "injection_name": get(json_data, "samples.0.name"),
                "injection_time": get(json_data, "runs.0.injection_time"),
                "sequence_name": get(json_data, "runs.0.sequence.name"),
            }

            peaks = get(json_data, "results.0.peaks")
            found_peak = {}
            for peak in peaks:
                if get(peak, "name") == "ibuprofen":
                    found_peak = peak
                    break

            data.append({**system_dict, **found_peak})

df = pd.DataFrame.from_dict(data)
# df.to_csv("./column_testing.csv", index=False)
print(df.head())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data)

# Convert 'injection_time' to datetime
df["injection_time"] = pd.to_datetime(df["injection_time"])

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="injection_time",
    y="resolution_EP",
    hue="column_serial_number",
    marker="o",
)

# Set plot labels and title
plt.xlabel("Injection Time")
plt.ylabel("Resolution EP")
plt.title("Resolution over Time by Column Serial Number")
plt.xticks(rotation=45)

plt.show()
