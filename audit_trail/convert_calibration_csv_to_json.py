import pandas as pd
import json
from pydash import get

output_file = "./audit_trail/calibration_processing.json"
# calibration_data = csv_to_json(input_calibration_file)

import json
import pandas as pd

# Load the JSON file
with open("./audit_trail/input_json/processing_methods.json", "r") as f:
    processing_methods = json.load(f)

# Load the CSV file
calibration_data = pd.read_csv("./audit_trail/calibration.csv")


# Function to group calibration data by channel
def group_calibrations_by_channel(df):
    grouped = df.groupby("channel")
    result = {}
    for channel, group in grouped:
        result[channel] = group[["area", "amount"]].to_dict("records")
    return result


# Iterate over the peak_identification array and update the calibration data
for peak in processing_methods[0]["peak_identification"]:
    compound_name = peak["name"]
    # Filter the calibration data for the current compound
    compound_data = calibration_data[calibration_data["cmpd"] == compound_name]

    if not compound_data.empty:
        # Group the calibration data by channel
        grouped_calibrations = group_calibrations_by_channel(compound_data)
        # Update the calibration points in the JSON
        peak["calibration"] = [
            {
                "channel": channel,
                "type": "linear",
                "amount_unit": "umol/mL",
                "points": points,
            }
            for channel, points in grouped_calibrations.items()
        ]

# Save the updated JSON
with open(output_file, "w") as f:
    json.dump(processing_methods, f, indent=4)

print("Updated JSON saved as 'processing_methods_updated.json'")
