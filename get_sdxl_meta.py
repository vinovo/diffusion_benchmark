import json
import os

# File paths
metadata_file = "meta_data.json"
output_folder = "output/sdxl-turbo/MJHQ-1024"
filtered_metadata_file = "meta_data_sdxl.json"

# Load the metadata from the original JSON file
with open(metadata_file, 'r') as file:
    metadata = json.load(file)

# Get all file names in the output folder without the .png suffix
existing_files = set(
    os.path.splitext(file_name)[0]
    for file_name in os.listdir(output_folder)
    if file_name.endswith(".png")
)

# Filter the metadata to include only entries with matching file names
filtered_metadata = {
    key: value
    for key, value in metadata.items()
    if key in existing_files
}

# Write the filtered metadata to a new JSON file
with open(filtered_metadata_file, 'w') as file:
    json.dump(filtered_metadata, file, indent=4)

print(f"Filtered metadata saved to {filtered_metadata_file}")
