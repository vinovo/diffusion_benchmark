import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Calculate CLIP scores for images.")
parser.add_argument("--mode", choices=["reference", "generate"], required=True, help="Specify the mode: 'reference' or 'generate'.")
parser.add_argument("--input_folder", default="./output/", help="Optionally specify an input folder. Defaults to './output/'.")
args = parser.parse_args()

# Set the folder path based on mode
input_folder = os.path.join(args.input_folder, "reference" if args.mode == "reference" else "generate")
metadata_file = os.path.join(input_folder, "meta_data.json")

# Load metadata
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Initialize CLIP score function
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to a fixed size
    transforms.ToTensor(),          # Convert to tensor
    lambda x: (x * 255).byte()      # Scale to 8-bit (0–255)
])

# Initialize variables for storing results
results = {}
total_score = 0
count = 0

# Iterate through metadata
for image_key, meta in metadata.items():
    image_path = os.path.join(input_folder, f"{image_key}.png")
    prompt = meta["prompt"]

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Skipping...")
        continue

    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Calculate CLIP score
    try:
        score = clip_score_fn(image_tensor, [prompt])  # Pass as a list for batch processing
        score_value = float(score.item())  # Extract scalar value
        results[image_key] = score_value
        total_score += score_value
        count += 1
    except Exception as e:
        print(f"Error processing {image_key}: {e}")

# Calculate aggregated CLIP score
aggregated_score = total_score / count if count > 0 else 0

# Add the aggregated score to the results
results["aggregated_score"] = aggregated_score

# Save results to a JSON file
output_file = os.path.join(input_folder, "clip_scores.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# Print aggregated score
print(f"Aggregated CLIP Score: {aggregated_score}")
print(f"Scores saved to {output_file}")
