import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.functional import peak_signal_noise_ratio

# Set input folders and output folder
generated_folder = "./output/generate"
reference_folder = "./output/reference"
output_metrics_folder = "./output/metrics"
os.makedirs(output_metrics_folder, exist_ok=True)

# Load metadata
metadata_file = os.path.join(generated_folder, "meta_data.json")
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to a fixed size
    transforms.ToTensor()           # Convert to tensor
])

# Initialize variables for storing results
results = {}
total_psnr = 0
count = 0

# Iterate through metadata
for image_key, meta in metadata.items():
    generated_image_path = os.path.join(generated_folder, f"{image_key}.png")
    reference_image_path = os.path.join(reference_folder, f"{image_key}.png")

    # Check if both images exist
    if not os.path.exists(generated_image_path) or not os.path.exists(reference_image_path):
        print(f"Missing pair for {image_key}. Skipping...")
        continue

    # Open and preprocess the images
    generated_image = Image.open(generated_image_path).convert("RGB")
    reference_image = Image.open(reference_image_path).convert("RGB")

    generated_tensor = transform(generated_image).unsqueeze(0)  # Add batch dimension
    reference_tensor = transform(reference_image).unsqueeze(0)  # Add batch dimension

    # Calculate PSNR using TorchMetrics
    try:
        psnr_score = peak_signal_noise_ratio(generated_tensor, reference_tensor, data_range=1.0)
        psnr_value = psnr_score.item()
        results[image_key] = psnr_value
        total_psnr += psnr_value
        count += 1
    except Exception as e:
        print(f"Error processing {image_key}: {e}")

# Calculate aggregated PSNR (mean PSNR)
aggregated_psnr = total_psnr / count if count > 0 else 0

# Add the aggregated PSNR to the results
results["aggregated_psnr"] = aggregated_psnr

# Save results to a JSON file
output_file = os.path.join(output_metrics_folder, "psnr_scores.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# Print aggregated PSNR
print(f"Aggregated PSNR: {aggregated_psnr}")
print(f"Scores saved to {output_file}")
