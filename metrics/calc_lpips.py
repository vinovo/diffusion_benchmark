import os
import json
import torch
from torchvision import transforms
from PIL import Image
from lpips import LPIPS  # Assuming LPIPS is installed

# Set input folders and output folder
generated_folder = "./output/generate"
reference_folder = "./output/reference"
output_metrics_folder = "./output/metrics"
os.makedirs(output_metrics_folder, exist_ok=True)

# Load metadata
metadata_file = os.path.join(generated_folder, "meta_data.json")
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Initialize LPIPS model
lpips_model = LPIPS(net="vgg")  # You can use other backbones like 'alex' or 'squeeze'
lpips_model = lpips_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to a fixed size
    transforms.ToTensor(),          # Convert to tensor
    lambda x: (x * 2 - 1)  # Normalize to [-1, 1] for LPIPS
])

# Initialize variables for storing results
results = {}
total_score = 0
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

    generated_tensor = transform(generated_image).unsqueeze(0)
    reference_tensor = transform(reference_image).unsqueeze(0)

    # Move tensors to the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated_tensor = generated_tensor.to(device)
    reference_tensor = reference_tensor.to(device)

    # Calculate LPIPS score
    try:
        lpips_score = lpips_model(generated_tensor, reference_tensor).item()
        results[image_key] = lpips_score
        total_score += lpips_score
        count += 1
    except Exception as e:
        print(f"Error processing {image_key}: {e}")

# Calculate aggregated LPIPS score
aggregated_score = total_score / count if count > 0 else 0

# Add the aggregated score to the results
results["aggregated_score"] = aggregated_score

# Save results to a JSON file
output_file = os.path.join(output_metrics_folder, "lpips_scores.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# Print aggregated score
print(f"Aggregated LPIPS Score: {aggregated_score}")
print(f"Scores saved to {output_file}")
