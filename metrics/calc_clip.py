import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for progress tracking
from itertools import islice
import gc


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Calculate CLIP scores for images.")
parser.add_argument("--mode", choices=["reference", "generate"], required=True, help="Specify the mode: 'reference' or 'generate'.")
parser.add_argument("--input_folder", default="./output/", help="Optionally specify an input folder. Defaults to './output/'.")
parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count()), help="Number of parallel workers (default: number of CPU cores).")
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

# Function to process an individual image
def process_image(image_key, meta):
    image_path = os.path.join(input_folder, f"{image_key}.png")
    prompt = meta["prompt"]

    if not os.path.exists(image_path):
        return image_key, None, f"Image {image_path} not found. Skipping..."

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Compute CLIP score
        score = clip_score_fn(image_tensor, [prompt])  # Pass as a list for batch processing
        return image_key, float(score.item()), None
    except Exception as e:
        return image_key, None, f"Error processing {image_key}: {e}"

# Parallel processing of images with progress tracking
results = {}
total_score = 0
count = 0

def batch_iterator(iterable, batch_size):
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

batch_size = 1000  # Process 1000 images at a time

for batch in batch_iterator(metadata.items(), batch_size):
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_image, key, meta): key for key, meta in batch}
        
        with tqdm(total=len(batch), desc="Processing Images", unit="img") as pbar:
            for future in as_completed(futures):
                image_key, score_value, error_msg = future.result()
                if score_value is not None:
                    results[image_key] = score_value
                    total_score += score_value
                    count += 1
                if error_msg:
                    print(error_msg)
                pbar.update(1)

    gc.collect()  # Free memory between batches

# Calculate aggregated CLIP score
aggregated_score = total_score / count if count > 0 else 0
results["aggregated_score"] = aggregated_score

# Save results to a JSON file
output_file = os.path.join(input_folder, "clip_scores.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# Print aggregated score
print(f"\nAggregated CLIP Score: {aggregated_score}")
print(f"Scores saved to {output_file}")
