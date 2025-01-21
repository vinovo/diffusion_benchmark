import os
import json
import time
import torch
from diffusers import FluxPipeline
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

# Path to the JSON file (ensure it exists in your working directory)
json_file_path = "meta_data.json"
output_meta_file_path = "./output/generated/meta_data.json"

# Set a fixed random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)

# Load the metadata from the JSON file
with open(json_file_path, "r") as file:
    meta_data = json.load(file)

# Extract the first 5 items (image ID and prompt) as tuples
data_tuples = [(image_id, info) for idx, (image_id, info) in enumerate(meta_data.items()) if idx < 5]

# Load the quantized Nunchaku model and pipeline
transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-schnell")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Create output directory
output_dir = "./output/generated"
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionary to store processed metadata
processed_metadata = {}

# Run inference for each prompt, record latency, and save images
for idx, (image_id, info) in enumerate(data_tuples):
    prompt = info["prompt"]
    print(f"Generating image for prompt {idx + 1}: {prompt}")

    # Start timing
    start_time = time.time()

    # Generate the image with the fixed random seed
    generator = torch.Generator("cuda").manual_seed(random_seed)  # Fixed seed for reproducibility
    generated_image = pipeline(prompt, num_inference_steps=4, guidance_scale=0, generator=generator).images[0]

    # End timing
    end_time = time.time()

    # Calculate latency
    latency = end_time - start_time

    # Update metadata with latency
    info["latency"] = latency

    # Add to processed metadata
    processed_metadata[image_id] = info

    # Save the image using the image ID as the filename
    output_path = os.path.join(output_dir, f"{image_id}.png")
    generated_image.save(output_path)
    print(f"Image saved to {output_path}. Latency: {latency:.4f} seconds.")

# Save processed metadata to a JSON file
with open(output_meta_file_path, "w") as output_meta_file:
    json.dump(processed_metadata, output_meta_file, indent=4)

print(f"Processed metadata saved to {output_meta_file_path}")
print("Inference complete. Images and metadata saved.")
