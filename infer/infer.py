from .models import *
import os
import json
import time
import torch
import argparse
import random

def run_inference(model: ModelInterface, data_tuples, output_dir):
    """Run inference on a set of prompts and save results."""
    os.makedirs(output_dir, exist_ok=True)
    processed_metadata = {}

    for idx, (image_id, info) in enumerate(data_tuples):
        prompt = info["prompt"]
        print(f"Generating image for prompt {idx + 1}: {prompt}")

        # Start timing
        start_time = time.time()

        # Generate the image
        generated_image = model.infer(prompt)

        # End timing
        end_time = time.time()

        # Calculate latency
        latency = end_time - start_time

        # Update metadata with latency
        info["latency"] = latency

        # Save the processed metadata
        processed_metadata[image_id] = info

        # Save the image using the image ID as the filename
        output_path = os.path.join(output_dir, f"{image_id}.png")
        generated_image.save(output_path)
        print(f"Image saved to {output_path}. Latency: {latency:.4f} seconds.")

    return processed_metadata

def main():
    parser = argparse.ArgumentParser(description="Run inference using specified model.")
    parser.add_argument("--mode", choices=["generate", "reference"], required=True, help="Specify whether to generate or reference.")
    parser.add_argument("--model", choices=["FluxSchnell", "FluxDev", "FluxSchnellSD", "FluxDevSD"], required=True, help="Specify the model family to use.")
    parser.add_argument("--output_folder", type=str, default="./output", help="Base output folder.")
    parser.add_argument("--num_images", type=int, default=200, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    is_reference = args.mode == "reference"

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Paths
    json_file_path = "meta_data.json"
    output_dir = os.path.join(args.output_folder, args.mode)
    output_meta_file_path = os.path.join(output_dir, "meta_data.json")

    # Load metadata
    with open(json_file_path, "r") as file:
        meta_data = json.load(file)

    # Sample specified number of items
    data_tuples = random.sample(list(meta_data.items()), min(args.num_images, len(meta_data)))

    # Load model based on arguments and mode
    if args.model == "FluxSchnell":
        model = FluxSchnellBF16(seed=args.seed) if is_reference else FluxSchnellW4A4(seed=args.seed)
    elif args.model == "FluxDev":
        model = FluxDevBF16(seed=args.seed) if is_reference else FluxDevW4A4(seed=args.seed)
    elif args.model == "FluxSchnellSD":
        model = FluxSchnellSDFP16(seed=args.seed) if is_reference else FluxSchnellSDQ40(seed=args.seed)
    elif args.model == "FluxDevSD":
        model = FluxDevSDFP16(seed=args.seed) if is_reference else FluxDevSDQ40(seed=args.seed)

    # Run inference
    processed_metadata = run_inference(model, data_tuples, output_dir)

    # Save processed metadata to a JSON file
    with open(output_meta_file_path, "w") as output_meta_file:
        json.dump(processed_metadata, output_meta_file, indent=4)

    print(f"Processed metadata saved to {output_meta_file_path}")
    print("Inference complete. Images and metadata saved.")

if __name__ == "__main__":
    main()
