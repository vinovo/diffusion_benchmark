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
    parser.add_argument("--model", choices=["FluxSchnell", "FluxDev", "FluxSchnellSD", "FluxDevSD", "SDXLTurbo", "SDXLTurboSD"], required=True, help="Specify the model family to use.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base output folder.")
    parser.add_argument("--num_images", type=int, default=200, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--meta_data_file", type=str, default="meta_data.json", help="Path to the metadata JSON file.")
    parser.add_argument("--quant_type", type=str, default="w4a4", help="Type of quantization [w4a4, q40, q2k]")
    parser.add_argument("--offset", type=int, default=0, help="Offset index to start sampling from.")
    args = parser.parse_args()

    is_reference = args.mode == "reference"

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Paths
    json_file_path = args.meta_data_file
    output_dir = os.path.join(args.output_dir, args.mode)
    output_meta_file_path = os.path.join(output_dir, "meta_data.json")

    # Load metadata
    with open(json_file_path, "r") as file:
        meta_data = json.load(file)

    # Convert metadata to list and sample first
    data_list = list(meta_data.items())
    sampled_data = random.sample(data_list, min(args.num_images, len(data_list)))
    
    # Apply offset
    offset = min(args.offset, len(sampled_data))
    sampled_data = sampled_data[offset:]

    # Load model based on arguments and mode
    if args.model == "FluxSchnell":
        model = FluxSchnellBF16(seed=args.seed) if is_reference else FluxSchnellW4A4(seed=args.seed)
    elif args.model == "FluxDev":
        model = FluxDevBF16(seed=args.seed) if is_reference else FluxDevW4A4(seed=args.seed)
    elif args.model == "FluxSchnellSD":
        if is_reference:
            model = FluxSchnellSDFP16(seed=args.seed)
        else:
            if args.quant_type == 'q2k':
                print('using q2k')
                model = FluxSchnellSDQ2K(seed=args.seed)
            else:
                print('using q40')
                model = FluxSchnellSDQ40(seed=args.seed)
    elif args.model == "FluxDevSD":
        if is_reference:
            model = FluxDevSDFP16(seed=args.seed)
        else:
            if args.quant_type == 'q2k':
                print('using q2k')
                model = FluxDevSDQ2K(seed=args.seed)
            else:
                print('using q40')
                model = FluxDevSDQ40(seed=args.seed)
    elif args.model == "SDXLTurbo":
        model = SDXLTurboFP16(seed=args.seed) if is_reference else None
    elif args.model == "SDXLTurboSD":
        model = SDXLTurboSDFP16(seed=args.seed) if is_reference else SDXLTurboSDQ40(seed=args.seed)

    # Run inference
    processed_metadata = run_inference(model, sampled_data, output_dir)

    # Save processed metadata to a JSON file
    with open(output_meta_file_path, "w") as output_meta_file:
        json.dump(processed_metadata, output_meta_file, indent=4)

    print(f"Processed metadata saved to {output_meta_file_path}")
    print("Inference complete. Images and metadata saved.")

if __name__ == "__main__":
    main()
