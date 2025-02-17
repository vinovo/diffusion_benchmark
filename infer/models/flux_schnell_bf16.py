import torch
from diffusers import FluxPipeline
from .model_inference import ModelInterface

class FluxSchnellBF16(ModelInterface):
    def __init__(self, seed=42):
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        ).to("cuda")
        # self.pipeline.enable_model_cpu_offload()
        self.seed = seed

    def infer(self, prompt: str):
        # Initialize the generator with a seed for reproducibility
        generator = torch.Generator("cuda").manual_seed(self.seed)
        
        # Reset CUDA peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Run the inference
        result = self.pipeline(prompt, num_inference_steps=4, guidance_scale=1.0, generator=generator).images[0]
        
        # Print memory usage during inference
        current_memory_MB = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_memory_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Current memory allocated: {current_memory_MB:.2f} MB")
        print(f"Peak memory allocated during inference: {peak_memory_MB:.2f} MB")
        
        return result
