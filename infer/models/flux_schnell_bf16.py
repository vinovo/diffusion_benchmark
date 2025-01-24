import torch
from diffusers import FluxPipeline
from .model_inference import ModelInterface

class FluxSchnellBF16(ModelInterface):
    def __init__(self, seed=42):
        from diffusers import FluxPipeline
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        ).to("cuda")
        # self.pipeline.enable_model_cpu_offload()
        self.seed = seed

    def infer(self, prompt: str):
        generator = torch.Generator("cuda").manual_seed(self.seed)
        return self.pipeline(prompt, num_inference_steps=4, guidance_scale=1.0, generator=generator).images[0]