import torch
from diffusers import FluxPipeline
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel
from .model_inference import ModelInterface

class FluxSchnellW4A4(ModelInterface):
    def __init__(self, seed=42):
        from diffusers import FluxPipeline
        from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            "mit-han-lab/svdq-int4-flux.1-schnell"
        )
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda")
        # self.pipeline.enable_model_cpu_offload()
        self.seed = seed

    def infer(self, prompt: str):
        generator = torch.Generator("cuda").manual_seed(self.seed)
        return self.pipeline(prompt, num_inference_steps=4, guidance_scale=1.0, generator=generator).images[0]